"""Configuration and page reuse through the supported Ouro engine entry point."""

import os
from dataclasses import replace

import pytest
import torch

from benchmarks.ascend_main_prefix_linear import (
    benchmark_dtype,
    check_linear_usage,
    linear_usage,
    padded_fp32_linears,
)
from tests.test_device import npu_model_stub
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.device import resolve_device
from vllm_rlt.engine import stage_engine
from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.models import OuroForCausalLM

CACHE = CacheConfig(block_size=16, num_blocks=64, enable_prefix_caching=True)
SYNC = ExecutionConfig(multi_stream=False, use_uva=False)


def engine(model, **changes):
    options = dict(cache_config=CACHE, execution_config=SYNC, attention_backend="ascend_eager")
    return stage_engine.StageEngine(model, **(options | changes))


@pytest.mark.parametrize(
    "changes",
    [
        dict(attention_backend="torch"),
        dict(execution_config=replace(SYNC, async_scheduling=True)),
        dict(execution_config=replace(SYNC, multi_stream=True)),
        dict(execution_config=replace(SYNC, static_buffers=True)),
        dict(execution_config=replace(SYNC, cuda_graphs=True)),
        dict(cache_config=replace(CACHE, incremental_allocation=True)),
        dict(speculative_config=SpeculativeConfig(num_speculative_tokens=2)),
        dict(cache_config=replace(CACHE, kv_cache_dtype="fp8_e4m3")),
        dict(scheduler_config=SchedulerConfig(wavefront_chunked_prefill=True)),
        dict(scheduler_config=SchedulerConfig(mode="mixed")),
        dict(scheduler_config=SchedulerConfig(enable_preemption=True)),
        dict(exit_config=ExitConfig(mode="ouro_delayed")),
        dict(stage_config=StageConfig(role="prefill")),
    ],
)
def test_unsupported_prefix_configuration_rejected_before_backend(monkeypatch, changes):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported prefix configuration reached backend allocation")

    monkeypatch.setattr(stage_engine, "create_backend", unexpected)
    with pytest.raises(ValueError, match="(?i)NPU|Ascend|prefix cach"):
        engine(npu_model_stub(), **changes)


def test_prefix_cache_requires_last_exited_kv():
    with pytest.raises(ValueError, match="last_exited"):
        replace(CACHE, layout="shared")


@pytest.mark.parametrize("packed", [None, True])
def test_supported_prefix_configuration_reaches_packed_backend(monkeypatch, packed):
    class BackendReached(Exception):
        pass

    def select(name, device, dtype, *args, prefer_packed_prefill, **kwargs):
        assert (name, device.type, dtype) == ("ascend_eager", "npu", torch.bfloat16)
        assert prefer_packed_prefill
        raise BackendReached

    monkeypatch.setattr(stage_engine, "create_backend", select)
    with pytest.raises(BackendReached):
        engine(npu_model_stub(), execution_config=replace(SYNC, packed_prefill=packed))


def generate(runtime, tokens):
    params = SamplingParams(
        max_tokens=4, temperature=0, min_loops=4, max_loops=4, exit_threshold=1.0, ignore_eos=True
    )
    hits_before = runtime.cache_manager.prefix_hits
    runtime.add_request("reused-id", tokens, params)
    finished = []
    while runtime.has_unfinished_requests():
        finished.extend(out for out in runtime.step() if out.finished)
    assert len(finished) == 1
    out = finished[0]
    assert len(out.token_ids) == 4 and out.exit_depths == [4] * 4 and out.finish_reason == "length"
    assert runtime.cache_manager.num_used_blocks == 0
    return list(out.token_ids), runtime.cache_manager.prefix_hits - hits_before


@pytest.fixture(scope="module")
def ouro_checkpoint():
    folder = os.environ.get("VLLM_RLT_OURO_CHECKPOINT")
    if not folder:
        pytest.skip("set VLLM_RLT_OURO_CHECKPOINT to local Ouro weights")
    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    model = OuroForCausalLM.from_pretrained(folder, device=device, dtype=benchmark_dtype()).eval()
    rows = os.environ.get("RLT_BENCH_LINEAR_FP32_CHUNK")
    with padded_fp32_linears(model, int(rows) if rows else None) as work:
        yield model
        if work is not None:
            check_linear_usage(linear_usage(work))


@pytest.mark.npu
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "per-row"])
@torch.inference_mode()
def test_released_donor_reuses_one_full_page(ouro_checkpoint, packed):
    policy = os.environ.get("RLT_PR1_UNPACKED", "support")
    assert policy in ("support", "reject")
    execution = replace(SYNC, packed_prefill=packed)
    if not packed and policy == "reject":
        with pytest.raises(ValueError, match="(?i)pack"):
            engine(ouro_checkpoint, execution_config=execution).close()
        return

    # Same first page, distinct suffixes, followed by a different private page.
    shared = list(range(10, 26))
    prompts = [shared + [41, 42], shared + [43, 44], list(range(100, 118)), shared + [45, 46]]
    reference = []
    for enabled in (False, True):
        runtime = engine(
            ouro_checkpoint,
            cache_config=replace(CACHE, enable_prefix_caching=enabled),
            execution_config=execution,
        )
        try:
            assert runtime.execution_config.packed_prefill is packed
            for index, tokens in enumerate(prompts):
                if enabled and index == 1:
                    before = runtime.cache_manager.prefix_hits
                    runtime.add_request(
                        "cancelled", tokens, SamplingParams(min_loops=4, max_loops=4)
                    )
                    runtime.step()
                    assert runtime.cache_manager.prefix_hits - before == 16
                    assert runtime.abort_request("cancelled").finish_reason == "abort"
                    assert runtime.cache_manager.num_used_blocks == 0
                selected, hits = generate(runtime, tokens)
                assert hits == (16 if enabled and index in (1, 3) else 0)
                if enabled:
                    assert selected == reference[index], (index, selected, reference[index])
                else:
                    reference.append(selected)
        finally:
            runtime.close()


@pytest.mark.npu
def test_invalid_sampling_does_not_admit_cached_request(ouro_checkpoint):
    runtime = engine(ouro_checkpoint)
    try:
        for params in (
            SamplingParams(temperature=1),
            SamplingParams(max_loops=3),
            SamplingParams(exit_threshold=0.5),
        ):
            with pytest.raises(ValueError, match="(?i)greedy|four loops|threshold"):
                runtime.add_request("invalid", [10, 11], params)
            assert runtime.stats() == {"requests": 0, "used_blocks": 0}
    finally:
        runtime.close()
