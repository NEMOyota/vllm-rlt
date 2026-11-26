"""Configuration and page reuse through the supported Ouro engine entry point."""

import json
import os
import random
from dataclasses import replace

import pytest
import torch

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
from vllm_rlt.request import RequestPhase

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


def generate(runtime, tokens, max_tokens=4):
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
        min_loops=4,
        max_loops=4,
        exit_threshold=1.0,
        ignore_eos=True,
    )
    hits_before = runtime.cache_manager.prefix_hits
    runtime.add_request("reused-id", tokens, params)
    finished = []
    rows = 0
    while runtime.has_unfinished_requests():
        finished.extend(out for out in runtime.step() if out.finished)
        batch = runtime.last_schedule
        if batch.phase == RequestPhase.PREFILL:
            rows += sum(item.token_count for item in batch.requests)
    assert len(finished) == 1
    out = finished[0]
    assert len(out.token_ids) == max_tokens and out.exit_depths == [4] * max_tokens
    assert out.finish_reason == "length"
    assert runtime.cache_manager.num_used_blocks == 0
    hit = runtime.cache_manager.prefix_hits - hits_before
    assert rows == len(tokens) - hit
    return list(out.token_ids), hit


@pytest.fixture(scope="module")
def ouro_checkpoint():
    folder = os.environ.get("VLLM_RLT_OURO_CHECKPOINT")
    if not folder:
        pytest.skip("set VLLM_RLT_OURO_CHECKPOINT to local Ouro weights")
    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    return OuroForCausalLM.from_pretrained(folder, device=device, dtype=torch.bfloat16).eval()


@pytest.mark.npu
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "per-row"])
@torch.inference_mode()
def test_released_donor_reuses_one_full_page(ouro_checkpoint, packed):
    execution = replace(SYNC, packed_prefill=packed)

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


@pytest.mark.npu
@torch.inference_mode()
def test_aligned_donor_tail_on_ascend(ouro_checkpoint):
    prefix = list(range(10, 138))
    extended = prefix + list(range(200, 232))
    results = []
    for enabled in (False, True):
        runtime = engine(
            ouro_checkpoint, cache_config=replace(CACHE, enable_prefix_caching=enabled)
        )
        try:
            generate(runtime, prefix)
            same, same_hit = generate(runtime, prefix)
            longer, longer_hit = generate(runtime, extended)
            assert same_hit == (112 if enabled else 0)
            assert longer_hit == (128 if enabled else 0)
            results.append((same, longer))
        finally:
            runtime.close()
    assert results[0] == results[1]


@pytest.mark.npu
@torch.inference_mode()
def test_leaf_eviction_on_ascend(ouro_checkpoint):
    shared = list(range(10, 26))
    cold = shared + list(range(30, 46))
    hot = shared + list(range(50, 66))
    runtime = engine(ouro_checkpoint, cache_config=replace(CACHE, num_blocks=24))
    try:
        generate(runtime, cold)
        generate(runtime, hot)
        cache = runtime.cache_manager
        assert len(cache.lookup_prefix(hot + [9])) == 2
        # Six logical pages. Three retained pages plus this four-page
        # reservation force eviction of exactly one page across four depths.
        generate(runtime, list(range(100, 148)))
        assert (cache.prefix_evictions, cache.prefix_blocks_reclaimed) == (1, 4)
        assert len(cache.lookup_prefix(cold + [9])) == 1
        assert len(cache.lookup_prefix(hot + [9])) == 2
    finally:
        runtime.close()


def test_generated_prefix_requires_supported_runtime():
    from tests.helpers import tiny_ouro_config

    with pytest.raises(ValueError, match="requires enable_prefix_caching"):
        CacheConfig(cache_generated_prefixes=True)
    with pytest.raises(ValueError, match="Ascend Ouro"):
        stage_engine.StageEngine(
            OuroForCausalLM(tiny_ouro_config()),
            cache_config=replace(CACHE, cache_generated_prefixes=True),
        )


def measure_second_turn(runtime, tokens, start, end):
    """Capture only the reusable decode span, and the first CODA logits."""
    cache = runtime.cache_manager
    before = cache.prefix_hits
    logits = []

    def capture_first(_module, _inputs, output):
        if not logits:
            logits.append(output.detach().reshape(-1, output.shape[-1])[0].float().cpu())

    hook = runtime.model.lm_head.register_forward_hook(capture_first)
    try:
        runtime.add_request(
            "second",
            tokens,
            SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True),
        )
        request = runtime.scheduler.requests["second"]
        while request.num_prefilled_tokens < len(tokens):
            runtime.step()
        pages = {}
        for depth in range(cache.storage_depths):
            for layer in range(cache.num_layers):
                k, v = cache.read(layer, "second", depth, end)
                pages[depth, layer] = torch.stack((k[start:], v[start:])).cpu()
        outputs = []
        while runtime.has_unfinished_requests():
            outputs.extend(out for out in runtime.step() if out.finished)
        assert len(outputs) == 1 and outputs[0].exit_depths == [4] * 4
        assert cache.num_used_blocks == 0
        assert logits[0].argmax().item() == outputs[0].token_ids[0]
        return cache.prefix_hits - before, pages, logits[0], outputs[0].token_ids
    finally:
        hook.remove()


@pytest.mark.npu
@pytest.mark.parametrize(
    "seed",
    [
        114281796,
        273728779,
        355234819,
        421559278,
        641794792,
        770929007,
        797543261,
        1179450028,
        1560366951,
        1727668039,
    ],
)
@torch.inference_mode()
def test_generated_prefix_two_turn_on_ascend(ouro_checkpoint, seed, record_property):
    rng = random.Random(seed)
    prompt = rng.choices(range(8, ouro_checkpoint.config.vocab_size), k=128)
    suffix = rng.choices(range(8, ouro_checkpoint.config.vocab_size), k=8)
    # The 64th sampled token has no KV. Only 48 of the 63 forwarded
    # outputs make complete pages. Turn two includes all 64 output IDs.
    expected_hits = {"off": 0, "prompt": 128, "generated": 176}
    history = reference = None
    evidence = {}
    for arm, expected in expected_hits.items():
        runtime = engine(
            ouro_checkpoint,
            cache_config=replace(
                CACHE,
                enable_prefix_caching=arm != "off",
                cache_generated_prefixes=arm == "generated",
            ),
        )
        try:
            donor, donor_hit = generate(runtime, prompt, max_tokens=64)
            assert donor_hit == 0
            if history is None:
                history = prompt + donor + suffix
            assert history == prompt + donor + suffix
            hit, pages, logits, output = measure_second_turn(runtime, history, 128, 176)
            assert hit == expected
            if reference is None:
                reference = pages, logits, output
            ref_pages, ref_logits, _ = reference
            kv_deltas = {
                f"d{depth}/l{layer}": (kv.float() - ref_pages[depth, layer].float())
                .abs()
                .flatten(1)
                .amax(1)
                .tolist()
                for (depth, layer), kv in pages.items()
            }
            top, ids = logits.topk(2)
            evidence[arm] = dict(
                hit=hit,
                output=output,
                top2=ids.tolist(),
                top2_margin=(top[0] - top[1]).item(),
                max_logit_delta=(logits - ref_logits).abs().max().item(),
                kv_max_abs_k_v=kv_deltas,
            )
        finally:
            runtime.close()
    # Preserve differences even on failing seeds. No arbitrary BF16 tolerance
    # is used to turn an output mismatch into a pass.
    record_property("second_turn", json.dumps(evidence, allow_nan=False))
    for arm in ("prompt", "generated"):
        assert evidence[arm]["output"] == evidence["off"]["output"], (seed, arm, evidence[arm])


@pytest.mark.parametrize("end", ["length", "stop", "abort"])
@torch.inference_mode()
def test_completed_generation_lifecycle_on_cpu_reference(end):
    from tests.helpers import tiny_ouro_config
    from vllm_rlt.request import FinishReason

    torch.manual_seed(1)
    model = OuroForCausalLM(tiny_ouro_config()).eval()
    runtime = stage_engine.StageEngine(
        model, cache_config=replace(CACHE, block_size=2), execution_config=SYNC
    )
    # Exercise the same finish/release path with real CPU tensors. Enabling the
    # public option on a non-Ascend engine is tested separately above.
    runtime.cache_generated_prefixes = True
    prompt = [1, 2, 3]
    try:
        runtime.add_request("r", prompt, SamplingParams(max_tokens=4, min_loops=4, max_loops=4))
        request = runtime.scheduler.requests["r"]
        runtime.scheduler.eos_token_ids = ()
        while len(request.generated_token_ids) < 3:
            runtime.step()
        if end == "abort":
            runtime.abort_request("r")
        else:
            if end == "stop":
                runtime.scheduler.eos_token_ids = tuple(range(model.config.vocab_size))
            while runtime.has_unfinished_requests():
                runtime.step()
        assert request.finish_reason == FinishReason(end)
        continuation = prompt + request.generated_token_ids + [6]
        assert len(runtime.cache_manager.lookup_prefix(continuation)) == (
            1 if end == "abort" else 3
        )
        assert runtime.cache_manager.num_used_blocks == 0
        # Immediate ID reuse must see only its own private suffix.
        runtime.add_request("r", [9, 10], SamplingParams(max_tokens=1))
        while runtime.has_unfinished_requests():
            runtime.step()
    finally:
        runtime.close()
