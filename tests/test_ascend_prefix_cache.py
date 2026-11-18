"""Synchronous paged Ouro prefix reuse, including a live shared allocation."""

import pytest
import torch

from benchmarks.ascend_prefix_cache import experiment
from tests.helpers import tiny_ouro_config, tiny_ouro_model
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestPhase


def test_prefix_experiment_compares_the_same_target():
    report = experiment(tiny_ouro_model(dtype=torch.bfloat16), "torch", 33, 4, 1)
    assert report["status"] == "PASS", report
    assert report["hit_tokens"] == report["prefill_start"] == 32
    assert all(ms > 0 for ms in report["ttft_p50_ms"].values())


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu:0", marks=pytest.mark.npu)])
@torch.inference_mode()
def test_prefix_pages_survive_donor_cancellation(device):
    device = resolve_device(device)
    if device.type == "npu":
        torch.npu.set_device(0)
    torch.manual_seed(41)
    config = tiny_ouro_config(hidden_size=256, intermediate_size=512, head_dim=64)
    model = OuroForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    engine = StageEngine(
        model,
        attention_backend="ascend_eager" if device.type == "npu" else "torch",
        cache_config=CacheConfig(num_blocks=32, block_size=16, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64),
    )
    cache = engine.cache_manager
    prefix = list(range(1, 33))
    params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    try:
        engine.add_request("donor", prefix + [35, 36, 37], params)
        engine.step()
        donor = cache.plane_block_tables("donor")
        prefix_pages = [p for plane in donor for p in plane[:2]]
        assert len(set(prefix_pages)) == 8  # Two pages at each of four depths.
        pages = torch.tensor(prefix_pages, dtype=torch.long, device=device)
        keys = cache.key_cache.index_select(0, pages).cpu()
        values = cache.value_cache.index_select(0, pages).cpu()

        engine.add_request("target", prefix + [45, 46, 47], params)
        while not cache.has_allocation("target"):
            engine.step()
        assert engine.last_schedule.phase == RequestPhase.PREFILL
        (span,) = engine.last_schedule.requests
        assert (span.request.request_id, span.token_start, span.token_count) == ("target", 32, 3)
        assert cache.prefix_hits == 32
        for a, b in zip(donor, cache.plane_block_tables("target"), strict=True):
            assert a[:2] == b[:2]
            assert a[2] != b[2]
        assert engine.abort_request("donor").finished
        finished = []
        while engine.has_unfinished_requests():
            finished.extend(out for out in engine.step() if out.finished)
        assert [out.request_id for out in finished] == ["target"]
        assert finished[0].exit_depths == [4] * params.max_tokens
        assert torch.equal(cache.key_cache.index_select(0, pages).cpu(), keys)
        assert torch.equal(cache.value_cache.index_select(0, pages).cpu(), values)
        assert cache.num_used_blocks == 0
    finally:
        engine.close()
