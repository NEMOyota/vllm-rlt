"""Exercise the engine's KV manager with a real paged NPU op or a CPU contract double."""

import math

import pytest
import torch
from torch.nn import functional as F

from vllm_rlt.attention.backend import AttentionBackend, BackendCapabilities
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.device import resolve_device
from vllm_rlt.kernels.ascend_attention import AscendAttention, AscendPagedPrefill

# Absolute starting position and query length, in packed request order.
SCENARIOS = [
    ("c4_h1", [(128, 16), (0, 144), (0, 144), (0, 144)], True),
    ("all_suffix", [(96, 32)] * 4, True),
    ("cold_ragged", [(0, 15), (0, 16), (0, 17)], True),
    ("mixed_q_and_kv", [(128, 16), (0, 96), (160, 32)], True),
    ("one_token_suffix", [(16, 1), (32, 17), (0, 31)], True),
    ("long_suffix", [(2048, 16), (64, 16)], True),
    ("cold_uniform", [(0, 32)] * 3, False),
    ("single_suffix", [(16, 7)], False),
    ("decode", [(16, 1), (32, 1)], False),
]


def dense(q, k, v, start):
    # Full original request in FP32, independent of the paged operator metadata.
    causal = torch.arange(len(k))[None, :] <= (start + torch.arange(len(q)))[:, None]
    return F.scaled_dot_product_attention(
        q.float().transpose(0, 1),
        k.float().transpose(0, 1),
        v.float().transpose(0, 1),
        attn_mask=causal,
    ).transpose(0, 1)


def cpu_paged(q, keys, values, **args):
    assert args["input_layout"] == "TND" and args["block_size"] == 16
    assert args["num_heads"] == args["num_key_value_heads"] == 16
    assert args["sparse_mode"] == 3 and args["inner_precise"] == 0
    assert args["scale"] == 128**-0.5
    mask = args["atten_mask"]
    assert mask.shape == (2048, 2048) and mask.dtype == torch.bool
    assert mask[0, 1] and not mask[1, 0] and not mask[2, 2]
    outputs = []
    begin = 0
    for end, kv, table in zip(
        args["actual_seq_lengths"], args["actual_seq_lengths_kv"], args["block_table"], strict=True
    ):
        pages = table[: math.ceil(kv / 16)].long()
        k = keys[pages].view(-1, 16, 128)[:kv]
        v = values[pages].view(-1, 16, 128)[:kv]
        outputs.append(dense(q[begin:end], k, v, kv - (end - begin)))
        begin = end
    assert begin == len(q)
    return torch.cat(outputs).to(q.dtype), None


@pytest.mark.parametrize("device_name", ["cpu", pytest.param("npu:0", marks=pytest.mark.npu)])
@pytest.mark.parametrize("case,spans,fuses", SCENARIOS, ids=[s[0] for s in SCENARIOS])
@torch.inference_mode()
def test_varlen_pool_attention(device_name, case, spans, fuses):
    device = resolve_device(device_name)
    if device_name == "cpu":
        operator = cpu_paged
    else:
        import torch_npu

        torch.npu.set_device(device)
        operator = torch_npu.npu_fused_infer_attention_score

    def checked_operator(q, k, v, **kwargs):
        # Full pool aliases, including the second layer; no layer repack or gather.
        assert k.data_ptr() == cache.key_cache.data_ptr()
        assert v.data_ptr() == cache.value_cache.data_ptr()
        assert kwargs["block_table"].dtype == torch.int32
        return operator(q, k, v, **kwargs)

    strategy = AscendPagedPrefill(checked_operator)
    backend = AttentionBackend(
        AscendAttention(),
        BackendCapabilities(packed_prefill=True),
        {"selected_backend": "ascend_eager"},
        strategy,
    )
    layers, loops = 2, 4
    blocks = loops * sum(math.ceil((s + n) / 16) for s, n in spans)
    cache = KVCacheManager(
        layers, 16, 128, blocks, 16, loops, device=device, dtype=torch.bfloat16, attention=backend
    )
    generator = torch.Generator().manual_seed(421559278)

    def tensor(rows):
        return torch.randn(rows, 16, 128, generator=generator).to(torch.bfloat16)

    query = tensor(sum(n for _, n in spans))
    ids, depths, positions = [], [], []
    # Use separate logical sequences and recurrence planes across the batch.
    for i, (start, n) in enumerate(spans):
        rid = str(i)
        assert cache.allocate(rid, start + n)
        ids.extend([rid] * n)
        depths.extend([i % loops] * n)
        positions.extend(range(start, start + n))
    batch = cache._prepare_batch(ids, depths, positions, for_write=False, packed_prefill=True)
    with pytest.raises(RuntimeError, match="uninitialized KV history"):
        cache._attend_prepared(0, batch, query.to(device))
    assert strategy.calls == 0
    for layer in range(layers):
        references = []
        begin = 0
        for i, (start, n) in enumerate(spans):
            kv = start + n
            k, v = tensor(kv), tensor(kv)
            cache.write(
                layer, [str(i)] * kv, [i % loops] * kv, list(range(kv)), k.to(device), v.to(device)
            )
            references.append(dense(query[begin : begin + n], k, v, start))
            begin += n
        out = cache._attend_prepared(layer, batch, query.to(device))
        torch.testing.assert_close(out.cpu().float(), torch.cat(references), atol=0.016, rtol=0.02)
    assert strategy.calls == (layers if fuses else 0)
    assert strategy.plans == int(fuses)
    cache.free("0")
    with pytest.raises(RuntimeError, match="stale"):
        cache._attend_prepared(0, batch, query.to(device))
    assert strategy.calls == (layers if fuses else 0)


@pytest.fixture
def cpu_ouro_engine(monkeypatch):
    # Keep the real scheduler, prompt publication, four loops, model layers and decode.
    # Only emulate the device selection and the BF16 FIA operator.
    from dataclasses import replace

    from vllm_rlt.engine import stage_engine
    from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

    model = (
        OuroForCausalLM(
            OuroConfig(hidden_size=32, intermediate_size=32, num_hidden_layers=1, vocab_size=128)
        )
        .to(torch.bfloat16)
        .eval()
    )

    def build(model, *, execution_config, **kwargs):
        strategy = AscendPagedPrefill(cpu_paged) if execution_config.ascend_paged_prefill else None
        backend = AttentionBackend(
            AscendAttention(),
            BackendCapabilities(packed_prefill=True),
            {"selected_backend": "ascend_eager"},
            strategy,
        )
        monkeypatch.setattr(stage_engine, "create_backend", lambda *args, **kw: backend)
        monkeypatch.setattr(stage_engine, "packed_prefill_eligible", lambda *args, **kw: True)
        return stage_engine.StageEngine(
            model, execution_config=replace(execution_config, ascend_paged_prefill=False), **kwargs
        )

    return model, build


@pytest.mark.parametrize(
    "case",
    ["c4_h1", "mixed", "one_token_suffix", "cold_uniform", "single", "w2_shape", "long_suffix"],
)
@torch.inference_mode()
def test_ouro_scheduler_to_paged_prefill_on_cpu(case, monkeypatch, cpu_ouro_engine):
    from benchmarks import ascend_varlen_prefill as bench

    model, build = cpu_ouro_engine
    monkeypatch.setattr(bench, "StageEngine", build)
    params = bench.SamplingParams(
        temperature=0, min_loops=4, max_loops=4, max_tokens=2, exit_threshold=1.0, ignore_eos=True
    )
    old = bench.run_arm(model, case, 797543261, "legacy", params)
    new = bench.run_arm(model, case, 797543261, "paged", params)
    assert old["issues"] == new["issues"] == []
    assert bench.compare(old, new)["same_inputs"]
    assert old["prefill_sdpa_calls"] > 0
    if new["fia_calls"]:
        assert new["prefill_sdpa_calls"] == 0
        assert len(new["prefill_batches"]) == 1
    if case == "c4_h1":
        off = bench.run_arm(model, case, 797543261, "off", params)
        assert off["issues"] == [] and off["hits"] == [0] * 4
        assert off["input_sha256"] == new["input_sha256"]
        assert old["prefill_sdpa_calls"] == 4 * new["fia_calls"]
    assert old["output_tok_s"] > 0 and new["ttft_p50_ms"] > 0


@pytest.mark.parametrize("changes", [{"page": 128}, {"packed": False}, {"streams": True}])
def test_invalid_paged_prefill_configuration(changes):
    from tests.test_device import npu_model_stub
    from vllm_rlt import CacheConfig, ExecutionConfig
    from vllm_rlt.engine.stage_engine import StageEngine

    model = npu_model_stub(num_attention_heads=16, num_key_value_heads=16, head_dim=128)
    with pytest.raises(ValueError, match="Ascend paged prefill"):
        StageEngine(
            model,
            attention_backend="ascend_eager",
            cache_config=CacheConfig(block_size=changes.get("page", 16), num_blocks=8),
            execution_config=ExecutionConfig(
                ascend_paged_prefill=True,
                multi_stream=changes.get("streams", False),
                packed_prefill=changes.get("packed", True),
            ),
        )


@pytest.mark.parametrize("hot", [0, 1, 4])
@torch.inference_mode()
def test_prefix_benefit_same_backend_queued_and_chunked(hot, monkeypatch, cpu_ouro_engine):
    from benchmarks import ascend_prefix_benefit as bench

    model, build = cpu_ouro_engine
    monkeypatch.setattr(bench, "StageEngine", build)
    shape = bench.Workload(length=64, outputs=2, clients=4, residents=2, prefix=48, hot_clients=hot)
    off, on = [bench.measure_arm(model, shape, 797543261, 32, enabled) for enabled in (False, True)]
    assert off["ascend_paged_prefill"] and on["ascend_paged_prefill"]
    assert off["num_blocks"] == on["num_blocks"]
    assert off["prefill_rows"] == 256 and on["prefill_rows"] == 256 - hot * 48
    assert off["hit_tokens"] == 0 and on["hit_tokens"] == hot * 48
    assert off["output_tokens"] == on["output_tokens"] == 8
    pair = bench.paired(off, on)
    both = bench.summary([dict(off=off, on=on, pair=pair)])
    assert both["on_hit_fraction"] == hot * 48 / 256
    assert pair["on_with_primer_vs_off_pct"] < pair["output_tok_s_change_pct"]
