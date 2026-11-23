"""Attention-oracle and dispatch checks for grouped Ascend prefix-cache prefill."""

import argparse

import pytest
import torch

from tests.test_device import npu_model_stub
from vllm_rlt import CacheConfig, ExecutionConfig
from vllm_rlt.attention.metadata import AttentionRows, build_attention_metadata
from vllm_rlt.device import resolve_device
from vllm_rlt.engine import stage_engine
from vllm_rlt.entrypoints.runtime_args import add_runtime_args, runtime_configs
from vllm_rlt.kernels import ascend_attention as ascend
from vllm_rlt.kernels.ascend_prefix_prefill import Span, partition


def test_grouping_config_routes_only_supported_cache_and_packed_backend(monkeypatch):
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    args = parser.parse_args(
        ["--enable-prefix-caching", "--single-stream", "--ascend-prefix-prefill-slack", "0"]
    )
    args.num_blocks, args.block_size = 64, 16
    configs = runtime_configs(args)
    assert configs["execution_config"].ascend_prefix_prefill_slack == 0
    assert configs["cache_config"].enable_prefix_caching
    with pytest.raises(ValueError, match="nonnegative"):
        ExecutionConfig(ascend_prefix_prefill_slack=-1)
    device = next(npu_model_stub().parameters()).device
    real = stage_engine.create_backend(
        "ascend_eager",
        device,
        torch.bfloat16,
        64,
        16,
        prefer_packed_prefill=True,
        ascend_prefix_prefill_slack=4,
    )
    assert isinstance(real.kernel, ascend.AscendAttention) and real.kernel.prefix_prefill_slack == 4

    class Reached(Exception):
        pass

    def select(*args, **kwargs):
        assert kwargs["ascend_prefix_prefill_slack"] == 4 and kwargs["prefer_packed_prefill"]
        raise Reached

    monkeypatch.setattr(stage_engine, "create_backend", select)
    for enabled, packed in ((False, True), (True, False), (True, None)):
        with pytest.raises(Reached if enabled and packed is None else ValueError):
            stage_engine.StageEngine(
                npu_model_stub(),
                attention_backend="ascend_eager",
                cache_config=CacheConfig(enable_prefix_caching=enabled),
                execution_config=ExecutionConfig(
                    multi_stream=False, packed_prefill=packed, ascend_prefix_prefill_slack=4
                ),
            )


def test_partition_bounds_query_context_depth_padding_and_workspace():
    def plan(shapes, slack, size=8):
        spans = [Span(i, 0, depth, start, width) for i, (depth, start, width) in enumerate(shapes)]
        groups, singles = partition(spans, slack, q_row_bytes=size, kv_row_bytes=size)
        assert sorted(s.table for group in [*groups, singles] for s in group) == list(
            range(len(spans))
        )
        return [sorted(s.table for s in g) for g in groups]

    assert plan([(1, 48, 23), (0, 48, 20), (1, 48, 20)], 4) == [[0, 2]]
    assert plan([(1, 48, 23), (1, 48, 20)], 2) == []
    assert plan([(1, 48, 23), (1, 80, 23)], 4) == []
    assert plan([(0, 0, 2), (0, 0, 6)], 4) == []  # Fails the 12.5% work bound.
    assert plan([(0, 48, 20), (0, 48, 20)], 0) == [[0, 1]]
    assert plan([(0, 48, 20), (0, 48, 20)], 0, size=1024**2) == []


class Packed:
    pass


def make_batch(shapes, device, dtype, seed):
    torch.manual_seed(seed)
    block_size, heads, dim = 16, 2, 64
    block_count = 1 + sum((start + width + 15) // 16 for _, start, width in shapes)
    page_ids = iter(torch.randperm(block_count - 1).add(1).tolist())
    # Fill inaccessible slots and inter-layer gaps with NaNs to expose bad gathers.
    kv = torch.full((2, block_count, 2, block_size, heads, dim), float("nan"), dtype=dtype)
    k, v = kv[0, :, 1], kv[1, :, 1]
    shared = torch.randn(2, block_size, heads, dim).to(dtype)
    k[0], v[0] = shared
    batch = Packed()
    rows, tables, positions, keys, queries, expected, write_blocks = [], [], [], [], [], [], []
    for depth, start, width in shapes:
        owner = object()
        context = start + width
        table = tuple(
            0 if j == 0 and start >= block_size else next(page_ids)
            for j in range((context + block_size - 1) // block_size)
        )
        dense_k, dense_v = torch.randn(2, context, heads, dim).to(dtype)
        if table[0] == 0:
            dense_k[:block_size], dense_v[:block_size] = shared
        for j in range(context):
            page, offset = table[j // block_size], j % block_size
            k[page, offset], v[page, offset] = dense_k[j], dense_v[j]
        query = torch.randn(width, heads * 2, dim).to(dtype)
        qf = query.float().transpose(0, 1)
        kf = dense_k.float().repeat_interleave(2, dim=1).transpose(0, 1)
        vf = dense_v.float().repeat_interleave(2, dim=1).transpose(0, 1)
        logits = qf @ kf.transpose(1, 2) / dim**0.5
        future = torch.arange(context) > (start + torch.arange(width))[:, None]
        expected.append((logits.masked_fill(future, -torch.inf).softmax(-1) @ vf).transpose(0, 1))
        queries.append(query)
        for pos in range(start, context):
            rows.append((owner, depth, pos))
            keys.append((id(owner), depth))
            positions.append(pos)
            tables.append(table)
            write_blocks.append(table[pos // block_size])
    meta = build_attention_metadata(
        AttentionRows(tables, positions, keys),
        block_size=block_size,
        device=device,
        packed_prefill=True,
    )
    batch.rows, batch.block_tables, batch.cu_seqlens_q = rows, meta.block_tables, meta.cu_seqlens_q
    batch.write_blocks = torch.tensor(write_blocks, device=device)
    batch.write_offsets = torch.tensor(positions, device=device) % block_size
    # Transfer the full storage, retaining the actual per-layer strides.
    kv = kv.to(device)
    return batch, torch.cat(queries).to(device), kv, torch.cat(expected)


@pytest.mark.parametrize("device_name", ["cpu", pytest.param("npu:0", marks=pytest.mark.npu)])
@pytest.mark.parametrize(
    "shapes,slack,expected_calls,padded",
    [
        ([(0, 48, 23), (0, 0, 9), (0, 48, 20)], 4, 2, 1),
        ([(0, 48, 24), (0, 48, 22), (0, 48, 20), (0, 48, 23)], 4, 1, 1),
        ([(0, 48, 19), (0, 48, 16)], 4, 1, 1),  # Cross a page boundary with a -1 table entry.
        ([(0, 48, 80), (0, 64, 72)], 8, 1, 1),  # Different absolute causal origins.
        ([(0, 0, 23), (0, 0, 21), (1, 32, 1), (1, 48, 1)], 4, 2, 1),
        ([(0, 48, 23), (0, 48, 20)], 0, 2, 0),
        ([(0, 48, 23), (0, 48, 23)], 0, 1, 0),
        ([(0, 0, 21), (0, 0, 21)], 4, 1, 0),  # Whole-grid cold fast path.
    ],
)
@torch.inference_mode()
def test_grouped_sdpa_matches_independent_dense_oracle(
    device_name, shapes, slack, expected_calls, padded, monkeypatch
):
    device = resolve_device(device_name)
    if device.type == "npu":
        torch.npu.set_device(device)
    dtype = torch.bfloat16 if device.type == "npu" else torch.float32
    kernel = ascend.AscendAttention(prefix_prefill_slack=slack)
    calls = []
    attend = ascend._attend

    def recorded(q, k, v, *args, **kwargs):
        calls.append((q.shape, k.shape))
        return attend(q, k, v, *args, **kwargs)

    monkeypatch.setattr(ascend, "_attend", recorded)
    for seed in (168, 169):  # Same geometry with different allocations and physical page tables.
        batch, q, kv, expected = make_batch(shapes, device, dtype, seed)
        original = kv.clone()
        for _ in range(2):  # Descriptor reuse across layers.
            calls.clear()
            output = kernel.attend_metadata(q, kv[0, :, 1], kv[1, :, 1], batch)
            tol = 0.02 if dtype is torch.bfloat16 else 2e-6
            torch.testing.assert_close(output.float().cpu(), expected, atol=tol, rtol=tol)
            assert len(calls) == expected_calls
        torch.testing.assert_close(kv, original, rtol=0, atol=0, equal_nan=True)
    assert kernel.prefix_group_stats["padded_groups"] == 4 * padded
    if padded:
        assert kernel.layout_reuses > 0


def test_no_grouping_on_per_row_tables_or_nonflat_cache_and_ignore_trailing_queries():
    shapes = [(0, 48, 23), (0, 48, 20)]
    batch, q, kv, expected = make_batch(shapes, torch.device("cpu"), torch.float32, 44)
    kernel = ascend.AscendAttention(prefix_prefill_slack=4)
    # Trailing execution-buffer rows have no request, KV write or output owner.
    padded = torch.cat((q, torch.randn_like(q[:5])))
    got = kernel.attend_metadata(padded, kv[0, :, 1], kv[1, :, 1], batch)
    torch.testing.assert_close(got[: len(q)], expected, atol=2e-6, rtol=2e-6)
    assert got[len(q) :].count_nonzero() == 0
    before = dict(kernel.prefix_group_stats)
    # Last-dimension stride 2 forces the ordinary per-request fallback.
    k, v = (torch.stack((x, x), dim=-1)[..., 0] for x in (kv[0, :, 1], kv[1, :, 1]))
    assert ascend._flat_kv_rows(k, v) is None
    torch.testing.assert_close(
        kernel.attend_metadata(q, k, v, batch), expected, atol=2e-6, rtol=2e-6
    )
    # New descriptor has a page table for each query row.
    unpacked = Packed()
    unpacked.rows, unpacked.cu_seqlens_q = batch.rows, None
    unpacked.block_tables = batch.block_tables.repeat_interleave(torch.tensor([23, 20]), dim=0)
    torch.testing.assert_close(
        kernel.attend_metadata(q, kv[0, :, 1], kv[1, :, 1], unpacked),
        expected,
        atol=2e-6,
        rtol=2e-6,
    )
    assert dict(kernel.prefix_group_stats) == before
