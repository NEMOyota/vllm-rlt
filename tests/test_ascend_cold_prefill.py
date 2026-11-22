from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_rlt.kernels.ascend_eager import AscendEagerAttention
from vllm_rlt.worker.metadata import KVMetadata, MetadataBuilder
from vllm_rlt.worker.model_runner import _cold_prefill_groups


def span(length, *, start=0, width=None, depth=0, hit=0):
    req = SimpleNamespace(prompt_token_ids=range(length), prefill_window_origin=hit)
    return SimpleNamespace(
        request=req,
        token_start=start,
        token_count=length - start if width is None else width,
        depth=depth,
    )


def test_group_only_complete_prompts_with_equal_depth_and_width():
    suffix = span(9, start=6, hit=6)
    cold = span(3)
    assert _cold_prefill_groups([cold] * 4) == ()
    assert _cold_prefill_groups([span(9, start=6), cold, cold]) == ()
    spans = [cold, suffix, span(9, width=3), span(3, depth=1), cold, span(3, depth=1)]
    assert _cold_prefill_groups(spans) == ((0, 12), (9, 15))
    assert _cold_prefill_groups([suffix] + [cold] * 6) == ((3, 6, 9, 12), (15, 18))


@pytest.mark.parametrize(
    "lengths,hot_slot,group_count",
    [([6] * 4, 0, 1), ([6] * 4, 1, 1), ([6, 5, 6, 5, 6, 5], 2, 2)],
)
@torch.inference_mode()
def test_framework_batches_cold_prompts_with_private_pages_and_scatter(
    lengths, hot_slot, group_count
):
    gen = torch.Generator().manual_seed(168)
    blocks_per_depth = len(lengths) * 2
    keys, values = [
        torch.randn(2 * blocks_per_depth, 2, 4, 1, 8, generator=gen, dtype=torch.bfloat16)
        for _ in range(2)
    ]
    requests = [span(n) for n in lengths]
    requests[hot_slot] = span(6, start=4, hit=4)
    cold = _cold_prefill_groups(requests)
    reference, grouped = AscendEagerAttention(), AscendEagerAttention()
    for depth in range(2):
        rows, tables = [], []
        for i, s in enumerate(requests):
            for p in range(s.token_start, s.token_start + s.token_count):
                rows.append((s.request, depth, p))
                first_page = blocks_per_depth * depth + 2 * i
                tables.append([first_page, first_page + 1])
        positions = torch.tensor([p for _, _, p in rows])
        tables = torch.tensor(tables)
        data = KVMetadata(
            owner=None,
            rows=tuple(rows),
            allocations=(),
            position_ids=positions,
            write_blocks=tables[torch.arange(len(rows)), positions // 4],
            write_offsets=positions % 4,
            block_tables=tables,
            context_lengths=positions + 1,
            writable=True,
        )
        batch = replace(data, cold_prefill_groups=cold)
        assert not MetadataBuilder.slice_rows(batch, 0, 1).cold_prefill_groups
        q = torch.randn(len(rows) + 2, 2, 8, generator=gen, dtype=keys.dtype)
        for layer in range(2):
            k, v = keys[:, layer], values[:, layer]
            for layer_k, layer_v in ((k, v), (k.contiguous(), v.contiguous())):
                expected = reference.attend_metadata(q, layer_k, layer_v, data)
                actual = grouped.attend_metadata(q, layer_k, layer_v, batch)
                assert torch.equal(actual, expected)
                assert not actual[-2:].count_nonzero()
    assert grouped.cold_prefill_calls == 8 * group_count
    assert reference.cold_prefill_calls == 0
