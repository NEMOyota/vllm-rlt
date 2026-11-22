from dataclasses import replace

import pytest
import torch

from tests.test_ascend_cold_prefill import span
from vllm_rlt.kernels.ascend_eager import AscendEagerAttention
from vllm_rlt.worker.metadata import KVMetadata, MetadataBuilder
from vllm_rlt.worker.model_runner import _cold_prefill_groups, _suffix_prefill_groups


def test_schedule_cached_suffixes_by_origin_depth_and_width():
    h = span(8, start=4, hit=4)
    spans = [
        h,
        span(9, start=5, hit=5),
        span(7, start=4, hit=4),
        span(8, start=4),
        span(9, start=4, hit=4, width=4),
        span(8, start=4, hit=4, depth=1),
        span(5, start=4, hit=4),
        h,
    ]
    assert _suffix_prefill_groups(spans) == ((0, 24),)
    assert _suffix_prefill_groups([h] * 7) == ((0, 4, 8, 12), (16, 20, 24))


@pytest.mark.parametrize("hot_slots", [(0, 2), (0, 1, 3), (0, 1, 2, 3)])
@torch.inference_mode()
def test_real_suffix_path_preserves_private_history_and_absolute_causality(hot_slots):
    rng = torch.Generator().manual_seed(168)
    spans = [span(7, start=4, hit=4) if i in hot_slots else span(7) for i in range(4)]
    keys, values = [
        torch.randn(16, 2, 4, 2, 8, generator=rng, dtype=torch.bfloat16) for _ in range(2)
    ]
    native, grouped = AscendEagerAttention(), AscendEagerAttention()
    for depth in (0, 1):
        rows, tables = [], []
        for i, s in enumerate(spans):
            for pos in range(s.token_start, s.token_start + s.token_count):
                rows.append((s.request, depth, pos))
                # Some requests share a prefix page, each has a private suffix page.
                tables.append([8 * depth + (i % 2 if i in hot_slots else i), 8 * depth + 7 - i])
        tables = torch.tensor(tables)
        pos = torch.tensor([r[2] for r in rows])
        meta = KVMetadata(
            None,
            tuple(rows),
            (),
            pos,
            tables[torch.arange(len(pos)), pos // 4],
            pos % 4,
            tables,
            pos + 1,
            True,
        )
        batch = replace(
            meta,
            cold_prefill_groups=_cold_prefill_groups(spans),
            suffix_prefill_groups=_suffix_prefill_groups(spans),
        )
        assert not MetadataBuilder.slice_rows(batch, 0, 1).suffix_prefill_groups
        q = torch.randn(len(pos) + 2, 4, 8, generator=rng, dtype=keys.dtype)
        for layer in range(2):
            k, v = keys[:, layer], values[:, layer]
            for layer_k, layer_v in ((k, v), (k.contiguous(), v.contiguous())):
                y = grouped.attend_metadata(q, layer_k, layer_v, batch)
                assert torch.equal(y, native.attend_metadata(q, layer_k, layer_v, meta))
                assert not y[-2:].count_nonzero()
            changed = values.clone()[:, layer]
            changed[8 * depth + 7, 2] += 128  # request 0, absolute key position 6
            y1 = grouped.attend_metadata(q, k, changed, batch)
            diffs = (y1 != y).any(dim=(1, 2)).nonzero().flatten().tolist()
            assert diffs == [2]
    assert grouped.suffix_prefill_calls == 12
    assert grouped.cold_prefill_calls == (12 if len(hot_slots) == 2 else 0)
