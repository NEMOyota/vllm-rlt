from functools import partial
from unittest.mock import patch

import pytest
import torch

from benchmarks import ascend_prefix_two_plus_two as experiment
from vllm_rlt.kernels.ascend_eager import AscendEagerAttention
from vllm_rlt.worker.metadata import KVMetadata


def metadata(owners, depth, second_start):
    # Interleave cold and hot requests. The hot requests share the first page,
    # but have different suffix pages. Each loop depth owns its own pages.
    rows, tables = [], []
    for owner, slots, positions in zip(
        owners,
        ([3, 6], [1, 5], [4, 0], [1, 2]),
        (range(6), range(4, 6), range(6), range(second_start, 6)),
        strict=True,
    ):
        for pos in positions:
            rows.append((owner, depth, pos))
            tables.append([7 * depth + s for s in slots])
    tables = torch.tensor(tables)
    pos = torch.tensor([p for _, _, p in rows])
    return KVMetadata(
        owner=None,
        rows=tuple(rows),
        allocations=(),
        position_ids=pos,
        write_blocks=tables[torch.arange(len(pos)), pos // 4],
        write_offsets=pos % 4,
        block_tables=tables,
        context_lengths=pos + 1,
        writable=True,
    )


@pytest.mark.parametrize("mode", ["cold_pair", "two_plus_two", "unequal_suffixes"])
@torch.inference_mode()
def test_interleaved_requests_use_their_own_absolute_context(mode):
    random = torch.Generator().manual_seed(172)
    keys, values = [
        torch.randn(14, 2, 4, 1, 8, generator=random, dtype=torch.bfloat16) for _ in range(2)
    ]
    native, candidate = AscendEagerAttention(), AscendEagerAttention()
    owners = [object() for _ in range(4)]
    audit = experiment.TwoPlusTwo.new_audit()
    plan = (
        partial(experiment.mixed.ColdBatch.plan, cold_count=2)
        if mode == "cold_pair"
        else experiment.SuffixPair.plan
    )
    split = experiment.mixed.split_forward(AscendEagerAttention._forward, audit, plan)
    for depth in range(2):
        batch = metadata(owners, depth, 3 if mode == "unequal_suffixes" else 4)
        q = torch.randn(len(batch.rows), 2, 8, generator=random, dtype=keys.dtype)
        for layer in range(2):
            args = (q, keys[:, layer], values[:, layer], batch)
            expected = native.attend_metadata(*args)
            with patch.object(AscendEagerAttention, "_forward", split):
                actual = candidate.attend_metadata(*args)
                assert torch.equal(actual, expected)
                if mode == "two_plus_two":
                    # Change only the *last* value of one suffix. Neither its
                    # earlier query nor any other request may observe that change.
                    altered = values.clone()[:, layer]
                    altered[7 * depth + 5, 1] += 64
                    changed = candidate.attend_metadata(q, keys[:, layer], altered, batch)
                    affected = [
                        i for i, (a, _, p) in enumerate(batch.rows) if a is owners[1] and p == 5
                    ]
                    assert (changed != actual).any(
                        dim=(1, 2)
                    ).nonzero().flatten().tolist() == affected
    assert audit["split_calls"] == {"cold_pair": 4, "two_plus_two": 8, "unequal_suffixes": 0}[mode]
