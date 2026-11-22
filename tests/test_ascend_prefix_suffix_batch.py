from dataclasses import replace
from functools import partial
from unittest.mock import patch

import pytest
import torch

from benchmarks import ascend_prefix_suffix_batch as experiment
from vllm_rlt.kernels.ascend_eager import AscendEagerAttention
from vllm_rlt.worker.metadata import KVMetadata


@pytest.mark.parametrize("width", [2, 4])
@torch.inference_mode()
def test_suffix_only_plans_preserve_pages_absolute_masks_and_row_order(width):
    generator = torch.Generator().manual_seed(168)
    keys, values = [
        torch.randn(10, 2, 4, 1, 8, generator=generator, dtype=torch.bfloat16) for _ in range(2)
    ]
    owners = [object() for _ in range(4)]
    native, candidate = AscendEagerAttention(), AscendEagerAttention()
    audit = experiment.SuffixExperiment.new_audit()
    split = experiment.two.mixed.split_forward(
        AscendEagerAttention._forward, audit, partial(experiment.SuffixPlan.plan, width=width)
    )
    # Cover repeated metadata within a depth, distinct layer storage offsets,
    # shared prefix pages and four private suffix pages in a nontrivial order.
    for depth in range(2):
        rows = [(owners[i], depth, p) for p in (4, 5) for i in (3, 1, 0, 2)]
        pages = {owner: slot for owner, slot in zip(owners, (4, 2, 3, 1), strict=True)}
        tables = torch.tensor([[5 * depth, 5 * depth + pages[owner]] for owner, _, _ in rows])
        pos = torch.tensor([p for _, _, p in rows])
        batch = KVMetadata(
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
        q = torch.randn(len(pos), 2, 8, generator=generator, dtype=keys.dtype)
        for layer in range(2):
            args = (q, keys[:, layer], values[:, layer], batch)
            expected = native.attend_metadata(*args)
            with patch.object(AscendEagerAttention, "_forward", split):
                got = candidate.attend_metadata(*args)
                assert torch.equal(got, expected)
                # Only the very last row belongs to request 2 at position 5.
                changed = values.clone()[:, layer]
                changed[5 * depth + pages[owners[2]], 1] += 64
                after = candidate.attend_metadata(q, keys[:, layer], changed, batch)
                assert (got != after).any(dim=(1, 2)).nonzero().flatten().tolist() == [7]
        # Neither limiting the batch width to 2 nor grouping four may hide a
        # different absolute interval or loop depth across the two pairs.
        for last_depth, last_start in ((depth + 1, 4), (depth, 8)):
            other = [
                (a, last_depth, p + last_start - 4) if a is owners[2] else (a, d, p)
                for a, d, p in rows
            ]
            metadata = replace(batch, rows=tuple(other))
            groups = native._plan(
                [(id(a), d) for a, d, p in other], [p + 1 for a, d, p in other], q.device
            )
            assert experiment.SuffixPlan.plan(metadata, groups, width=width) is None
    assert audit["split_calls"] == 8
