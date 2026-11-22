from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from benchmarks import ascend_prefix_three_plus_one as bench
from vllm_rlt.kernels.ascend_eager import AscendEagerAttention
from vllm_rlt.worker.metadata import KVMetadata


@pytest.mark.parametrize("hot_slot", [0, 1])
@torch.inference_mode()
def test_three_cold_rows_keep_their_own_pages_order_and_depth(hot_slot):
    generator = torch.Generator().manual_seed(168)
    owners = [object() for _ in range(4)]
    # Distinct layer storage offsets and depth-specific page tables expose stale
    # row selectors or an accidentally shared slot/mask cache.
    keys = torch.randn(24, 2, 4, 1, 8, dtype=torch.bfloat16, generator=generator)
    values = torch.randn(keys.shape, dtype=keys.dtype, generator=generator)
    reference, candidate = AscendEagerAttention(), AscendEagerAttention()
    audit = bench.ThreePlusOne.new_audit()
    split = bench.split_forward(AscendEagerAttention._forward, audit)
    for depth in range(2):
        rows, tables = [], []
        for slot, owner in enumerate(owners):
            positions = (4, 5) if slot == hot_slot else range(6)
            rows.extend((owner, depth, p) for p in positions)
            tables.extend([[depth * 12 + 2 * slot, depth * 12 + 2 * slot + 1] for _ in positions])
        positions = torch.tensor([pos for _, _, pos in rows])
        tables = torch.tensor(tables)
        batch = KVMetadata(
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
        q = torch.randn(len(rows), 2, 8, dtype=keys.dtype, generator=generator)
        for layer in range(2):
            args = (q, keys[:, layer], values[:, layer], batch)
            expected = reference.attend_metadata(*args)
            with patch.object(AscendEagerAttention, "_forward", split):
                actual = candidate.attend_metadata(*args)
            assert torch.equal(actual, expected)
    assert audit["split_calls"] == 4


def test_trace_stops_after_every_target_has_a_first_token_and_restores_step(monkeypatch):
    events = []

    @contextmanager
    def capture(*_):
        events.append("start")
        try:
            yield
        finally:
            events.append("stop")

    monkeypatch.setattr(bench, "trace_target", capture)
    outputs = iter(
        [
            [],
            [SimpleNamespace(request_id="target-0", token_ids=[7])],
            [SimpleNamespace(request_id="target-1", token_ids=[8])],
            [SimpleNamespace(request_id="target-1", token_ids=[8, 9])],
        ]
    )
    engine = SimpleNamespace(step=lambda: next(outputs))
    original = engine.step
    with bench.first_tokens(engine, [(), ()], profiler=None, directory=None, mode="original"):
        engine.step()
        engine.step()
        assert events == ["start"]
        engine.step()
        assert events == ["start", "stop"]
        engine.step()
    assert events == ["start", "stop"]
    assert engine.step is original
