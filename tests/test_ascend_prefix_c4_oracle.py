"""Four-request prefix replays and controlled replacement of donor KV pages."""

import copy

import pytest
import torch

from benchmarks.ascend_prefix_c4_oracle import c4_experiment
from benchmarks.ascend_prefix_page_control import ColdPages, DonorPageControl
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_workload import Case
from tests.helpers import tiny_ouro_config
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models import OuroForCausalLM


@pytest.fixture(scope="module")
def recorded_c4():
    torch.manual_seed(11)
    model = OuroForCausalLM(tiny_ouro_config()).to(dtype=torch.bfloat16)
    case = Case("c4_h4", "concurrency", 32, 16, concurrency=4, hits=4)
    runs = {arm: replay_mode(model, "torch", case, arm, 3, 168) for arm in ("off", "reuse")}
    return model, case, {"seed": 168, "output_tokens": 3}, {"runs": runs}


def test_c4_oracle_preserves_order_and_shared_page_geometry(recorded_c4):
    model, case, contract, prior = recorded_c4
    stale = copy.deepcopy(prior)
    for run in stale["runs"].values():
        run["requests"][2]["prompt_sha256"] = "0" * 64
    out = c4_experiment(model, "torch", case, contract, [stale, prior])
    summary = out["summary"]
    assert summary["matched_prior_replays"] == [2]
    assert summary["comparable"] and not summary["controls"]
    assert summary["focus"] == "target-2"
    ids = [f"target-{i}" for i in range(4)]
    assert set(summary["cold_vs_donor"]) == set(ids)
    assert summary["sham_vs_reuse"] == []
    assert summary["sham_pages"] == {"checked_pages": 8, "changed_pages": 0}
    assert summary["cold_pages"]["checked_pages"] == 8
    assert 0 <= summary["cold_pages"]["changed_pages"] <= 8
    assert set(out["runs"]) == {"off", "reuse", "sham", "cold_pages"}
    for mode, run in out["runs"].items():
        rows = run["requests"]
        hit = 0 if mode == "off" else 32
        assert [r["request_id"] for r in rows] == ids
        assert [r["prefill_start"] for r in rows] == [hit] * 4
        assert [r["hit_tokens"] for r in rows] == [hit] * 4
        assert (run["prefix_hits"], run["prefix_queries"]) == (hit * 4, 4)
        assert [r["prompt_sha256"] for r in rows] == [
            r["prompt_sha256"] for r in prior["runs"]["off"]["requests"]
        ]

    off = [r["tokens"] for r in out["runs"]["off"]["requests"]]
    for mode, key in (("reuse", "native_off_vs_reuse"), ("cold_pages", "off_vs_cold_pages")):
        other = [r["tokens"] for r in out["runs"][mode]["requests"]]
        assert bool(summary[key]) == (other != off)


def test_c4_oracle_does_not_match_a_changed_focus_prompt(recorded_c4):
    model, case, contract, prior = recorded_c4
    stale = copy.deepcopy(prior)
    for run in stale["runs"].values():
        run["requests"][2]["prompt_sha256"] = "f" * 64
    summary = c4_experiment(model, "torch", case, contract, [stale])["summary"]
    assert not summary["comparable"] and not summary["matched_prior_replays"]


def _fill_pages(cache, rid):
    assert cache.allocate(rid, 5)
    tables = cache.plane_block_tables(rid)
    for depth, table in enumerate(tables):
        for logical, physical in enumerate(table):
            value = 10 * depth + logical + 1
            cache.key_cache[physical].fill_(value)
            cache.value_cache[physical].fill_(value + 20)
    cache.mark_imported_prefix(rid, 5)
    return tables


def test_donor_page_control_replaces_one_logical_value_page_without_touching_tail():
    tokens = [1, 2, 3, 4, 5]
    cold = KVCacheManager(2, 1, 4, 20, 2, 2, dtype=torch.bfloat16)
    with ColdPages(cold, 4, "target-2") as capture:
        _fill_pages(cold, "target-2")
        cold.free("target-2")
    # Freed source storage can be reused before another engine consumes the snapshot.
    cold.value_cache.fill_(-100)
    assert (capture.pages[1][1][1] == 32).all()
    # Depth 1, logical page 1, value tensor: flip a single bit in layer 1.
    capture.pages[1][1][1][1].view(torch.uint8).reshape(-1)[0] ^= 1

    warm = KVCacheManager(2, 1, 4, 20, 2, 2, dtype=torch.bfloat16, enable_prefix_caching=True)
    warm.key_cache.zero_()
    warm.value_cache.zero_()
    # A different physical layout prevents using page IDs from the cold engine.
    assert warm.allocate("padding", 3)
    control = DonorPageControl(warm, 4, replacement=capture.pages)
    with control:
        tables = _fill_pages(warm, "donor")
        warm.free("padding")
        warm.publish_prefix("donor", tokens, len(tokens))
        expected_keys = warm.key_cache.clone()
        expected_values = warm.value_cache.clone()
        expected_values[tables[1][1], 1].view(torch.uint8).reshape(-1)[0] ^= 1
        warm.free("donor")

    assert control.report() == {"checked_pages": 4, "changed_pages": 1}
    assert torch.equal(warm.key_cache.view(torch.int16), expected_keys.view(torch.int16))
    assert torch.equal(warm.value_cache.view(torch.int16), expected_values.view(torch.int16))
    hits = warm.lookup_prefix(tokens)
    assert len(hits) == 2
    assert warm.allocate("receiver", 5, prefix=hits)
    _, values = warm.read(layer=1, request_id="receiver", depth=1, length=4)
    expected = torch.cat([value[1] for _, value in capture.pages[1]])
    assert torch.equal(values.view(torch.int16), expected.view(torch.int16))
    warm.free("receiver")
