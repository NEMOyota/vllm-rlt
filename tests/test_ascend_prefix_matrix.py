"""Engine-level checks for the Ascend prefix benchmark using a small CPU model."""

import copy
from dataclasses import replace

import pytest
import torch

from benchmarks.ascend_prefix_scenarios import summarize
from benchmarks.ascend_prefix_workload import Case, run_trial
from tests.helpers import tiny_ouro_model


@pytest.fixture(scope="module")
def model():
    return tiny_ouro_model(dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "case, expected",
    [
        (Case("below-page", "pages", 15, 2), [0]),
        (Case("exact-prompt", "pages", 32, 0), [16]),
        (Case("half-warm", "mix", 33, 5, concurrency=2, hits=1), [32, 0]),
    ],
)
def test_target_reuse_matches_schedule(model, case, expected):
    trial = run_trial(model, "torch", case, "reuse", output_tokens=2, seed=168)
    for key in ("expected_hit_tokens", "hit_tokens", "prefill_start"):
        assert [row[key] for row in trial["requests"]] == expected
    assert trial["prefix_hits"] == sum(expected)
    assert trial["output_tokens"] == 2 * case.concurrency


def test_eviction_keeps_target_identical_and_removes_previously_resident_pages(model):
    case = Case("resident", "retention", 32, 5, tight_pool=True)
    resident = run_trial(model, "torch", case, "reuse", output_tokens=2, seed=168)
    evicted = run_trial(model, "torch", replace(case, churn=2), "reuse", output_tokens=2, seed=168)
    (hot,) = resident["requests"]
    (cold,) = evicted["requests"]
    assert resident["num_blocks"] == evicted["num_blocks"]
    assert resident["primed_hit_tokens"] == evicted["primed_hit_tokens"] == hot["hit_tokens"] == 32
    assert evicted["prefix_hits"] == cold["prefill_start"] == cold["hit_tokens"] == 0
    assert cold["prompt_sha256"] == hot["prompt_sha256"]
    assert cold["tokens"] == hot["tokens"]


def test_mixed_burst_reports_per_request_parity_and_failures(model):
    case = Case("mixed", "mix", 33, 5, concurrency=4, hits=2)
    trial = {
        arm: run_trial(model, "torch", case, arm, output_tokens=3, seed=168)
        for arm in ("off", "miss", "reuse")
    }
    for arm, sample in trial.items():
        expected = [32, 32, 0, 0] if arm == "reuse" else [0, 0, 0, 0]
        rows = sample["requests"]
        assert [row["hit_tokens"] for row in rows] == expected
        assert [row["prefill_start"] for row in rows] == expected
        assert [row["tokens"] for row in rows] == [r["tokens"] for r in trial["off"]["requests"]]
        assert all(0 < row["ttft_ms"] <= row["e2e_ms"] for row in rows)
        assert sample["prefix_hits"] == sum(expected)

    report = summarize(case, [trial])
    assert report["status"] == "PASS", report["mismatches"]
    assert report["arms"]["reuse"]["hit_requests"] == 2
    assert report["arms"]["reuse"]["hit_tokens"] == 64

    for key in ("tokens", "prefill_start", "hit_tokens"):
        broken = copy.deepcopy(trial)
        row = broken["reuse"]["requests"][1]
        if key == "tokens":
            row[key][2] = (row[key][2] + 1) % model.config.vocab_size
        else:
            row[key] = 0
        report = summarize(case, [broken])
        assert report["status"] == "FAIL", key
        assert report["mismatch_count"] >= 1, key
        assert not report["comparisons"]

    trial["reuse"]["prefix_queries"] -= 1
    assert summarize(case, [trial])["mismatches"][0]["kind"] == "query_counter"


def test_summary_compares_identical_work_with_both_baselines(model):
    case = Case("compare", "mix", 16, 1)
    trial = {}
    for arm, ttft, wall in (("off", 100, 200), ("miss", 120, 250), ("reuse", 90, 160)):
        sample = run_trial(model, "torch", case, arm, output_tokens=2, seed=168)
        sample["requests"][0].update(ttft_ms=ttft, e2e_ms=wall)
        sample["wall_ms"] = wall
        trial[arm] = sample
    report = summarize(case, [trial])
    assert report["status"] == "PASS", report["mismatches"]
    assert report["comparisons"]["reuse_vs_off"]["ttft_p50_saved_pct"] == pytest.approx(10)
    assert report["comparisons"]["reuse_vs_miss"]["ttft_p50_saved_pct"] == pytest.approx(25)
    assert report["comparisons"]["reuse_vs_off"]["output_tok_s_gain"] == pytest.approx(2.5)
    assert report["arms"]["reuse"]["tpot_p50_ms"] == pytest.approx(70)
