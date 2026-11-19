"""Verify that the benchmark replays the original cold burst at a live greedy decision."""

from copy import deepcopy

import pytest
import torch

from benchmarks import ascend_prefix_guard_trial
from benchmarks.ascend_prefix_guard_runner import guarded_burst, native_burst
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_workload import Case
from tests.helpers import tiny_ouro_model


def test_cold_fallback_replays_the_native_c4_geometry_and_all_four_prompts():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    contract = {"seed": 168, "output_tokens": 3}
    case = Case("c4_h4", "concurrency", 32, 16, concurrency=4, hits=4)
    archive = replay_mode(model, "torch", case, "off", 3, 168)["requests"]
    ids = [row["request_id"] for row in archive]
    cold = native_burst(model, "torch", case, contract, "off")
    assert list(cold["tokens"]) == ids
    assert cold["tokens"] == {row["request_id"]: row["tokens"] for row in archive}
    assert cold["hit_tokens"] == dict.fromkeys(ids, 0)

    result = guarded_burst(model, "torch", case, contract, 1000.0)
    assert result["fallback_used"] and not result["prefix_conflicts"]
    assert result["tokens"] == result["cold"]["tokens"] == cold["tokens"]
    assert result["prompt_sha256"] == result["cold"]["prompt_sha256"] == cold["prompt_sha256"]
    assert result["cold"]["hit_tokens"] == cold["hit_tokens"]
    assert result["hit_tokens"] == dict.fromkeys(ids, 32)
    assert result["trigger"]["position"] == 1
    assert [row["request_id"] for row in result["trigger_step"]] == ids
    for row in result["trigger_step"]:
        rid = row["request_id"]
        assert row["position"] == 1 and result["hot_observed"][rid] == [row["selected_id"]]
        assert result["committed_hot"][rid] == []
    assert 0 < result["hot_target_ms"] < result["total_target_ms"]
    assert result["cold_target_ms"] == result["cold"]["target_ms"] > 0
    assert result["total_target_ms"] == pytest.approx(
        result["hot_target_ms"] + result["cold_target_ms"]
    )
    assert result["hot_target_ms"] < result["first_committed_ms"] <= result["total_target_ms"]


def test_a_tie_on_a_cache_miss_never_replays_a_cold_burst():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    with torch.no_grad():
        model.lm_head.weight.zero_()
    case = Case("c4_h0", "concurrency", 32, 16, concurrency=4, hits=0)
    result = guarded_burst(model, "torch", case, {"seed": 168, "output_tokens": 3}, 0.0)
    assert set(result["hit_tokens"].values()) == {0}
    assert result["trigger"] is None and not result["fallback_used"]
    assert not result["trigger_step"] and result["cold"] is None
    assert result["tokens"] == result["hot_observed"] == result["committed_hot"]
    assert list(result["tokens"].values()) == [[0, 0, 0]] * 4
    assert result["cold_target_ms"] == 0 and result["hot_target_ms"] == result["total_target_ms"]


def test_mixed_burst_ignores_tied_misses_and_checks_only_the_hit_margin():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = Case("c4_h1", "concurrency", 32, 16, concurrency=4, hits=1)
    contract = {"seed": 168, "output_tokens": 2}

    def margins(_, inputs, output):
        if output.shape[0] == 4:
            scores = torch.zeros_like(output)
            scores[:, :2] = 2
            scores[0, 1] = 1.9375  # Only target-0 is a cache hit, with a gap of 0.0625.
            return scores

    with model.lm_head.register_forward_hook(margins):
        below = guarded_burst(model, "torch", case, contract, 0.06)
        exact = guarded_burst(model, "torch", case, contract, 0.0625)
    assert list(below["hit_tokens"].values()) == [32, 0, 0, 0]
    assert below["trigger"] is None and below["tokens"] == below["committed_hot"]
    assert all(seq == [0, 0] for seq in below["tokens"].values())
    assert exact["fallback_used"] and exact["trigger"] == dict(
        request_id="target-0", position=1, gap=0.0625, selected_id=0
    )


@pytest.mark.parametrize("previous_conflict", [False, True])
def test_mid_burst_tie_checks_every_committed_prefix_before_accepting_cold_output(
    previous_conflict,
):
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = Case("c4_h4", "concurrency", 32, 16, concurrency=4, hits=4)
    ids = [f"target-{i}" for i in range(4)]
    shapes = []

    def controlled_logits(_, inputs, logits):
        # Both original donor primers still run. Only edit the four true target
        # rows at CODA; the normal engine, sampler, hit and cold paths all run.
        if logits.shape[0] != 4:
            return
        shapes.append(tuple(logits.shape))
        scores = torch.zeros_like(logits)
        scores[:, 1] = 2
        if len(shapes) == 1 and previous_conflict:
            scores[3, 2] = 4  # Already emitted peer will disagree with its cold history.
        if len(shapes) == 2:
            scores[2, 0] = 2  # A tie only for target-2; native greedy selects smaller id 0.
        return scores

    with model.lm_head.register_forward_hook(controlled_logits):
        result = guarded_burst(model, "torch", case, {"seed": 168, "output_tokens": 3}, 0.0)
    assert len(shapes) == 5  # Two hot output steps followed by all three cold output steps.
    assert result["trigger"] == dict(request_id="target-2", position=2, gap=0.0, selected_id=0)
    assert [row["request_id"] for row in result["trigger_step"]] == ids
    assert result["hot_observed"]["target-2"] == [1, 0]
    assert result["committed_hot"]["target-2"] == [1]
    assert list(result["cold"]["tokens"].values()) == [[1, 1, 1]] * 4
    assert result["first_committed_ms"] <= result["hot_target_ms"]
    if previous_conflict:
        assert result["committed_hot"]["target-3"] == [2]
        assert result["prefix_conflicts"] == {"target-3": 1} and result["tokens"] is None
    else:
        assert not result["prefix_conflicts"] and result["tokens"] == result["cold"]["tokens"]


@pytest.mark.parametrize("scenario", ["recovered", "missed", "late", "stale_archive", "bad_replay"])
def test_trial_classifies_token_flip_and_rejects_invalid_controls(monkeypatch, scenario):
    ids = [f"target-{i}" for i in range(4)]
    focus = ids[2]
    off = dict(
        tokens={rid: [7] * 7 + [30, 1] for rid in ids},
        hit_tokens=dict.fromkeys(ids, 0),
        prompt_sha256={rid: f"prompt-{i}" for i, rid in enumerate(ids)},
        schedule=["cold prefill", "decode"],
        target_ms=10.0,
    )
    hot = dict(
        deepcopy(off), hit_tokens=dict.fromkeys(ids, 128), schedule=["hot prefill", "decode"]
    )
    hot["tokens"][focus][7] = 93
    native = {"off": off, "reuse": hot}
    trial = {}
    for mode, run in native.items():
        trial[mode] = {
            "requests": [
                dict(
                    prompt_sha256=run["prompt_sha256"][rid],
                    tokens=run["tokens"][rid],
                    hit_tokens=run["hit_tokens"][rid],
                    prefill_start=run["hit_tokens"][rid],
                )
                for rid in ids
            ]
        }

    stopped = scenario != "missed"
    late = scenario == "late"
    cut = 9 if late or not stopped else 8
    guard = dict(
        prompt_sha256=hot["prompt_sha256"],
        hit_tokens=hot["hit_tokens"],
        hot_schedule=hot["schedule"],
        hot_observed={rid: seq[:cut] for rid, seq in hot["tokens"].items()},
        cold=deepcopy(off) if stopped else None,
        fallback_used=stopped,
        tokens=None if late else (off if stopped else hot)["tokens"],
        prefix_conflicts={focus: 8} if late else {},
        trigger={"position": cut, "request_id": focus} if stopped else None,
        total_target_ms=15.0 if stopped else 5.0,
        cold_target_ms=10.0 if stopped else 0.0,
        hot_target_ms=5.0,
    )
    if scenario == "stale_archive":
        # Every cold token is identical across slots; prompt identity must still match.
        trial["off"]["requests"].reverse()
    elif scenario == "bad_replay":
        guard["cold"]["schedule"].reverse()
    monkeypatch.setattr(ascend_prefix_guard_trial, "native_burst", lambda *args: native[args[-1]])
    monkeypatch.setattr(ascend_prefix_guard_trial, "guarded_burst", lambda *args: guard)
    case = Case("c4_h4", "concurrency", 128, 16, concurrency=4, hits=4)
    row = ascend_prefix_guard_trial.evaluate_trial(
        None, "torch", case, {}, {"trials": [trial]}, 0.0625, 0
    )
    assert row["native_first_diff"] == {focus: 8}
    if scenario in ("stale_archive", "bad_replay"):
        assert row["result"] == "incomparable" and not row["comparable"]
        issue = (
            "no_joint_archived_native_trial"
            if scenario == "stale_archive"
            else "cold_fallback_did_not_reproduce_independent_off"
        )
        assert issue in row["controls"]
    else:
        assert (
            row["comparable"]
            and row["result"] == scenario
            and row["archived_matches"]["trials"] == [1]
        )
        assert (
            row["guarded_first_diff"]
            == {"late": None, "missed": {focus: 8}, "recovered": {}}[scenario]
        )
