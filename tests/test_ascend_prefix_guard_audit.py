"""Check when a guard could have withheld the first mismatching sampling step."""

import pytest

from benchmarks.ascend_prefix_guard_audit import audit_diagnosis


def _arm(hit, tokens, margins):
    count = len(tokens)
    rid = "target-0"
    points = [
        dict(
            position=i,
            selected_id=token,
            top_ids=[token, 93 if token == 30 else 30],
            top_logits=[8.4375 + gap, 8.4375],
            margin=gap,
        )
        for i, (token, gap) in enumerate(zip(tokens, margins, strict=True), 1)
    ]
    row = dict(
        request_id=rid,
        prompt_sha256="same",
        tokens=tokens,
        hit_tokens=hit,
        prefill_start=hit,
        expected_hit_tokens=hit,
        exit_depths=[4] * count,
        finish_reason="length",
    )
    steps = [
        dict(
            step=i,
            spans=[dict(request_id=rid, phase="coda")],
            visible=[dict(request_id=rid, token_ids=[token])],
        )
        for i, token in enumerate(tokens, 1)
    ]
    return dict(
        requests=[row], prefix_hits=hit, prefix_queries=1, steps=steps, logits={rid: points}
    )


def _diagnosis(*, hot_ids=(30, 93, 30), hit=128):
    comparison = dict(
        primary_issues=[],
        diagnostic_issues=[],
        status="PASS",
        joint_archive_outputs_on_same_inputs_1based={"trials": [1]},
    )
    if 93 in hot_ids:
        comparison.update(status="FAIL", primary_issues=["off_reuse:tokens"])
    return dict(
        state="complete",
        contract={"output_tokens": 3},
        cases={
            "case": [
                dict(
                    comparison=comparison,
                    runs=dict(
                        off=_arm(0, [30] * 3, [0] * 3),
                        reuse=_arm(hit, list(hot_ids), [0.25, 0.0625, 0.03125]),
                    ),
                )
            ]
        },
    )


def test_first_flip_uses_observable_hot_gap_instead_of_cold_tie():
    row = audit_diagnosis(_diagnosis(), [0, 0.03125, 0.0625, 0.25])["case"][0]
    assert row["comparable"] and not row["controls"]
    assert row["first_diff"] == dict(
        request_id="target-0",
        step=2,
        position=2,
        off_id=30,
        hot_id=93,
        hot_margin=0.0625,
        hit_tokens=128,
    )
    trials = row["thresholds"]
    assert [t["status"] for t in trials] == ["missed", "late", "on", "early"]
    assert [t["trigger"] and t["trigger"]["position"] for t in trials] == [None, 3, 2, 1]


def test_pass_trigger_and_no_hit_requests():
    row = audit_diagnosis(_diagnosis(hot_ids=(30,) * 3), [0, 0.25])["case"][0]
    assert row["first_diff"] is None
    assert [t["status"] for t in row["thresholds"]] == ["unneeded", "unnecessary"]
    row = audit_diagnosis(_diagnosis(hit=0), [1])["case"][0]
    assert row["first_diff"]["hit_tokens"] == 0
    assert row["thresholds"][0] == dict(threshold=1, status="missed", trigger=None)


def test_same_step_trigger_withholds_other_rows_mismatch():
    diag = _diagnosis(hit=0)
    for arm, run in diag["cases"]["case"][0]["runs"].items():
        hit = 128 if arm == "reuse" else 0
        peer = _arm(hit, [30] * 3, [0.25, 0, 0.25])
        peer["requests"][0]["request_id"] = "target-1"
        run["prefix_hits"] += hit
        run["prefix_queries"] += 1
        run["requests"].extend(peer["requests"])
        run["logits"]["target-1"] = peer["logits"]["target-0"]
        for step in run["steps"]:
            step["spans"].append(dict(request_id="target-1", phase="coda"))
            step["visible"].append(dict(request_id="target-1", token_ids=[30]))
    row = audit_diagnosis(diag, [0])["case"][0]
    assert row["thresholds"][0]["status"] == "on"
    assert row["thresholds"][0]["trigger"]["request_id"] == "target-1"
    assert row["first_diff"]["request_id"] == "target-0"


@pytest.mark.parametrize(
    "mutation", ["wrong_gap", "wrong_visible_id", "stale_archive", "missing_scores"]
)
def test_incompatible_replay_is_reported(mutation):
    diag = _diagnosis()
    replay = diag["cases"]["case"][0]
    hot = replay["runs"]["reuse"]
    if mutation == "wrong_gap":
        hot["logits"]["target-0"][1]["margin"] = 0
    elif mutation == "wrong_visible_id":
        hot["steps"][1]["visible"][0]["token_ids"] = [30]
    elif mutation == "stale_archive":
        replay["comparison"]["joint_archive_outputs_on_same_inputs_1based"] = {}
    else:
        del hot["logits"]
    row = audit_diagnosis(diag, [0.0625])["case"][0]
    assert not row["comparable"] and row["controls"] and not row["thresholds"]
