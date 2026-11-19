"""Replay the native hot target's KV history in a private cold C4 target."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

from benchmarks.ascend_prefix_evidence import _arm_checks
from benchmarks.ascend_prefix_fix_trial import _differences, _identity, _schedule, load_replay_model
from benchmarks.ascend_prefix_history_kv import FocusHistoryKV
from benchmarks.ascend_prefix_logits_probe import LogitsProbe
from benchmarks.ascend_prefix_matrix import git_info, save_report
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_scenarios import suites

FOCUS = "target-2"
IDS = tuple(f"target-{i}" for i in range(4))

# The sampling boundary is after token 1 (the entire prompt has been consumed)
# or token 7 (the prompt and the first six generated tokens have been consumed).
TRANSFERS = {
    "sham1": (1, "all"),
    "prompt": (1, "all"),
    "prefix": (1, "prefix"),
    "suffix": (1, "suffix"),
    "sham7": (7, "all"),
    "history": (7, "all"),
    "late_prompt": (7, "prompt"),
    "generated": (7, "generated"),
}


def _replay(model, backend, case, contract, arm, **transfer):
    observer = logits = None

    @contextmanager
    def target_context(engine, targets, donor):
        nonlocal observer, logits
        observer = FocusHistoryKV(
            engine, IDS, FOCUS, case.prefix + case.suffix, case.reusable_tokens, **transfer
        )
        logits = LogitsProbe(engine.model_runner)
        with observer, logits:
            yield

    run = replay_mode(
        model,
        backend,
        case,
        arm,
        contract["output_tokens"],
        contract["seed"],
        target_context=target_context,
    )
    run.update(
        logits=logits.report(),
        kv_history=observer.report(),
        output_tokens_per_request=contract["output_tokens"],
    )
    return run, observer.snapshots


def _outcome(run, cold, hot):
    def focus_tokens(arm):
        return next(row["tokens"] for row in arm["requests"] if row["request_id"] == FOCUS)

    tokens = focus_tokens(run)
    same = tokens[:7] == focus_tokens(cold)[:7] == focus_tokens(hot)[:7]
    return dict(
        off_vs_run=_differences(cold, run),
        reuse_vs_run=_differences(hot, run),
        same_focus_history_at_8=same,
        focus_token_8=tokens[7],
        scores_at_8=run["logits"][FOCUS][7] if same else None,
    )


def history_trial(model, backend, case, contract, archive):
    if (case.concurrency, case.hits, case.churn) != (
        4,
        4,
        0,
    ) or case.prefix != case.reusable_tokens:
        raise ValueError("expected four hot targets with a page-aligned prefix")
    if contract["output_tokens"] < 8:
        raise ValueError("the source replay must include the eighth sampling decision")
    runs, controls, boundary = {}, [], {}
    hot_pages = {}
    for arm in ("off", "reuse"):
        run, captured = _replay(model, backend, case, contract, arm, capture=True)
        runs[arm] = run
        prior = archive["runs"][arm]
        if (
            _identity(prior) != _identity(run)
            or _schedule(prior) != _schedule(run)
            or prior["donor_prefill_spans"] != run["donor_prefill_spans"]
        ):
            controls.append(f"{arm}:archived_replay")
        if "logits" in prior and prior["logits"][FOCUS][6:8] != run["logits"][FOCUS][6:8]:
            controls.append(f"{arm}:archived_focus_scores")
        taps = run["kv_history"]["boundaries"]
        if taps["1"]["all"]["before_sha256"] != taps["7"]["prompt"]["before_sha256"]:
            controls.append(f"{arm}:prompt_kv_changed_between_boundaries")
        # Only the hot values are needed as sources for the remaining replays.
        if arm == "reuse":
            hot_pages = captured
        del captured
        checks = _arm_checks(run)
        controls.extend(f"{arm}:{issue}" for issue in checks["issues"] + checks["probe_issues"])

    cold, hot = runs["off"], runs["reuse"]
    histories = cold["kv_history"]["histories"]
    if histories != hot["kv_history"]["histories"]:
        controls.append("native_token_histories_at_cutpoints")

    transfers = TRANSFERS if not controls else {}
    for name, (after, region) in transfers.items():
        cut = str(after)
        sham = name in ("sham1", "sham7")
        source = None if sham else hot_pages[after]
        run, _ = _replay(
            model, backend, case, contract, "off", after=after, region=region, source=source
        )
        runs[name] = run
        kv = run["kv_history"]
        snapshot = kv["boundaries"][cut][region]
        boundary[name] = snapshot
        if kv["histories"][cut] != histories[cut]:
            controls.append(f"{name}:token_history_before_copy")
        if _schedule(run) != _schedule(cold):
            controls.append(f"{name}:cold_schedule")
        before = cold["kv_history"]["boundaries"][cut][region]["before_sha256"]
        reference = cold if sham else hot
        intended = reference["kv_history"]["boundaries"][cut][region]["before_sha256"]
        if (snapshot["before_sha256"], snapshot["source_sha256"], snapshot["after_sha256"]) != (
            before,
            intended,
            intended,
        ):
            controls.append(f"{name}:copy_fingerprints")
        if sham and (run["logits"] != cold["logits"] or snapshot["changed_pages"]):
            controls.append(f"{name}:round_trip_changed_native")
        for rid in IDS:
            if rid != FOCUS and run["logits"][rid] != cold["logits"][rid]:
                controls.append(f"{name}:{rid}:peer_scores_changed")

        checks = _arm_checks(run)
        controls.extend(f"{name}:{issue}" for issue in checks["issues"] + checks["probe_issues"])
    return dict(
        summary=dict(
            case=case.name,
            focus=FOCUS,
            comparable=not controls,
            controls=controls,
            boundary=boundary,
            native_off_vs_reuse=_differences(cold, hot),
            hit_tokens={
                m: [r["hit_tokens"] for r in runs[m]["requests"]] for m in ("off", "reuse")
            },
            outcomes={name: _outcome(run, cold, hot) for name, run in runs.items()},
        ),
        runs=runs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--suffix", type=Path, required=True, help="previous C4_SUFFIX JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.suffix.read_text())
    summary = source["summary"]
    if summary["case"] != "c4_h4" or not summary["comparable"] or "first_layer" not in summary:
        parser.error("supply the comparable c4_h4 suffix trial report")
    contract = source["contract"]
    if contract["output_tokens"] < 8:
        parser.error("the suffix report must include at least eight generated tokens")
    case = next(c for c in suites()["full"] if c.name == "c4_h4")
    model = load_replay_model(parser, args, contract)
    result = history_trial(model, contract["backend"], case, contract, source)
    result.update(
        contract=contract, commit=git_info("rev-parse", "HEAD"), suffix_commit=source.get("commit")
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=False)
    save_report(args.out, result)
    summary = result["summary"].copy()
    summary["changed_pages"] = {
        name: tap["changed_pages"] for name, tap in summary.pop("boundary").items()
    }
    outcomes = summary.pop("outcomes")
    # Do not present a position-8 comparison once the preceding tokens differ.
    summary["focus_8_on_native_history"] = {
        name: outcome["focus_token_8"] if outcome["same_focus_history_at_8"] else None
        for name, outcome in outcomes.items()
    }
    cold_id, hot_id = (outcomes[arm]["focus_token_8"] for arm in ("off", "reuse"))
    if cold_id != hot_id:
        gaps = {}
        for name, outcome in outcomes.items():
            point = outcome["scores_at_8"]
            scores = dict(zip(point["top_ids"], point["top_logits"], strict=True)) if point else {}
            gaps[name] = (
                scores[hot_id] - scores[cold_id] if hot_id in scores and cold_id in scores else None
            )
        summary["logit_gap_at_8"] = dict(hot_id=hot_id, cold_id=cold_id, hot_minus_cold=gaps)
    summary["first_diff_vs_off"] = {
        name: outcome["off_vs_run"] for name, outcome in outcomes.items() if name in TRANSFERS
    }
    print("C4_HISTORY", json.dumps(summary, separators=(",", ":"), allow_nan=False), flush=True)
    return int(not summary["comparable"])


if __name__ == "__main__":
    raise SystemExit(main())
