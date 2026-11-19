"""Evaluate a hot greedy margin trigger and whole-burst cold replay against archived outputs."""

import argparse
import json
import math
from pathlib import Path

from benchmarks.ascend_prefix_evidence import PHASES, _first_diff
from benchmarks.ascend_prefix_fix_trial import load_replay_model
from benchmarks.ascend_prefix_guard_audit import audit_diagnosis
from benchmarks.ascend_prefix_guard_runner import guarded_burst, native_burst
from benchmarks.ascend_prefix_matrix import git_info, save_report
from benchmarks.ascend_prefix_workload import Case


def _matches(run, rows):
    if len(rows) != len(run["tokens"]):
        return False
    for i, row in enumerate(rows):
        rid = f"target-{i}"
        if (
            row.get("prompt_sha256") != run["prompt_sha256"][rid]
            or row["tokens"] != run["tokens"][rid]
            or row["prefill_start"] != row["hit_tokens"]
            or row["hit_tokens"] != run["hit_tokens"][rid]
        ):
            return False
    return True


def _diffs(reference, candidate):
    if candidate is None:
        return None
    return {
        rid: pos
        for rid in reference
        if (pos := _first_diff(reference[rid], candidate[rid])) is not None
    }


def evaluate_trial(model, backend, case, contract, entry, threshold, iteration):
    modes = ("off", "reuse", "guard")
    order = modes[iteration % 3 :] + modes[: iteration % 3]
    runs = {}
    for mode in order:
        if mode == "guard":
            runs[mode] = guarded_burst(model, backend, case, contract, threshold)
        else:
            runs[mode] = native_burst(model, backend, case, contract, mode)
    off, reuse, guard = (runs[m] for m in modes)
    matches = {
        phase: [
            i
            for i, trial in enumerate(entry.get(phase, []), 1)
            if all(_matches(runs[a], trial[a]["requests"]) for a in ("off", "reuse"))
        ]
        for phase in PHASES
    }
    controls = []
    if not any(matches.values()):
        controls.append("no_joint_archived_native_trial")
    if not off["prompt_sha256"] == reuse["prompt_sha256"] == guard["prompt_sha256"]:
        controls.append("target_prompts")
    hot_schedule = guard["hot_schedule"]
    if hot_schedule != reuse["schedule"][: len(hot_schedule)]:
        controls.append("hot_schedule_changed")
    for rid, observed in guard["hot_observed"].items():
        if reuse["tokens"][rid][: len(observed)] != observed:
            controls.append(f"{rid}:hot_prefix_changed")
        hits = guard["hit_tokens"][rid]
        if hits is not None and hits != reuse["hit_tokens"][rid]:
            controls.append(f"{rid}:hot_hit_changed")
    fallback = guard["cold"]
    if fallback is not None and any(
        fallback[key] != off[key] for key in ("tokens", "prompt_sha256", "hit_tokens", "schedule")
    ):
        controls.append("cold_fallback_did_not_reproduce_independent_off")

    native_diff = _diffs(off["tokens"], reuse["tokens"])
    guarded_diff = _diffs(off["tokens"], guard["tokens"])
    comparable = not controls
    if not comparable:
        result = "incomparable"
    elif guard["prefix_conflicts"]:
        result = "late"
    elif guarded_diff != {}:
        result = "missed"
    elif native_diff:
        result = "recovered"
    elif guard["fallback_used"]:
        result = "unnecessary_replay"
    else:
        result = "clean"
    return dict(
        case=case.name,
        iteration=iteration + 1,
        order=order,
        comparable=comparable,
        controls=controls,
        archived_matches=matches,
        result=result,
        native_first_diff=native_diff,
        guarded_first_diff=guarded_diff,
        prefix_conflicts=guard["prefix_conflicts"],
        trigger=guard["trigger"],
        fallback_used=guard["fallback_used"],
        target_ms=dict(
            off=off["target_ms"],
            reuse=reuse["target_ms"],
            guarded=guard["total_target_ms"],
            guarded_hot=guard["hot_target_ms"],
            cold_replay=guard["cold_target_ms"],
        ),
        runs=runs,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diag", type=Path, help="previous diagnosis with per-step logits; usable on its own"
    )
    parser.add_argument("--report", type=Path, help="source matrix JSON for the live trial")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path, help="new JSON path for live results")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case", action="append", help="select one or more archived cases")
    group.add_argument("--all", action="store_true", help="run every case in the matrix")
    parser.add_argument(
        "--threshold", type=float, default=0.0625, help="inclusive hot top-2 gap for live replay"
    )
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold) or args.threshold < 0 or args.repeats < 1:
        parser.error("expected a finite nonnegative threshold and positive repeats")
    if any((args.report, args.model, args.out)) and not all((args.report, args.model, args.out)):
        parser.error("live comparison requires --report, --model and --out together")
    if not args.report and not args.diag:
        parser.error("supply --diag or a live matrix comparison")

    diag = json.loads(args.diag.read_text()) if args.diag else None
    audits = {}
    if diag is not None:
        if diag.get("state") != "complete":
            parser.error("supply a completed diagnosis")
        audits = audit_diagnosis(diag, sorted(set((0, 0.0625, 0.125, 0.25, args.threshold))))
        for case, replays in audits.items():
            if args.case and case not in args.case:
                continue
            for row in replays:
                print(
                    "GUARD_AUDIT",
                    json.dumps(dict(case=case, **row), separators=(",", ":")),
                    flush=True,
                )
    if not args.report:
        if args.case and set(args.case).difference(audits):
            parser.error("one or more selected cases have no archived diagnosis")
        return int(
            any(
                not r["comparable"]
                for c, rows in audits.items()
                if not args.case or c in args.case
                for r in rows
            )
        )

    matrix = json.loads(args.report.read_text())
    if matrix.get("schema") != 1 or matrix.get("state") != "complete":
        parser.error("supply a complete schema 1 matrix JSON")
    contract = matrix["contract"]
    if diag is not None and diag["contract"] != contract:
        parser.error("matrix and diagnosis must share the execution contract")
    specs = {spec["name"]: Case(**spec) for spec in matrix["metadata"]["cases"]}
    default = [n for n, e in matrix["cases"].items() if e["summary"]["status"] != "PASS"]
    default.extend(
        n for n in ("p1024_s8", "total512_p384", "c4_h4", "p512_s8", "c4_h1", "c4_h0") if n in specs
    )
    selected = list(dict.fromkeys(args.case or (list(matrix["cases"]) if args.all else default)))
    if not selected or set(selected).difference(specs) or set(selected).difference(matrix["cases"]):
        parser.error("select one or more cases present in the matrix")
    model = load_replay_model(parser, args, contract)
    report = dict(
        contract=contract,
        commit=git_info("rev-parse", "HEAD"),
        source=str(args.report),
        threshold=args.threshold,
        archive_audit=audits,
        cases={},
        state="running",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=False)
    try:
        for name in selected:
            report["cases"][name] = rows = []
            for i in range(args.repeats):
                row = evaluate_trial(
                    model,
                    contract["backend"],
                    specs[name],
                    contract,
                    matrix["cases"][name],
                    args.threshold,
                    i,
                )
                rows.append(row)
                compact = {
                    k: v for k, v in row.items() if k not in ("runs", "archived_matches", "order")
                }
                compact["target_ms"] = {k: round(v, 2) for k, v in compact["target_ms"].items()}
                print("GUARD_TRIAL", json.dumps(compact, separators=(",", ":")), flush=True)
                save_report(args.out, report)
        rows = [r for batch in report["cases"].values() for r in batch]
        valid = [r for r in rows if r["comparable"]]
        review = dict(
            bursts=len(rows),
            comparable=len(valid),
            hot_mismatches=sum(bool(r["native_first_diff"]) for r in valid),
            cold_replays=sum(r["fallback_used"] for r in valid),
            recovered=sum(r["result"] == "recovered" for r in valid),
            unnecessary_replays=sum(r["result"] == "unnecessary_replay" for r in valid),
            rejected_or_mismatched=sum(r["result"] in ("late", "missed") for r in valid),
        )
        report.update(state="complete", review=review)
        print("GUARD_REVIEW", json.dumps(review, separators=(",", ":")), flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(state="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_report(args.out, report)
    return int(review["comparable"] != review["bursts"] or bool(review["rejected_or_mismatched"]))


if __name__ == "__main__":
    raise SystemExit(main())
