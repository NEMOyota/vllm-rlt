"""Compare two attention geometries on identical operands and check cold prefix causality."""

import argparse
import json
from pathlib import Path

from benchmarks.ascend_prefix_c4_slot import cold_slot_run
from benchmarks.ascend_prefix_evidence import _arm_checks
from benchmarks.ascend_prefix_fix_trial import _identity, _schedule, load_replay_model
from benchmarks.ascend_prefix_logical_kv import compare_logical_kv
from benchmarks.ascend_prefix_matrix import git_info, save_report
from benchmarks.ascend_prefix_scenarios import suites


def _short_comparison(a, b):
    comparison = compare_logical_kv(a, b)
    first = comparison["first_diff"]
    if first:
        comparison["first_diff"] = {
            k: first[k] for k in ("depth", "layer", "k_changed", "v_changed")
        }
    return comparison


def consumer_experiment(model, backend, case, contract, previous_off):
    focus = "target-2"
    runs = {
        name: cold_slot_run(
            model,
            backend,
            case,
            contract,
            name,
            observe_attention=(backend == "ascend_eager" and name == "base"),
        )
        for name in ("base", "repeat", "future", "clones")
    }
    base, repeat, future, clones = runs.values()
    controls = []
    if (
        _identity(base) != _identity(previous_off)
        or _schedule(base) != _schedule(previous_off)
        or base["fingerprints"] != previous_off["fingerprints"]
    ):
        controls.append("base:archived_cold_run")
    if (
        _identity(base) != _identity(repeat)
        or base["prefill_fingerprints"] != repeat["prefill_fingerprints"]
        or base["fingerprints"] != repeat["fingerprints"]
    ):
        controls.append("repeat:identical_input_drift")
    ids = [r["request_id"] for r in base["requests"]]
    for name, run in runs.items():
        controls.extend(f"{name}:{s}" for s in _arm_checks(run)["issues"])
        if (
            _schedule(run) != _schedule(base)
            or run["donor_prefill_spans"] != base["donor_prefill_spans"]
            or run["donor_sha256"] != base["donor_sha256"]
        ):
            controls.append(f"{name}:schedule")
        if (
            run["prefix_sha256"] != base["prefix_sha256"]
            or len(set(run["prefix_sha256"].values())) != 1
        ):
            controls.append(f"{name}:prefix_tokens")
        if [r["request_id"] for r in run["requests"]] != ids:
            controls.append(f"{name}:request_order")
    prompts = {
        n: {r["request_id"]: r["prompt_sha256"] for r in run["requests"]} for n, run in runs.items()
    }
    if [rid for rid in ids if prompts["future"][rid] != prompts["base"][rid]] != [focus]:
        controls.append("future:prompt_change")
    if set(prompts["clones"].values()) != {prompts["base"][focus]}:
        controls.append("clones:prompt_change")

    a, b, c = (r["prefill_fingerprints"] for r in (base, future, clones))
    summary = dict(
        case=case.name,
        comparable=not controls,
        controls=controls,
        future_prefix_changes={rid: _short_comparison(a[rid], b[rid]) for rid in ids},
        clone_slot_vs_focus={rid: _short_comparison(c[focus], c[rid]) for rid in ids},
        focus_with_other_rows_cloned=_short_comparison(a[focus], c[focus]),
        changed_between_prefill_and_release={
            n: [
                rid
                for rid in ids
                if run["prefill_fingerprints"][rid]["sha256"] != run["fingerprints"][rid]["sha256"]
            ]
            for n, run in runs.items()
        },
        frozen_attention=base.get("attention"),
    )
    return {"summary": summary, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    previous = json.loads(args.oracle.read_text())
    summary = previous.get("summary", {})
    if not summary.get("comparable") or summary.get("case") != "c4_h4":
        parser.error("expected a comparable C4 page-control report")
    contract = previous["contract"]
    model = load_replay_model(parser, args, contract)
    case = next(c for c in suites()["full"] if c.name == "c4_h4")
    result = consumer_experiment(
        model, contract["backend"], case, contract, previous["runs"]["off"]
    )
    result.update(
        contract=contract,
        commit=git_info("rev-parse", "HEAD"),
        oracle_commit=previous.get("commit"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=False)
    save_report(args.out, result)
    summary = result["summary"].copy()
    for key in ("future_prefix_changes", "clone_slot_vs_focus"):
        summary[key] = {rid: diff["first_diff"] for rid, diff in summary[key].items()}
    attention = summary.pop("frozen_attention")
    print("C4_SLOT", json.dumps(summary, separators=(",", ":"), allow_nan=False), flush=True)
    print("C4_ATTN", json.dumps(attention, separators=(",", ":"), allow_nan=False), flush=True)
    return int(not summary["comparable"] or bool(attention and attention["control_issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
