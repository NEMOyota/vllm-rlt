"""Replay native C4 with a verified copy of the cold target's logical prefix pages."""

import argparse
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path

from benchmarks.ascend_prefix_evidence import _arm_checks
from benchmarks.ascend_prefix_fix_trial import (
    _differences,
    _identity,
    _schedule,
    load_replay_inputs,
)
from benchmarks.ascend_prefix_kv_probe import KVProbe
from benchmarks.ascend_prefix_logical_kv import LogicalKVProbe
from benchmarks.ascend_prefix_logits_probe import LogitsProbe
from benchmarks.ascend_prefix_matrix import git_info, save_report
from benchmarks.ascend_prefix_page_control import ColdPages, DonorPageControl
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_scenarios import suites
from benchmarks.ascend_prefix_workload import expected_hits


def c4_experiment(model, backend, case, contract, priors):
    if case.concurrency != 4 or case.hits != 4 or not case.reusable_tokens or case.churn:
        raise ValueError("expected four hot targets and a page-aligned shared prefix")
    focus = "target-2"
    p = case.reusable_tokens
    runs, pages, controls = {}, None, []
    for name in ("off", "reuse", "sham", "cold_pages"):
        arm = "off" if name == "off" else "reuse"
        probes = {}

        @contextmanager
        def lifetime(engine, targets, donor):
            if not all(t[:p] == donor[:p] for t in targets):
                raise ValueError("the donor and all four targets must share the observed prefix")
            cache = engine.cache_manager
            rids = [f"target-{i}" for i in range(4)] if name == "off" else ["donor"]
            with ExitStack() as stack:
                probes["kv"] = stack.enter_context(KVProbe(cache, expected_hits(case, arm)[1]))
                probes["fingerprints"] = stack.enter_context(LogicalKVProbe(cache, p, rids))
                # Install last: the following free hook runs before the two witnesses.
                if name == "off":
                    probes["capture"] = stack.enter_context(ColdPages(cache, p, focus))
                if name in ("sham", "cold_pages"):
                    replacement = pages if name == "cold_pages" else None
                    probes["page_copy"] = stack.enter_context(
                        DonorPageControl(cache, p, replacement)
                    )
                yield

        def target(engine, targets, donor):
            probes["logits"] = LogitsProbe(engine.model_runner)
            return probes["logits"]

        run = replay_mode(
            model,
            backend,
            case,
            arm,
            contract["output_tokens"],
            contract["seed"],
            engine_context=lifetime,
            target_context=target,
        )
        if name == "off":
            pages = probes.pop("capture").pages
        if "page_copy" in probes:
            run["donor_before_sha256"] = probes["page_copy"].before_sha256
        run.update({key: probe.report() for key, probe in probes.items()})
        run["output_tokens_per_request"] = contract["output_tokens"]
        checks = _arm_checks(run)
        controls.extend(f"{name}:{s}" for s in checks["issues"] + checks["probe_issues"])
        runs[name] = run

    cold, native, sham, copied = (runs[n] for n in ("off", "reuse", "sham", "cold_pages"))
    matches = [
        i
        for i, prior in enumerate(priors, 1)
        if all(
            _identity(prior["runs"][m]) == _identity(runs[m])
            and (
                "logits" not in prior["runs"][m] or runs[m]["logits"] == prior["runs"][m]["logits"]
            )
            for m in ("off", "reuse")
        )
    ]
    for name in ("sham", "cold_pages"):
        run = runs[name]
        if (
            _schedule(run) != _schedule(native)
            or run["donor_prefill_spans"] != native["donor_prefill_spans"]
        ):
            controls.append(f"{name}:schedule")
        if (
            run["kv"]["donor_prefix_pages"] != native["kv"]["donor_prefix_pages"]
            or run["kv"]["targets"] != native["kv"]["targets"]
        ):
            controls.append(f"{name}:page_tables")
    original_kv = native["fingerprints"]["donor"]
    focus_kv = cold["fingerprints"][focus]
    sham_diff = _differences(native, sham)
    if (
        sham["fingerprints"]["donor"] != original_kv
        or sham["page_copy"]["changed_pages"]
        or sham_diff
        or sham["logits"] != native["logits"]
    ):
        controls.append("sham:changed_native_replay")
    if copied["fingerprints"]["donor"] != focus_kv:
        controls.append("cold_pages:readback_differs_from_cold_target")
    if copied["donor_before_sha256"] != sham["donor_before_sha256"]:
        controls.append("cold_pages:starting_donor_changed")

    # A score at a fixed generated position is comparable to the cold arm
    # only when the complete preceding token history of this request matches.
    histories = {
        n: next(r["tokens"] for r in run["requests"] if r["request_id"] == focus)
        for n, run in runs.items()
    }
    scores = {}
    for pos in (7, 8):
        if pos <= contract["output_tokens"]:
            scores[pos] = {
                n: run["logits"][focus][pos - 1]
                if histories[n][: pos - 1] == histories["off"][: pos - 1]
                else None
                for n, run in runs.items()
                if n != "sham"
            }
    summary = dict(
        case=case.name,
        focus=focus,
        comparable=bool(matches) and not controls,
        matched_prior_replays=matches,
        controls=controls,
        cold_vs_donor={rid: kv == original_kv for rid, kv in cold["fingerprints"].items()},
        cold_vs_focus={rid: kv == focus_kv for rid, kv in cold["fingerprints"].items()},
        native_off_vs_reuse=_differences(cold, native),
        sham_vs_reuse=sham_diff,
        off_vs_cold_pages=_differences(cold, copied),
        sham_pages=sham["page_copy"],
        cold_pages=copied["page_copy"],
        hit_tokens=[r["hit_tokens"] for r in copied["requests"]],
        scores_on_cold_history=scores,
    )
    return {"summary": summary, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    diag, model = load_replay_inputs(parser, args, ["c4_h4"])
    case = next(c for c in suites()["full"] if c.name == "c4_h4")
    result = c4_experiment(
        model, diag["contract"]["backend"], case, diag["contract"], diag["cases"][case.name]
    )
    result.update(
        contract=diag["contract"],
        commit=git_info("rev-parse", "HEAD"),
        diagnostic_commit=diag.get("diagnostic_commit"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=False)
    save_report(args.out, result)
    print(
        "C4_ORACLE",
        json.dumps(result["summary"], separators=(",", ":"), allow_nan=False),
        flush=True,
    )
    return int(not result["summary"]["comparable"])


if __name__ == "__main__":
    raise SystemExit(main())
