"""Time ten alternating framework A/B pairs on the four cold targets of c4_h2."""

import csv
import json
import math
import os
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from statistics import median
from unittest.mock import patch

if __package__:
    from . import ascend_prefix_framework_regression as framework
else:  # The recipe exports the driver beside the server's locally patched FP32 matrix.
    import ascend_prefix_framework_regression as framework

paired = framework.paired


def checked_miss(run, mode, args, kwargs, *, pair=None, profile=None):
    trial = run.run_variant(mode, None, args, kwargs, profile_target=profile)
    rows = trial["requests"]
    hits = [r["hit_tokens"] for r in rows]
    starts = [r["prefill_start"] for r in rows]
    times = [r["ttft_ms"] for r in rows]
    issues = paired.token_diffs(run.reference, trial)
    if (trial["prefix_queries"], trial["prefix_hits"], hits, starts) != (4, 0, [0] * 4, [0] * 4):
        issues.append(
            dict(
                check="four_cold_targets",
                queries=trial["prefix_queries"],
                hits=trial["prefix_hits"],
                hit_tokens=hits,
                prefill_starts=starts,
            )
        )
    if not all(math.isfinite(t) and t > 0 for t in times):
        raise paired.GateFailure(f"{mode} miss pair={pair}: invalid TTFT values {times}")
    run.record(
        "trace_miss" if profile else "timed_miss",
        pair=pair,
        mode=mode,
        requests=rows,
        prefix_queries=trial["prefix_queries"],
        prefix_hits=trial["prefix_hits"],
        issues=issues,
    )
    if issues:
        raise paired.GateFailure(f"{mode} miss pair={pair}: {issues}")
    return median(times)


def measure(run, args, kwargs, repeats, out):
    rows = []
    with (out / "pairs.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "pair",
                "first",
                "second",
                "original_ms",
                "framework_both_ms",
                "extra_ms",
            ),
        )
        writer.writeheader()
        for pair in range(1, repeats + 1):
            order = run.modes if pair % 2 else run.modes[::-1]
            times = {mode: checked_miss(run, mode, args, kwargs, pair=pair) for mode in order}
            row = dict(
                pair=pair,
                first=order[0],
                second=order[1],
                original_ms=times["original"],
                framework_both_ms=times[run.candidate],
                extra_ms=times[run.candidate] - times["original"],
            )
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            print(
                f"MISS_PAIR {pair}/{repeats} {order[0]} first  "
                f"original={row['original_ms']:.2f} both={row['framework_both_ms']:.2f} "
                f"extra={row['extra_ms']:+.2f} ms",
                flush=True,
            )
    return rows


def summarize(rows):
    deltas = [r["extra_ms"] for r in rows]
    by_first = {}
    for first in dict.fromkeys(r["first"] for r in rows):
        group = [r["extra_ms"] for r in rows if r["first"] == first]
        by_first[first] = dict(pairs=len(group), median_extra_ms=median(group))
    return dict(
        pairs=len(rows),
        slower_pairs=sum(d > 0 for d in deltas),
        original_ms=median(r["original_ms"] for r in rows),
        framework_both_ms=median(r["framework_both_ms"] for r in rows),
        median_extra_ms=median(deltas),
        paired_extra_ms=deltas,
        by_first=by_first,
    )


def should_trace(summary, threshold_ms):
    # A collection trigger for the reported ~20 ms effect, independent of correctness.
    orders = summary["by_first"]
    return (
        threshold_ms is not None
        and len(orders) == 2
        and summary["slower_pairs"] >= math.ceil(0.8 * summary["pairs"])
        and all(v["median_extra_ms"] >= threshold_ms for v in orders.values())
    )


def trace_misses(run, args, kwargs, out):
    mixed = framework.cold.mixed
    profiler = mixed.npu_profiler()
    files = {}
    for mode in run.modes:
        directory = out / "traces" / mode
        directory.mkdir(parents=True)
        profile = partial(
            mixed.first_tokens, profiler=profiler, directory=directory, mode=f"miss_{mode}"
        )
        checked_miss(run, mode, args, kwargs, profile=profile)
        files[mode] = [
            str(p.relative_to(out))
            for p in sorted(directory.rglob("trace_view.json"))
            if p.stat().st_size
        ]
        if not files[mode]:
            raise RuntimeError(f"no nonempty trace_view.json under {directory}")
    return files


def main(argv=None):
    parser = paired.argument_parser(framework.RegressionCase, __doc__, reference_required=True)
    parser.set_defaults(repeats=10, chunk_rows=256)
    parser.add_argument(
        "--trace-if-slower-ms",
        type=float,
        metavar="MS",
        help="after timing, trace both miss modes if >=80%% of pairs are slower "
        "and each run order has median extra latency >= MS",
    )
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.chunk_rows < 1 or not args.model.is_dir():
        parser.error("provide a local checkpoint and positive chunk rows and repeats")
    threshold = args.trace_if_slower_ms
    if threshold is not None and not (math.isfinite(threshold) and threshold > 0):
        parser.error("--trace-if-slower-ms must be finite and positive")
    case = next(c for c in framework.suites()["full"] if c.name == "c4_h2")
    archive = json.loads(args.reference.read_text())
    ref = framework.references(archive, [case], args.chunk_rows)[case.name]
    out = args.out_dir or Path("results") / datetime.now(timezone.utc).strftime(
        "miss-control-c4_h2-%Y%m%dT%H%M%S%fZ"
    )
    out.mkdir(parents=True, exist_ok=False)
    print("RESULTS", out.resolve(), flush=True)
    audit_matrix = out / "audit-matrix.json"
    review = dict(
        status="RUNNING",
        case=case.name,
        arm="miss",
        requested_pairs=args.repeats,
        excluded_audit_pairs_per_arm=1,
        precision="all Linear FP32, fixed padded rows",
        chunk_rows=args.chunk_rows,
        seed=168,
        output_tokens=16,
        reference=str(args.reference),
        revision=os.environ.get("RLT_FRAMEWORK_REVISION"),
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        extra_ms_definition="framework_both minus original, burst median TTFT",
        trace_if_slower_ms=threshold,
    )
    with (out / "trials.jsonl").open("x") as stream:
        run = framework.RegressionCase(case, paired.matrix.run_trial, stream, ref)

        def trial(model, backend, scenario, arm, tokens, seed, *extra, **kwargs):
            # One genuine off/miss/reuse matrix iteration audits both modes. This also
            # verifies the switches take effect on two hits before timing only misses.
            result = run(model, backend, scenario, arm, tokens, seed, *extra, **kwargs)
            if run.gates == 1:
                contract = json.loads(audit_matrix.read_text())["contract"]
                if contract["checkpoint"] != archive["contract"]["checkpoint"]:
                    raise paired.GateFailure("checkpoint differs from the archived reference")
                review["audit"] = {
                    mode: {a: counters for a, counters in audits.items() if a in ("miss", "reuse")}
                    for mode, audits in run.audits.items()
                }
                # Stay inside the local matrix's active all-Linear precision context.
                miss = (model, backend, scenario, "miss", tokens, seed, *extra)
                rows = measure(run, miss, kwargs, args.repeats, out)
                review["ttft"] = summarize(rows)
                if should_trace(review["ttft"], threshold):
                    review["trace"] = "collecting"
                    paired.matrix.save_report(out / "review.json", review)
                    review["trace_files"] = trace_misses(run, miss, kwargs, out)
                    review["trace"] = "collected"
                else:
                    review["trace"] = "disabled" if threshold is None else "trigger_not_met"
            return result

        try:
            with patch.object(paired.matrix, "run_trial", trial):
                status = paired.matrix.main(
                    [
                        "--model",
                        str(args.model),
                        "--device",
                        "npu",
                        "--case",
                        case.name,
                        "--linear-fp32-chunk",
                        str(args.chunk_rows),
                        "--output-tokens",
                        "16",
                        "--seed",
                        "168",
                        "--warmups",
                        "0",
                        "--repeats",
                        "1",
                        "--out",
                        str(audit_matrix),
                    ]
                )
            if status or run.gates != 1 or "ttft" not in review:
                raise paired.GateFailure("incomplete matrix audit or miss timings")
            review["status"] = "PASS"
        except paired.GateFailure as exc:
            review.update(status="FAIL", error=str(exc))
            if run.failures:
                review["audit_failures"] = run.failures
        except (Exception, SystemExit, KeyboardInterrupt) as exc:
            review.update(status="ERROR", error=repr(exc))
            raise
        finally:
            if review.get("trace") == "collecting":
                review["trace"] = "failed"
            paired.matrix.save_report(out / "review.json", review)
            print(
                "MISS_CONTROL_REVIEW",
                review["status"],
                json.dumps(review.get("ttft", {})),
                f"trace={review.get('trace', 'not_run')}",
                review.get("error", ""),
                flush=True,
            )
    return int(review["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
