"""Compare unmodified scheduling and both framework batching switches on the local FP32 matrix."""

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

if __package__:
    from . import ascend_prefix_cold_step as cold
    from .ascend_prefix_scenarios import C4_H3, suites
else:  # Exported drivers share the server checkout's patched matrix and workload.
    import ascend_prefix_cold_step as cold
    from ascend_prefix_scenarios import C4_H3, suites

paired = cold.mixed.paired


class RegressionCase(cold.FrameworkCold):
    label = "FRAMEWORK"
    candidate = "framework_both"
    batched_suffix_mode = candidate
    stock_methods = (*cold.FrameworkCold.stock_methods, "_plan_suffix_prefills")

    def __init__(self, case, *args, **kwargs):
        super().__init__(*args, case_name=case.name, **kwargs)
        self.case = case
        self.failures = []
        self.gates = 0

    def execution_options(self, mode):
        enabled = mode == self.candidate
        return dict(ascend_batch_cold_prefill=enabled, ascend_batch_cached_suffix=enabled)

    def record(self, kind, **fields):
        super().record(kind, case=self.case_name, **fields)
        if kind == "gate":
            self.gates += 1
            if not fields["passed"]:
                self.failures.append(dict(repeat=fields["repeat"], issues=fields["issues"]))

    def audit_issues(self, audits, calls):
        c = self.case
        issues = self.concurrency_audit_issues(c, audits, calls) if c.group == "concurrency" else []
        for mode, arms in audits.items():
            for arm, audit in arms.items():
                eligible = mode == self.candidate and arm == "reuse" and c.concurrency > 1
                hot = c.hits or 0
                expected = {
                    "cold_calls": calls if eligible and hot and c.concurrency - hot >= 2 else 0,
                    "suffix_calls": calls if eligible and hot >= 2 else 0,
                }
                for key, value in expected.items():
                    if audit[key] != value:
                        issues.append(
                            dict(mode=mode, arm=arm, check=key, expected=value, actual=audit[key])
                        )
        return issues


def references(archive, cases, chunk_rows):
    if archive.get("state") != "complete":
        raise ValueError("reference matrix did not complete")
    contract = archive["contract"]
    if contract["seed"] != 168 or contract["output_tokens"] != 16:
        raise ValueError("reference must use seed 168 and 16 output tokens")
    saved = paired.reference_chunk_rows(archive)
    if saved is not None and saved != chunk_rows:
        raise ValueError(f"reference was produced with --linear-fp32-chunk {saved}")
    refs = {}
    for case in cases:
        if case.name == C4_H3.name:
            refs[case.name] = None
            continue
        if case.name not in archive["cases"]:
            raise ValueError(f"{case.name} is missing from the archived reference")
        entry = archive["cases"][case.name]
        if entry["summary"]["status"] != "PASS" or entry["summary"]["case"] != asdict(case):
            raise ValueError(
                f"reference must contain a passing run of the same {case.name} scenario"
            )
        refs[case.name] = entry["trials"][0]["off"]
    return refs


def case_review(run):
    reference = "live original/off" if run.case_name == C4_H3.name else "archived FP32/off"
    if run.gates != run.repeats + 1 or any(run.counts[arm] != run.gates for arm in paired.ARMS):
        return dict(status="INCOMPLETE", reference=reference, gates=run.gates)
    if run.failures:
        return dict(status="FAIL", reference=reference, gates=run.gates, failures=run.failures)
    return dict(
        status="PASS",
        reference=reference,
        gates=run.gates,
        ttft=paired.ttft_review(run.runs[1:], run.candidate),
    )


def slower_in_every_repeat(cases):
    slower = []
    for name, row in cases.items():
        if row["status"] != "PASS":
            continue
        for arm in paired.ARMS:
            ttft = row["ttft"][arm]
            if all(value < 0 for value in ttft["paired_saved_ms"]):
                slower.append(
                    dict(
                        case=name,
                        arm=arm,
                        extra_ms=-ttft["median_paired_saved_ms"],
                        paired_saved_ms=ttft["paired_saved_ms"],
                    )
                )
    return sorted(slower, key=lambda row: -row["extra_ms"])


def write_results(out, review):
    paired.matrix.save_report(out / "review.json", review)
    with (out / "ttft.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "case",
                "status",
                "reference",
                "arm",
                "original_ms",
                "framework_both_ms",
                "median_paired_saved_ms",
                "paired_saved_ms",
            )
        )
        for name, row in review["cases"].items():
            if row["status"] != "PASS":
                writer.writerow((name, row["status"], row["reference"], "", "", "", "", ""))
                continue
            for arm, values in row["ttft"].items():
                writer.writerow(
                    (
                        name,
                        row["status"],
                        row["reference"],
                        arm,
                        values["original_ms"],
                        values["framework_both_ms"],
                        values["median_paired_saved_ms"],
                        json.dumps(values["paired_saved_ms"]),
                    )
                )


def main(argv=None):
    parser = paired.argument_parser(RegressionCase, __doc__, reference_required=True)
    parser.add_argument(
        "--stage",
        choices=("focus", "full"),
        default="focus",
        help="focus: c4_h3 and c4_h4; full: the original 29 cases",
    )
    args = parser.parse_args(argv)
    rows = args.chunk_rows if args.chunk_rows is not None else 256
    if rows < 1 or args.repeats < 1 or not args.model.is_dir():
        parser.error("provide a local model, positive chunk rows and repeats")
    presets = suites()
    full = presets["full"]
    cases = [C4_H3, next(c for c in full if c.name == "c4_h4")] if args.stage == "focus" else full
    archive = json.loads(args.reference.read_text())
    refs = references(archive, cases, rows)
    out = args.out_dir or Path("results") / datetime.now(timezone.utc).strftime(
        f"framework-{args.stage}-%Y%m%dT%H%M%S%fZ"
    )
    out.mkdir(parents=True, exist_ok=False)
    review = dict(
        stage=args.stage,
        state="running",
        selected=list(refs),
        cases={},
        reference=str(args.reference),
        chunk_rows=rows,
        repeats=args.repeats,
        warmups=1,
        seed=168,
        output_tokens=16,
        modes=["original", "framework_both"],
        framework_revision=os.environ.get("RLT_FRAMEWORK_REVISION"),
    )
    print("RESULTS", out.resolve(), flush=True)
    write_results(out, review)
    with (out / "trials.jsonl").open("x") as stream:
        native = paired.matrix.run_trial
        runs = {}
        identify = paired.matrix.checkpoint_identity

        def check_checkpoint(path):
            identity = identify(path)
            if identity != archive["contract"]["checkpoint"]:
                raise ValueError("checkpoint files differ from the FP32 reference")
            return identity

        def trial(model, backend, case, arm, *extra, **kwargs):
            if case.name not in runs:
                runs[case.name] = RegressionCase(
                    case, native, stream, refs[case.name], repeats=args.repeats
                )
            run = runs[case.name]
            try:
                result = run(model, backend, case, arm, *extra, **kwargs)
            except paired.GateFailure:
                # Preserve the measured original for the matrix. Finish independent
                # cases, and exclude this failed case from all performance summaries.
                result = run.runs[-1]["original"][arm]
            if run.gates == args.repeats + 1:
                entry = review["cases"][case.name] = case_review(run)
                print("FRAMEWORK_CASE", case.name, entry["status"], flush=True)
                write_results(out, review)
            return result

        # The server's FP32 matrix may predate the supplemental named case.
        # Explicit names keep both historical suite membership and evaluation order.
        with (
            patch.object(paired.matrix, "run_trial", trial),
            patch.object(paired.matrix, "checkpoint_identity", check_checkpoint),
            patch.object(paired.matrix, "suites", return_value={**presets, "full": [*full, C4_H3]}),
        ):
            try:
                status = paired.matrix.main(
                    [
                        "--model",
                        str(args.model),
                        "--device",
                        "npu",
                        "--linear-fp32-chunk",
                        str(rows),
                        "--seed",
                        "168",
                        "--output-tokens",
                        "16",
                        "--warmups",
                        "1",
                        "--repeats",
                        str(args.repeats),
                        "--out",
                        str(out / "original-matrix.json"),
                        *[arg for c in cases for arg in ("--case", c.name)],
                    ]
                )
                review["state"] = (
                    "PASS"
                    if (
                        not status
                        and review["cases"].keys() == refs.keys()
                        and all(e["status"] == "PASS" for e in review["cases"].values())
                    )
                    else "FAIL"
                )
            except (Exception, SystemExit, KeyboardInterrupt) as exc:
                review.update(state="ERROR", error=repr(exc))
                raise
            finally:
                for name, run in runs.items():
                    if name not in review["cases"]:
                        review["cases"][name] = case_review(run)
                review["pending"] = [n for n in refs if n not in review["cases"]]
                review["failed"] = [n for n, r in review["cases"].items() if r["status"] != "PASS"]
                review["slower_in_every_repeat"] = slower_in_every_repeat(review["cases"])
                write_results(out, review)
                print(
                    "FRAMEWORK_REVIEW",
                    review["state"],
                    f"cases={len(review['cases'])}",
                    f"failed={review['failed']}",
                    f"pending={review['pending']}",
                    flush=True,
                )
    return int(review["state"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
