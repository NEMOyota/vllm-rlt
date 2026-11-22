"""Compare prefix-cache workloads on one synchronous Ouro engine. Run from the repo root."""

import argparse
import csv
import json
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from benchmarks.ascend_prefix_history import compare_baseline
from benchmarks.ascend_prefix_scenarios import ARMS, C4_H3, suites, summarize
from benchmarks.ascend_prefix_workload import Case, run_trial
from benchmarks.preemption import checkpoint_identity
from vllm_rlt.device import resolve_device
from vllm_rlt.models import OuroForCausalLM


def git_info(*args):
    proc = subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def save_report(path, report):
    # The destination is reserved before the first trial. Replace only complete JSON documents.
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        json.dump(report, tmp, ensure_ascii=False, allow_nan=False)
    Path(tmp.name).replace(path)


def row_fields(name, entry):
    summary = entry.get("summary")
    if not summary:
        return {"case": name, "status": "INCOMPLETE"}
    arms = summary["arms"]
    row = {"case": name, "status": summary["status"]}
    for arm in ARMS:
        metrics = arms[arm]
        for key in ("ttft_p50_ms", "ttft_p95_ms", "e2e_p50_ms", "output_tok_s"):
            row[f"{arm}_{key}"] = round(metrics[key], 2)
    reuse = arms["reuse"]
    row["reuse_hit_tokens_per_request"] = round(reuse["hit_tokens"] / reuse["request_count"], 2)
    for label, metrics in summary["comparisons"].items():
        for key in ("ttft_p50_saved_ms", "ttft_p50_saved_pct", "output_tok_s_gain_pct"):
            row[f"{label}_{key}"] = round(metrics[key], 2) if summary["status"] == "PASS" else None
    history = entry.get("history")
    if history:
        row["history"] = history["status"]
        if history["status"] == "PASS":
            for arm in ARMS:
                row[f"history_{arm}_ttft_p50_saved_ms"] = round(
                    history["arms"][arm]["ttft_p50_saved_ms"], 2
                )
    return row


def print_row(name, entry):
    row = row_fields(name, entry)
    if row["status"] != "PASS":
        summary = entry.get("summary", {})
        print(name, row["status"], summary.get("mismatches", [])[:1], flush=True)
        return
    print(
        f"{name} PASS  TTFT p50 ms off={row['off_ttft_p50_ms']:.2f} "
        f"miss={row['miss_ttft_p50_ms']:.2f} reuse={row['reuse_ttft_p50_ms']:.2f}  "
        f"miss-reuse={row['reuse_vs_miss_ttft_p50_saved_ms']:+.2f}  "
        f"hit_tokens/req={row['reuse_hit_tokens_per_request']:g}",
        end="",
    )
    if history := entry.get("history"):
        if history["status"] == "PASS":
            print(
                f"  historical reuse saved={row['history_reuse_ttft_p50_saved_ms']:+.2f} ms", end=""
            )
        else:
            print("  historical", history, end="")
    print(flush=True)


def write_csv(report, path):
    rows = [row_fields(name, entry) for name, entry in report["cases"].items()]
    if rows:
        columns = dict.fromkeys(key for row in rows for key in row)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)


def review(report):
    # Rank measured regressions; use paired signs to make run-to-run noise visible.
    result = {"state": report["state"], "cases": len(report["cases"]), "check": []}
    for label, candidate in (("reuse_vs_miss", "reuse"), ("miss_vs_off", "miss")):
        slower = []
        for name, entry in report["cases"].items():
            summary = entry.get("summary", {})
            if (
                summary.get("status") != "PASS"
                or entry.get("history", {}).get("status", "PASS") != "PASS"
            ):
                if name not in result["check"]:
                    result["check"].append(name)
                continue
            if candidate == "reuse" and not summary["arms"]["reuse"]["hit_requests"]:
                continue
            diff = summary["comparisons"][label]
            cost = -diff["ttft_p50_saved_ms"]
            if cost > 0:
                pairs = diff["paired_trial_ttft_p50_saved_ms"]
                slower.append(
                    dict(
                        case=name,
                        extra_ms=round(cost, 2),
                        slower_pairs=sum(p < 0 for p in pairs),
                        pairs=len(pairs),
                    )
                )
        result[label] = sorted(slower, key=lambda row: -row["extra_ms"])[:3]
    print("MATRIX_REVIEW", json.dumps(result, separators=(",", ":")), flush=True)


def compare(entry, contract, baseline):
    entry["history"] = compare_baseline(contract, baseline, entry["summary"], entry["trials"])


def evaluate(case, model, backend, args, entry):
    for index in range(args.warmups + args.repeats):
        warm = index < args.warmups
        bucket = entry["warmups"] if warm else entry["trials"]
        trial = {}
        bucket.append(trial)
        offset = index % len(ARMS)
        for arm in ARMS[offset:] + ARMS[:offset]:
            entry["progress"] = {"warmup": warm, "iteration": len(bucket), "arm": arm}
            trial[arm] = run_trial(model, backend, case, arm, args.output_tokens, args.seed)
    summary = summarize(case, entry["trials"])
    if entry["warmups"]:
        warm = summarize(case, entry["warmups"])
        if warm["status"] != "PASS":
            summary["status"] = "FAIL"
            summary["comparisons"] = {}
            summary["mismatch_count"] += warm["mismatch_count"]
            summary["mismatches"] = [
                dict(row, warmup=True) for row in warm["mismatches"]
            ] + summary["mismatches"]
    entry["summary"] = summary
    del entry["progress"]


def measurement(args, cases, baseline):
    if not args.model or not args.model.is_dir():
        raise ValueError("--model must name a prepared local Ouro checkpoint directory")
    if args.repeats < 1 or args.warmups < 0 or args.output_tokens < 1:
        raise ValueError("repeats and output tokens must be positive, warmups nonnegative")
    if (args.kv_v3_library or args.audit_native_kv) and args.device != "npu":
        raise ValueError("K/V dispatch experiments require --device npu")
    path = args.out or Path(
        datetime.now(timezone.utc).strftime("prefix-matrix-%Y%m%dT%H%M%S%fZ.json")
    )
    if path.suffix != ".json":
        raise ValueError("--out must have a .json extension")
    csv_path = path.with_suffix(".csv")
    if path.exists() or csv_path.exists():
        raise ValueError(f"Results already exist. Choose a new --out path: {path}")
    device = resolve_device("npu:0" if args.device == "npu" else "cpu")
    backend = "ascend_eager" if device.type == "npu" else "torch"
    device_name = platform.machine()
    npu_version = cann = None
    if device.type == "npu":
        import torch_npu
        from torch_npu.utils.collect_env import get_cann_version

        torch.npu.set_device(0)
        device_name = torch.npu.get_device_name(0)
        npu_version = torch_npu.__version__
        if os.environ.get("ASCEND_HOME_PATH"):
            try:
                cann = get_cann_version()
            except (OSError, RuntimeError):
                pass
    contract = dict(
        workload_version=1,
        device=args.device,
        device_name=device_name,
        backend=backend,
        torch_version=str(torch.__version__),
        torch_npu_version=npu_version,
        cann_version=cann,
        checkpoint=checkpoint_identity(args.model),
        seed=args.seed,
        output_tokens=args.output_tokens,
        loops=4,
        block_size=16,
        dtype="bfloat16",
        warmups=args.warmups,
        torch_threads=torch.get_num_threads(),
    )
    if args.kv_v3_library:
        contract["kv_projection"] = "MatMulV3"
    if args.audit_native_kv:
        contract["kv_projection"] = "native-dispatch-audit"
    if "IGNORE_INFER_ERROR" in os.environ:
        contract["ignore_infer_error"] = os.environ["IGNORE_INFER_ERROR"]
    report = dict(
        schema=1,
        state="running",
        contract=contract,
        metadata=dict(
            git_commit=git_info("rev-parse", "HEAD"),
            tracked_changes=git_info("status", "--short", "--untracked-files=no"),
            checkpoint_directory=str(args.model.resolve()),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            cpu_affinity=sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            python=platform.python_version(),
            repeats=args.repeats,
            cases=[asdict(case) for case in cases],
        ),
        cases={},
    )
    if baseline is not None:
        metadata = baseline.get("metadata", {})
        report["metadata"]["baseline"] = dict(
            file=args.baseline.name,
            git_commit=metadata.get("git_commit"),
            timestamp_utc=metadata.get("timestamp_utc"),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=False)
    save_report(path, report)
    print(f"Writing {path} and {csv_path}. Cases={len(cases)} repeats={args.repeats}", flush=True)
    try:
        model = OuroForCausalLM.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
        for case in cases:
            print(f"Starting {case.name}", flush=True)
            entry = {"trials": [], "warmups": []}
            report["cases"][case.name] = entry
            if not args.kv_v3_library:
                evaluate(case, model, backend, args, entry)
            if args.kv_v3_library or args.audit_native_kv:
                from benchmarks.ascend_matmul_v3 import use_v3_kv

                with use_v3_kv(model, args.kv_v3_library) as (calls, shapes):
                    entry["v3_calls"] = calls
                    if args.audit_native_kv:
                        # Observe a separate replay so production F.linear TTFT has no Python hooks.
                        for arm in ARMS:
                            run_trial(model, backend, case, arm, args.output_tokens, args.seed)
                    else:
                        evaluate(case, model, backend, args, entry)
                    entry["v3_shapes"] = sorted(shapes)
                    if not calls["k_proj"] or calls["v_proj"] != calls["k_proj"]:
                        raise RuntimeError("missing or unpaired V3 K/V projections")
            if baseline is not None:
                compare(entry, contract, baseline)
            save_report(path, report)
            print_row(case.name, entry)
        report["state"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        report["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save_report(path, report)
        write_csv(report, csv_path)
    return report


def passed(report):
    return report["state"] == "complete" and all(
        entry["summary"]["status"] == "PASS"
        and entry.get("history", {}).get("status", "PASS") == "PASS"
        for entry in report["cases"].values()
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Prepared local Ouro checkpoint directory")
    parser.add_argument(
        "--device", choices=("npu", "cpu"), default="npu", help="CPU reference for smoke testing"
    )
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    route = parser.add_mutually_exclusive_group()
    route.add_argument(
        "--kv-v3-library", type=Path, help="Use the experiment MatMulV3 bridge for Ouro K/V"
    )
    route.add_argument(
        "--audit-native-kv",
        action="store_true",
        help="Measure native K/V projection shapes for device dispatch verification",
    )
    parser.add_argument(
        "--case", action="append", help="Select a named case (repeatable, see --list)"
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=int,
        nargs=4,
        metavar=("PREFIX", "SUFFIX", "CONCURRENCY", "HOT"),
        help="Custom burst, e.g. --shape 2048 32 4 3 (three hot targets, repeatable)",
    )
    parser.add_argument("--list", action="store_true", help="Print all case definitions and exit")
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument(
        "--warmups", type=int, default=1, help="Excluded target bursts per case and arm"
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="Measured target bursts per case and arm"
    )
    parser.add_argument("--seed", type=int, default=168)
    parser.add_argument(
        "--out", type=Path, help="New JSON report; also writes CSV with the same stem"
    )
    parser.add_argument(
        "--baseline", type=Path, help="Compare with the same arms in a saved matrix JSON"
    )
    parser.add_argument(
        "--read-report", type=Path, help="Print or compare existing results without loading a model"
    )
    args = parser.parse_args(argv)
    presets = suites()
    full = {case.name: case for case in (*presets["full"], C4_H3)}
    if args.list:
        for case in full.values():
            print(json.dumps(asdict(case)))
        return 0
    try:
        baseline = json.loads(args.baseline.read_text()) if args.baseline else None
        if args.baseline and not isinstance(baseline, dict):
            parser.error("--baseline must contain a matrix JSON report")
        if args.read_report:
            report = json.loads(args.read_report.read_text())
            if not isinstance(report, dict) or report.get("schema") != 1 or "cases" not in report:
                parser.error("unsupported matrix report")
            if args.shape or args.out:
                parser.error("--shape and --out apply to new measurements")
            if args.case:
                if set(args.case) - report["cases"].keys():
                    parser.error("--case not present in report")
                report["cases"] = {name: report["cases"][name] for name in args.case}
            for name, entry in report["cases"].items():
                if baseline is not None and "summary" in entry:
                    compare(entry, report["contract"], baseline)
                print_row(name, entry)
            if report["state"] != "complete":
                print(f"Run state: {report['state']}  {report.get('error', '')}")
        else:
            names = list(dict.fromkeys(args.case or []))
            if set(names) - full.keys():
                parser.error("unknown --case (use --list to see names)")
            cases = [full[name] for name in names]
            for p, s, c, h in dict.fromkeys(tuple(shape) for shape in (args.shape or [])):
                cases.append(Case(f"custom_p{p}_s{s}_c{c}_h{h}", "custom", p, s, c, h))
            report = measurement(args, cases or presets[args.suite], baseline)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    review(report)
    return int(not passed(report))


if __name__ == "__main__":
    raise SystemExit(main())
