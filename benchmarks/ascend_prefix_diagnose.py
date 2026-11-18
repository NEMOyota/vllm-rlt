"""Diagnose matrix correctness failures with scheduler, KV and greedy sampling records."""

import argparse
import json
import os
from pathlib import Path

import torch

from benchmarks.ascend_prefix_cache import BLOCK_SIZE
from benchmarks.ascend_prefix_evidence import archive_evidence, compare_replays
from benchmarks.ascend_prefix_kv_probe import KVProbe
from benchmarks.ascend_prefix_logits_probe import LogitsProbe
from benchmarks.ascend_prefix_matrix import git_info, save_report
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_scenarios import ARMS
from benchmarks.ascend_prefix_workload import Case, expected_hits
from benchmarks.preemption import checkpoint_identity
from vllm_rlt.device import resolve_device
from vllm_rlt.models import OuroForCausalLM


def observe(model, backend, case, mode, output_tokens, seed):
    probes = {}

    def kv_context(engine, targets, donor):
        _, expected = expected_hits(case, mode)
        probes["kv"] = KVProbe(engine.cache_manager, expected)
        return probes["kv"]

    def logits_context(engine, targets, donor):
        probes["logits"] = LogitsProbe(engine.model_runner)
        return probes["logits"]

    result = replay_mode(
        model,
        backend,
        case,
        mode,
        output_tokens,
        seed,
        engine_context=kv_context,
        target_context=logits_context,
    )
    result.update({key: probe.report() for key, probe in probes.items()})
    result["output_tokens_per_request"] = output_tokens
    return result


def diagnose_pass(model, backend, case, entry, output_tokens, seed, order):
    runs = {mode: observe(model, backend, case, mode, output_tokens, seed) for mode in order}
    return {"runs": runs, "comparison": compare_replays(runs, entry)}


def emit(label, **fields):
    print(label, json.dumps(fields, separators=(",", ":"), ensure_ascii=False), flush=True)


def selected_cases(report, names):
    entries = report["cases"]
    selected = names or [
        name
        for name, entry in entries.items()
        if entry.get("summary", {}).get("status") != "PASS"
        or entry.get("history", {}).get("status", "PASS") != "PASS"
    ]
    specs = {item["name"]: item for item in report["metadata"]["cases"]}
    return {name: Case(**specs[name]) for name in dict.fromkeys(selected)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="Original matrix JSON")
    parser.add_argument("--case", action="append", help="Repeatable. Default: flagged cases")
    parser.add_argument(
        "--model", type=Path, help="Local checkpoint; omit to inspect saved data only"
    )
    parser.add_argument("--out", type=Path, help="New diagnostic JSON file for instrumented replay")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    if bool(args.model) != bool(args.out) or args.repeats < 1:
        parser.error("supply --model and --out together for replay; repeats must be positive")
    report = json.loads(args.report.read_text())
    if report.get("schema") != 1 or "contract" not in report:
        parser.error("expected a schema 1 matrix report")
    try:
        cases = selected_cases(report, args.case)
    except KeyError as exc:
        parser.error(f"no case specification for {exc}")
    archives = {}
    for name, case in cases.items():
        entry = report["cases"].get(name)
        if entry is None:
            parser.error(f"no original measurements for {name}")
        evidence = archive_evidence(entry, case)
        archives[name] = evidence
        for phase, details in evidence["phases"].items():
            emit(
                "ARCHIVE",
                case=name,
                phase=phase,
                failed=details["failed_trials_1based"],
                total=details["iterations"],
                kinds=details["counts_by_kind"],
                first=next(iter(details["examples"]), None),
            )
        if "history" in entry and entry["history"]["status"] != "PASS":
            emit("HISTORY", case=name, history=entry["history"])
    if not cases:
        print("No flagged cases.")
    if not args.model or not cases:
        return 0
    for case in cases.values():
        if (
            case.churn
            or case.tight_pool
            or min(case.prefix, case.prefix + case.suffix - 1) < BLOCK_SIZE
        ):
            parser.error(f"{case.name}: replay needs a full prefix page in an unconstrained pool")
    if not args.model.is_dir() or args.out.exists():
        parser.error("--model must be a local checkpoint directory and --out must name a new file")
    contract = report["contract"]
    if checkpoint_identity(args.model) != contract["checkpoint"]:
        parser.error("checkpoint bytes differ from the archived weights/config")
    device_type = contract["device"]
    if device_type not in ("npu", "cpu"):
        parser.error(f"unsupported archived device {device_type}")
    device = resolve_device("npu:0" if device_type == "npu" else "cpu")
    backend = "ascend_eager" if device_type == "npu" else "torch"
    if (contract["backend"], contract["block_size"], contract["dtype"], contract["loops"]) != (
        backend,
        BLOCK_SIZE,
        "bfloat16",
        4,
    ):
        parser.error("incompatible archived engine contract")
    runtime = {"torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads()}
    if device_type == "npu":
        import torch_npu

        torch.npu.set_device(0)
        runtime.update(
            torch_npu_version=str(torch_npu.__version__), device_name=torch.npu.get_device_name(0)
        )
    differences = {
        key: [contract[key], value] for key, value in runtime.items() if contract[key] != value
    }
    if differences:
        emit("RUNTIME_CHANGED", differences=differences)
    result = dict(
        source=str(args.report),
        source_commit=report["metadata"]["git_commit"],
        diagnostic_commit=git_info("rev-parse", "HEAD"),
        contract=contract,
        runtime=runtime,
        runtime_differences=differences,
        device_mapping=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        archive=archives,
        cases={},
        state="running",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=False)
    clean = True
    try:
        model = OuroForCausalLM.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
        for name, case in cases.items():
            passes = []
            result["cases"][name] = passes
            for i in range(args.repeats):
                result["progress"] = dict(case=name, replay=i + 1)
                offset = i % len(ARMS)
                order = (*ARMS[offset:], *ARMS[:offset], "split")
                run = diagnose_pass(
                    model,
                    backend,
                    case,
                    report["cases"][name],
                    contract["output_tokens"],
                    contract["seed"],
                    order,
                )
                passes.append(run)
                save_report(args.out, result)
                comparison = run["comparison"]
                clean &= comparison["status"] == "PASS"
                pairs = comparison["pairs"]
                emit(
                    "DIAG",
                    case=name,
                    replay=i + 1,
                    status=comparison["status"],
                    archive_status=archives[name]["status"],
                    same_archived_outputs=comparison["joint_archive_outputs_on_same_inputs_1based"],
                    issues=comparison["primary_issues"],
                    control_issues=comparison["diagnostic_issues"],
                    kv_issues={
                        arm: data["kv"]["issues"][:1]
                        for arm, data in run["runs"].items()
                        if not data["kv"]["ok"]
                    },
                    off_miss=pairs["off_miss"]["equal"],
                    off_split=pairs["off_split"]["equal"],
                    split_reuse=pairs["split_reuse"]["equal"],
                    first=pairs["off_reuse"]["first_diff"],
                )
        result["state"] = "complete"
        del result["progress"]
    except (Exception, KeyboardInterrupt) as exc:
        result.update(state="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_report(args.out, result)
    return int(not clean)


if __name__ == "__main__":
    raise SystemExit(main())
