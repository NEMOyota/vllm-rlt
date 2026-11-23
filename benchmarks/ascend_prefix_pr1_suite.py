"""Run the pre-PR checks against a committed candidate with a pinned external harness."""

import argparse
import csv
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "tests/test_ascend_prefix_cache_contract.py"
STOCK = "tests/test_ascend_serving.py"
HTTP_CASE = "::test_checkpoint_http_completions_stream_and_disconnect[ascend-prefix]"
DIAGNOSTICS = "benchmarks.ascend_prefix_pr1_diagnostics"
MATRIX = "benchmarks.ascend_main_prefix_matrix"
SERVICE = "benchmarks.ascend_main_prefix_service"
COMBO = "benchmarks.ascend_main_prefix_combo"
PROBE = "benchmarks/ascend_prefix_pr1_hardware.py"
CPU = ("pre-commit", "config")
NPU = (
    "environment",
    "npu-regression",
    "npu-contract",
    "stock-http",
    "bf16-combo",
    "bf16-matrix",
    "fp32-combo",
    "fp32-matrix",
    "bf16-burst",
    "bf16-paced",
    "bf16-benefit",
    "drift",
    "profile",
)
GROUPING = ("grouped-matrix", "grouped-burst", "grouped-paced", "observe-burst", "observe-paced")
HARNESS = (
    "benchmarks/ascend_main_prefix_combo.py",
    "benchmarks/ascend_main_prefix_linear.py",
    "benchmarks/ascend_main_prefix_grouping.py",
    "benchmarks/ascend_prefix_combo_checks.py",
    "benchmarks/ascend_main_prefix_matrix.py",
    "benchmarks/ascend_main_prefix_results.py",
    "benchmarks/ascend_main_prefix_service.py",
    "benchmarks/ascend_main_prefix_service_worker.py",
    "benchmarks/ascend_prefix_pr1_diagnostics.py",
    "benchmarks/ascend_prefix_opportunities.py",
    CONTRACT,
)


@dataclass
class Stage:
    name: str
    command: list
    artifact: Path | None = None


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def reference_manifest(root, name):
    current = root / ("bf16-" + name) / "traffic.json"
    historical = root / name / "traffic.json"
    return historical if historical.is_file() and not current.is_file() else current


def stage_commands(args, out):
    py = sys.executable
    model = ["--model", str(args.model)] if args.model else ["--model", "<MODEL>"]
    stages = [
        Stage(
            "pre-commit",
            ["pre-commit", "run", "--all-files", "--verbose", "--show-diff-on-failure"],
        )
    ]

    def pytest(name, options):
        xml = out / (name + ".xml")
        stages.append(
            Stage(name, [py, "-m", "pytest", "-q", "-ra", f"--junitxml={xml}", *options], xml)
        )

    pytest("config", ["-m", "not npu", CONTRACT])
    # Run the candidate's own NPU suite before installing any test-only files.
    pytest(
        "npu-regression",
        [
            "--run-npu",
            "-m",
            "npu",
            "--ignore=" + CONTRACT,
            "--deselect=" + STOCK + HTTP_CASE,
            "tests",
        ],
    )
    pytest("npu-contract", ["--run-npu", "-m", "npu", CONTRACT])
    pytest("stock-http", ["--run-npu", "tests/test_pr1_stock_http_replay.py" + HTTP_CASE])

    def benchmark(name, module, options):
        folder = out / name
        if module == COMBO:
            artifact = folder.with_suffix(".jsonl")
            dest = ["--out", str(artifact)]
        elif module == DIAGNOSTICS:
            artifact = folder / ("checkpoint.json" if name == "environment" else "review.json")
            dest = ["--out", str(folder)]
        else:
            artifact = folder / "review.json"
            dest = ["--out-dir", str(folder)]
        stages.append(Stage(name, [py, "-m", module, *options, *model, *dest], artifact))

    benchmark("environment", DIAGNOSTICS, ["env"])
    for precision in ("bf16", "fp32"):
        chunk = ["--linear-fp32-chunk", "256"] if precision == "fp32" else []
        benchmark(precision + "-combo", COMBO, ["--repeats", "2", *chunk])
        benchmark(
            precision + "-matrix", MATRIX, [precision, "--main-only", "--repeats", "3", *chunk]
        )
    for track, name in (
        ("bf16", "burst"),
        ("bf16", "paced"),
        ("grouped", "burst"),
        ("grouped", "paced"),
        ("observe", "burst"),
        ("observe", "paced"),
    ):
        options = [
            "--requests",
            "32",
            "--concurrency",
            "4",
            "--seed",
            "168",
            "--rounds",
            "1" if track == "observe" else "2",
            "--rps",
            "3" if name == "paced" else "0",
        ]
        if track != "grouped":
            options.append("--main-only")
        if track == "observe":
            options.append("--observe-prefill")
        if args.reference_traffic:
            options.extend(
                ["--reference-traffic", str(reference_manifest(args.reference_traffic, name))]
            )
        benchmark(track + "-" + name, SERVICE, options)
    benchmark(
        "bf16-benefit",
        MATRIX,
        ["benefit", "--main-only", "--all-cases", "--warmups", "1", "--repeats", "6"],
    )
    benchmark("grouped-matrix", MATRIX, ["bf16", "--repeats", "3"])
    benchmark("drift", DIAGNOSTICS, ["drift", "--matrix", str(out / "bf16-matrix" / "review.json")])
    benchmark("profile", DIAGNOSTICS, ["profile"])
    return {stage.name: stage for stage in stages}


def read_evidence(stage):
    path = stage.artifact
    if path is None:
        return {}
    if path.suffix == ".xml":
        cases = ElementTree.parse(path).findall(".//testcase")
        skipped = [c.get("name") for c in cases if c.find("skipped") is not None]
        failures = [
            c.get("name")
            for c in cases
            if c.find("failure") is not None or c.find("error") is not None
        ]
        return {"tested": len(cases), "skipped": skipped, "failed_tests": failures}
    if path.suffix == ".jsonl":
        review = None
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["kind"] == "COMBO_REVIEW":
                    review = row
        if review is None:
            raise ValueError("COMBO_REVIEW missing")
        return review
    return json.loads(path.read_text())


def run_stage(stage, cwd, env, out):
    log = out / (stage.name + ".log")
    print(f"[{stage.name}] {shlex.join(stage.command)}\n  log: {log}", flush=True)
    start = time.monotonic()
    result = {"status": "INCOMPLETE", "command": stage.command, "log": log.name}
    if stage.artifact:
        result["artifact"] = str(stage.artifact.relative_to(out))
    try:
        with log.open("x") as stream:
            proc = subprocess.run(
                stage.command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        result["exit_code"] = proc.returncode
        if proc.returncode < 0:
            raise KeyboardInterrupt
        evidence = read_evidence(stage)
        if stage.name != "environment":
            result["evidence"] = evidence
        if evidence.get("status") == "INCOMPLETE" or proc.returncode > 1:
            result["detail"] = evidence.get("detail", "command incomplete; see log")
        elif proc.returncode or evidence.get("status") == "FAIL" or evidence.get("failed_tests"):
            result["status"] = "FAIL"
        elif evidence.get("tested") == 0 or evidence.get("skipped"):
            result["detail"] = "some selected tests did not execute; inspect JUnit"
        else:
            result["status"] = "PASS"
    except (OSError, ValueError, ElementTree.ParseError) as exc:
        result["detail"] = str(exc)
    result["seconds"] = round(time.monotonic() - start, 2)
    print(f"[{stage.name}] {result['status']} ({result['seconds']} s)", flush=True)
    return result


def install_checks(checkout, tools_sha):
    # Use only benchmark/test files from the tools commit. Preserve all product
    # modules and run pre-commit + the upstream NPU suite on the clean candidate.
    for source in (*HARNESS, STOCK):
        target = "tests/test_pr1_stock_http_replay.py" if source == STOCK else source
        data = subprocess.check_output(["git", "show", f"{tools_sha}:{source}"], cwd=ROOT)
        (checkout / target).write_bytes(data)


def check_hardware(checkout, env, out, tools_sha, model, expected):
    selector = env.get("ASCEND_RT_VISIBLE_DEVICES")
    if not selector or not model or not model.is_dir():
        return {
            "verification": "unavailable",
            "detail": "requires local MODEL and assigned ASCEND_RT_VISIBLE_DEVICES",
        }
    if expected and not re.fullmatch("[0-9]+", selector):
        return {
            "verification": "unavailable",
            "detail": "--chip requires one numeric ASCEND_RT_VISIBLE_DEVICES selector",
        }
    # Run the pinned probe outside the measured workers. Exiting this process
    # releases its NPU context before any regression or timed workload starts.
    code = subprocess.check_output(["git", "show", f"{tools_sha}:{PROBE}"], cwd=ROOT)
    path = out / "hardware.json"
    command = [sys.executable, "-", str(path)]
    if expected:
        command.append(expected)
    with (out / "hardware.log").open("x") as log:
        result = subprocess.run(
            command, input=code, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode < 0:
        raise KeyboardInterrupt
    if result.returncode:
        return {
            "verification": "unavailable",
            "detail": f"NPU probe exited {result.returncode}; see hardware.log",
        }
    return json.loads(path.read_text())


def snapshot(out, report):
    # Write after each stage. An interrupted run keeps the remaining NOT_RUN rows.
    statuses = {r["status"] for r in report["stages"].values()}
    if statuses == {"PASS"} and report.get("hardware", {}).get("verification") != "unverified":
        status = "PASS"
    elif "FAIL" in statuses:
        status = "FAIL"
    else:
        status = "INCOMPLETE"
    report.update(status=status, full_pr1_pass=report["full_pr1_scope"] and status == "PASS")
    pending = out / ".review.tmp"
    pending.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    pending.replace(out / "review.json")
    return status


def markdown_report(out, report):
    title = "Ascend prefix cache 验证结果"
    lines = [
        f"# {title}",
        "",
        f"- 总状态 `{report['status']}`",
        f"- PR 1 全量自动检查通过 `{report['full_pr1_pass']}`",
        f"- 候选代码 `{report['candidate']}`",
        f"- 测试工具 `{report['harness']}`",
        "",
    ]
    if evidence := report.get("hardware"):
        lines.append(
            f"目标型号 `{report['expected_chip'] or '自动识别'}`，"
            f"运行时设备名 `{evidence.get('runtime_device_name', '未取得')}`，"
            f"型号核对 `{evidence['verification']}`。"
        )
        if "device_memory_bytes" in evidence:
            gib = 1024**3
            total = evidence["device_memory_bytes"]
            hbm = f"{total / gib:.2f} GiB" if total is not None else "未取得"
            lines.append(
                f"ASCEND_RT_VISIBLE_DEVICES=`{report['assigned_device']}`，逻辑卡 "
                f"`npu:{evidence['logical_device']}`，可见卡数 {evidence['visible_device_count']}，"
                f"单卡 HBM {hbm}，"
                f"操作系统可见物理内存 {evidence['os_physical_memory_bytes'] / gib:.2f} GiB。"
                " [设备记录](hardware.json)"
            )
        if detail := evidence.get("detail"):
            lines.append(detail)
        if (out / "hardware.log").is_file():
            lines.append("[设备探测日志](hardware.log)")
        lines.append("")
    environment = out / "environment" / "environment.json"
    if environment.exists():
        data = json.loads(environment.read_text())
        keys = ("chip", "visible", "cann", "torch", "torch_npu", "allow_hf32")
        lines.extend([" · ".join(f"{k} `{data.get(k)}`" for k in keys), ""])
    lines += ["| 检查 | 状态 | 日志或数据 |", "|---|---|---|"]
    for name, row in report["stages"].items():
        sources = [f"[日志]({row['log']})"] if row.get("log") else []
        if row.get("artifact") and (out / row["artifact"]).is_file():
            sources.append(f"[数据]({row['artifact']})")
        lines.append(f"| {name} | {row['status']} | {' · '.join(sources)} |")

    for name in ("bf16-matrix", "fp32-matrix"):
        evidence = report["stages"].get(name, {}).get("evidence", {})
        if "cases" in evidence:
            failed = evidence.get("failed", [])
            lines.extend(
                [
                    "",
                    f"**{name}**　{len(evidence['cases'])} 个场景，"
                    f"{len(failed)} 个失败。{'、'.join(failed)}",
                ]
            )
    fields = (
        "round",
        "mode",
        "completed",
        "http_errors",
        "hit_tokens",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "arrival_ttft_p95_ms",
        "output_tok_s",
    )
    for name in ("bf16-burst", "bf16-paced"):
        csv_path = out / name / "service.csv"
        if not csv_path.is_file():
            continue
        stage = report["stages"][name]
        lines.extend(
            [
                "",
                f"## {name}",
                "",
                "每行是一轮完整 HTTP 回放。延迟单位 ms。分组结果保留在 service.csv。",
                "",
                f"本阶段 {stage['status']}，"
                f"输出分歧记录 {stage.get('evidence', {}).get('output_diffs', '未知')}。",
                "",
                "| 轮 | 模式 | 完成 | HTTP错误 | 命中token | TTFT P50 | TTFT P95 | "
                "含客户端排队 P95 | 输出tok/s |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        with csv_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["cohort"] == "all":
                    lines.append("| " + " | ".join(row[key] for key in fields) + " |")
        lines.extend(
            ["", f"[按冷热和长度查看]({name}/service.csv) · [逐请求审计]({name}/raw.jsonl)"]
        )
    lines += [
        "",
        "提交前请核对输出分歧的首个 token 和证据、未执行项、"
        "未开 profiler 的同工作量结果、代表性 trace 及贡献者本人回答。",
        "",
    ]
    (out / "PR_RESULTS.md").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scope", nargs="?", default="all", choices=("all", "cpu", "npu", "pr2"))
    p.add_argument(
        "--only",
        action="append",
        choices=(*CPU, *NPU, *GROUPING),
        help="Run a named stage; repeat to select multiple stages",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=os.environ.get("MODEL") or os.environ.get("VLLM_RLT_OURO_CHECKPOINT"),
    )
    p.add_argument(
        "--repo", type=Path, default=ROOT, help="Local Git repo containing the candidate commit"
    )
    p.add_argument(
        "--candidate", default="HEAD", help="Committed candidate ref in --repo (default HEAD)"
    )
    p.add_argument(
        "--chip",
        choices=("910B3", "910B4"),
        help="Verify the runtime SKU of the assigned single card for this run",
    )
    p.add_argument(
        "--reference-traffic",
        type=Path,
        help="Prior results root with {bf16-burst,bf16-paced} or legacy {burst,paced} manifests",
    )
    p.add_argument(
        "--unpacked",
        choices=("support", "reject"),
        default="support",
        help="Expected explicit --no-packed-prefill behavior of the candidate",
    )
    p.add_argument("--out", type=Path, help="New results directory; existing paths are preserved")
    p.add_argument(
        "--list",
        action="store_true",
        help="Print selected commands without executing or creating output",
    )
    args = p.parse_args(argv)
    chosen = args.only or {"cpu": CPU, "npu": NPU, "all": CPU + NPU, "pr2": GROUPING}[args.scope]
    selected = set(chosen)
    # Checks of the untouched candidate precede installation of the harness.
    order = ["pre-commit", "npu-regression", "environment", *CPU[1:], *NPU[2:], *GROUPING]
    names = [name for name in order if name in selected]
    if args.model:
        args.model = args.model.expanduser().resolve()
    if args.reference_traffic:
        args.reference_traffic = args.reference_traffic.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    selector = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    suffix = f"-{args.chip}" if args.chip else ""
    if args.chip and re.fullmatch("[0-9]+", selector):
        suffix += f"-card{selector}"
    out = (args.out or (Path.cwd() / "results" / f"ascend-prefix-pr1{suffix}-{stamp}")).resolve()
    stages = stage_commands(args, out)
    if args.list:
        for name in names:
            print(name, shlex.join(stages[name].command))
        return 0
    repo = args.repo.resolve()
    candidate = git(repo, "rev-parse", "--verify", "--end-of-options", args.candidate + "^{commit}")
    tools_sha = git(ROOT, "rev-parse", "HEAD")
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, RLT_PR1_UNPACKED=args.unpacked)
    # A shell's local hook skip or pytest filter must not narrow a full validation.
    env.pop("SKIP", None)
    env.pop("PYTEST_ADDOPTS", None)
    env["PATH"] = os.pathsep.join((str(Path(sys.executable).parent), env.get("PATH", "")))
    if args.model:
        env["VLLM_RLT_OURO_CHECKPOINT"] = str(args.model)
    env["VLLM_RLT_ASCEND_RESULTS"] = str(out / "upstream-http")
    report = dict(
        candidate=candidate,
        harness=tools_sha,
        python=sys.version,
        platform=platform.platform(),
        model=str(args.model) if args.model else None,
        assigned_device=env.get("ASCEND_RT_VISIBLE_DEVICES"),
        expected_chip=args.chip,
        unpacked_policy=args.unpacked,
        full_pr1_scope=selected == set(CPU + NPU),
        stages={n: {"status": "NOT_RUN"} for n in names},
    )
    snapshot(out, report)
    print("RESULTS", out, flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="rlt-pr1-") as tmp:
            checkout = Path(tmp) / "candidate"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--no-hardlinks",
                    "--no-checkout",
                    str(repo),
                    str(checkout),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "checkout", "--quiet", "--detach", candidate], cwd=checkout, check=True
            )
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(checkout), env.get("PYTHONPATH")))
            )
            hardware = {}
            if selected.difference(CPU):
                hardware = check_hardware(checkout, env, out, tools_sha, args.model, args.chip)
                report["hardware"] = hardware
                detail = hardware.get("detail") or hardware["runtime_device_name"]
                print(f"[hardware] {hardware['verification']}  {detail}", flush=True)
                snapshot(out, report)
            # Keep hardware setup paths, but resolve all runtime modules from the candidate.
            installed = False
            for name in names:
                if name not in CPU and hardware["verification"] in ("unavailable", "mismatch"):
                    report["stages"][name] = {
                        "status": "INCOMPLETE",
                        "detail": hardware["detail"],
                    }
                else:
                    # Test-only files are installed after checking the unmodified candidate.
                    if not installed and name not in ("pre-commit", "npu-regression"):
                        install_checks(checkout, tools_sha)
                        installed = True
                    result = run_stage(stages[name], checkout, env, out)
                    report["stages"][name] = result
                snapshot(out, report)
    except (OSError, subprocess.CalledProcessError) as exc:
        report["setup_error"] = str(exc)
        print(f"Setup incomplete: {exc}", file=sys.stderr)
    status = snapshot(out, report)
    markdown_report(out, report)
    print(f"{status}  {out / 'PR_RESULTS.md'}", flush=True)
    if status == "INCOMPLETE" or any(
        r["status"] in ("INCOMPLETE", "NOT_RUN") for r in report["stages"].values()
    ):
        return 2
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
