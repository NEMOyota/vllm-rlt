"""Compare native-BF16 prefix cache evidence from one 910B3 and one 910B4 run."""

import argparse
import json
from pathlib import Path

TRAFFIC = ("bf16-burst", "bf16-paced")


def read(path):
    return json.loads(path.read_text())


def comparison(roots):
    reports = [read(root / "review.json") for root in roots]
    chips = []
    for root, report in zip(roots, reports, strict=True):
        device = report.get("hardware", {})
        if (
            device.get("verification") not in ("matched", "observed")
            or device.get("logical_device") != 0
            or device.get("visible_device_count") != 1
        ):
            raise ValueError(f"{root}: missing verified single-device evidence")
        chips.append(device["detected_sku"])
        for stage in ("environment", *TRAFFIC):
            statuses = ("PASS",) if stage == "environment" else ("PASS", "FAIL")
            if report["stages"].get(stage, {}).get("status") not in statuses:
                raise ValueError(f"{root}: {stage} evidence incomplete")
    if set(chips) != {"910B3", "910B4"}:
        raise ValueError(f"expected one 910B3 and one 910B4, observed {chips}")
    a, b = reports
    for key in ("candidate", "harness", "unpacked_policy"):
        if a[key] != b[key]:
            raise ValueError(f"{key} differs between reports")
    for file in ("environment/checkpoint.json", *(f"{t}/traffic.json" for t in TRAFFIC)):
        left, right = (read(root / file) for root in roots)
        if not left or left != right:
            raise ValueError(f"{file}: missing or different inputs")

    lines = [
        "# 910B3 与 910B4 单卡 prefix cache 对照",
        "",
        f"候选 `{a['candidate']}`，测试工具 `{a['harness']}`。两端的模型文件指纹、"
        "burst 和 paced 流量清单分别一致。",
        "",
        "| 型号 | 选卡 | CANN | torch | torch-npu | HF32 | 本次运行 | 全量 PR 1 通过 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for root, report, chip in zip(roots, reports, chips, strict=True):
        env = read(root / "environment/environment.json")
        if (
            env["chip"] != report["hardware"]["runtime_device_name"]
            or env["visible"] != report["assigned_device"]
            or env["commit"] != report["candidate"]
        ):
            raise ValueError(f"{root}: environment stage disagrees with run hardware or candidate")
        lines.append(
            f"| {chip} | {report['assigned_device']} | {env.get('cann')} | {env['torch']} |"
            f" {env['torch_npu']} | {env.get('allow_hf32')} | {report['status']} |"
            f" {report['full_pr1_pass']} |"
        )
    lines += [
        "",
        "每个差值均在同一张卡上计算。取各轮 TTFT P50 的 off 减 main，再取中位数，"
        "正值表示开缓存后更快。服务控制或输出校验失败时，收益栏留空。",
        "",
        "| 型号 | 服务负载 | 状态 | 控制异常 | 输出分歧 | 节省 ms | 更快轮数 / 配对轮数 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for name in TRAFFIC:
        left, right = (report["stages"][name]["evidence"] for report in reports)
        if any(left[k] != right[k] for k in ("rounds", "requests_per_mode")):
            raise ValueError(f"{name}: rounds or request counts differ")
        for report, chip in zip(reports, chips, strict=True):
            stage = report["stages"][name]
            result = stage["evidence"]
            if (
                result["precision"] != "native_bf16"
                or result["linear_fp32_chunk"] is not None
                or result["modes"] != ["off", "main"]
                or result["diagnostic_replay"]
            ):
                raise ValueError(f"{chip} {name}: expected native BF16 off/main service results")
            saved, pairs = "—", "—"
            if stage["status"] == "PASS" and not (
                result["control_issues"] or result["output_diffs"]
            ):
                paired = result["paired_ttft"]["off_to_main"]
                saved = str(paired["median_saved_ms"])
                pairs = f"{paired['faster_pairs']} / {paired['pairs']}"
            lines.append(
                f"| {chip} | {name} | {stage['status']} | {result['control_issues']} | "
                f"{result['output_diffs']} | {saved} | {pairs} |"
            )
    lines += [
        "",
        "各卡 PR_RESULTS.md 保留逐轮绝对 TTFT、P95、吞吐与命中 token，"
        "bf16-matrix/review.json 记录各自的逐 token 反例。",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs=2, metavar="RUN")
    args = parser.parse_args(argv)
    try:
        print(comparison(args.results))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Comparison incomplete: {exc}\n")


if __name__ == "__main__":
    main()
