"""Same-batch Ouro prefix hits through production main/exact/similar Ascend attention."""

import argparse
from pathlib import Path

import torch

from benchmarks import ascend_main_prefix_combo as combo
from benchmarks import ascend_main_prefix_matrix as matrix
from benchmarks.ascend_main_prefix_linear import add_precision_arg, padded_fp32_linears
from benchmarks.ascend_main_prefix_results import Report, ids_digest, savings
from benchmarks.ascend_prefix_combo_checks import require
from benchmarks.ascend_prefix_pr1_diagnostics import capture_context, checked_profile_files
from vllm_rlt.models import OuroForCausalLM

# (suffix lengths, number of hot requests, exact groups, additional padded pairs)
CASES = {
    "near": ((23, 20), 2, 0, 1),
    "exact": ((23, 23), 2, 1, 0),
    "mixed": ((23, 20, 23, 23), 2, 1, 1),
    "outlier": ((23, 4), 2, 0, 0),
    "cold_grid": ((23, 23), 0, 0, 0),
    "single": ((23,), 1, 0, 0),
}
MODES = {"main": None, "exact": 0, "similar": 4}


def workload(name, vocab):
    widths, hot, _, _ = CASES[name]
    case = combo.Case(48, max(widths), len(widths), hot)
    donor, miss, targets, fillers = combo.workload(case, vocab)
    targets = {
        rid: tokens[: 48 + width]
        for (rid, tokens), width in zip(targets.items(), widths, strict=True)
    }
    return case, (donor, miss, targets, fillers)


def run_trial(model, name, mode, inputs, case, linear_work, capture=None):
    out = matrix.trial(
        model,
        case,
        "main",
        "reuse",
        inputs,
        linear_work=linear_work,
        prefix_prefill_slack=MODES[mode],
        capture=capture,
    )
    widths, hot, exact, padded = CASES[name]
    scale = out["layer_traversals"]
    group = out["prefix_groups"]
    pairs = (exact if mode != "main" else 0) + (padded if mode == "similar" else 0)
    require(
        group.get("groups", 0) == group.get("saved_sdpa_calls", 0) == pairs * scale,
        f"{name}/{mode} did not execute the expected pairs: {group}",
    )
    padded_pairs = padded if mode == "similar" else 0
    require(group.get("padded_groups", 0) == padded_pairs * scale, f"wrong padded groups: {group}")
    require(
        group.get("extra_query_rows", 0) == 3 * padded_pairs * scale
        and group.get("extra_kv_rows", 0) == 3 * padded_pairs * scale
        and group.get("extra_score_cells", 0) == 273 * padded_pairs * scale,
        f"wrong padding for 23/20 rows at prefix 48: {group}",
    )
    require(
        out["prefill_rows"] == sum(widths) + 48 * (len(widths) - hot),
        "scheduler counted temporary rows as true prefill",
    )
    return out


def evaluate(model, args, report):
    failed = []
    with padded_fp32_linears(model, args.linear_fp32_chunk) as linear_work:
        for name in args.case or (["near"] if args.profile else CASES):
            case, inputs = workload(name, model.config.vocab_size)
            report.record(
                "INPUTS",
                case=name,
                donor=inputs[0],
                targets=inputs[2],
                digests={k: ids_digest(v) for k, v in inputs[2].items()},
            )
            first, times, differences = {}, {mode: [] for mode in MODES}, []

            def check(mode, out, round_id):
                references = [(mode + "_vs_first", first.get(mode))]
                if mode != "main":
                    references.append(("main_vs_" + mode, first.get("main")))
                if mode == "similar":
                    references.append(("exact_vs_similar", first.get("exact")))
                for label, base in references:
                    if base is not None:
                        differences.extend(
                            dict(round=round_id, **d) for d in combo.token_diffs(base, out, label)
                        )
                first.setdefault(mode, out)

            repeats = 1 if args.profile else args.repeats + 1
            for i in range(repeats):
                modes = list(MODES)
                order = modes[i % 3 :] + modes[: i % 3]
                for mode in order:
                    report.location = dict(case=name, mode=mode, round=i)
                    row = run_trial(model, name, mode, inputs, case, linear_work)
                    report.record("TRIAL", **report.location, warmup=i == 0, **row)
                    check(mode, row, i)
                    if i:
                        times[mode].append(matrix.metrics([row])["ttft_ms"]["p50"])
            if args.profile:
                for full in (False, True):
                    for mode in MODES:
                        folder = report.folder / name / (("full-" if full else "ops-") + mode)
                        folder.mkdir(parents=True)
                        report.location = dict(case=name, mode=mode, profile=folder.name)
                        row = run_trial(
                            model,
                            name,
                            mode,
                            inputs,
                            case,
                            linear_work,
                            capture_context(folder, full),
                        )
                        check(mode, row, folder.name)
                        artifacts = checked_profile_files(folder, full)
                        actual = artifacts["cpu_operators"].get(
                            "aten::scaled_dot_product_attention"
                        )
                        base_calls = 1 if name == "cold_grid" else case.count
                        expected = base_calls * row["layer_traversals"] - row["prefix_groups"].get(
                            "saved_sdpa_calls", 0
                        )
                        if actual is not None:
                            require(
                                actual == expected,
                                f"trace has {actual} SDPA calls, expected {expected}",
                            )
                        report.record(
                            "CAPTURE",
                            **report.location,
                            artifacts=artifacts,
                            expected_sdpa=expected,
                            observed_sdpa=actual,
                            prefix_groups=row["prefix_groups"],
                        )
            paired = {}
            if not differences and not args.profile:
                for before, after in (("main", "exact"), ("main", "similar"), ("exact", "similar")):
                    paired[f"{before}_to_{after}"] = savings(times[before], times[after])
            report.record("CASE", echo=True, case=name, differences=differences, paired_ttft=paired)
            if differences:
                failed.append(name)
    return report.review("FAIL" if failed else "PASS", failed=failed, profile=args.profile)


@torch.inference_mode()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--case", choices=CASES, action="append")
    p.add_argument(
        "--profile",
        action="store_true",
        help="Warm up and capture just one target prefill step per mode",
    )
    add_precision_arg(p)
    args = p.parse_args(argv)
    if not args.model.is_dir() or args.repeats < 1:
        p.error("provide a local model and at least one repetition")
    report = Report("PREFIX_GROUPS", args.out_dir, linear_fp32_chunk=args.linear_fp32_chunk)

    def run(args, device, report):
        model = OuroForCausalLM.from_pretrained(
            args.model, device=device, dtype=torch.bfloat16
        ).eval()
        try:
            return evaluate(model, args, report)
        except AssertionError as exc:
            return report.review("FAIL", stopped_at=report.location, detail=str(exc))

    return report.execute(args, run)


if __name__ == "__main__":
    raise SystemExit(main())
