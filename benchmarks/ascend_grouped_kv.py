"""Screen one public GroupedMatmul K/V call against two native BF16 projections."""

import argparse
import csv
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F

from benchmarks.ascend_matmul_v3 import ROWS
from benchmarks.ascend_prefix_linear_ops import _bits_different
from benchmarks.ascend_prefix_linear_repro import C1, C4, _load_fixture
from docs.recipes.ascend_bf16_matmul import row_runs

PROJECTIONS = ("k_proj", "v_proj")


def operators(weights, grouped_matmul):
    wk, wv = (weights[name] for name in PROJECTIONS)
    # Two transposed views of the same weights used by F.linear. Prepare outside timing.
    transposed = [wk.T, wv.T]

    def native(x):
        return F.linear(x, wk), F.linear(x, wv)

    def grouped(x):
        return grouped_matmul([x, x], transposed, split_item=0, group_type=-1)

    return {"native": native, "grouped": grouped}


def host_outputs(outputs, x, weights):
    result = {}
    for name, y in zip(PROJECTIONS, outputs, strict=True):
        if y.shape != (len(x), weights[name].size(0)) or y.dtype != x.dtype or y.device != x.device:
            raise ValueError(f"unexpected {name} output: {y.shape}, {y.dtype}, {y.device}")
        result[name] = y.detach().to(device="cpu", copy=True).contiguous()
        if not result[name].isfinite().all():
            raise ValueError(f"{name} produced nonfinite output")
    return result


@torch.inference_mode()
def measure(x, weights, ops, *, rows=ROWS, warmups=3, repeats=10):
    sync = torch.npu.synchronize if x.device.type == "npu" else lambda: None
    control = torch.zeros_like(x[: min(rows)])
    control[::2, 7] = 1
    control[1::2, 8] = 1
    for arm, project in ops.items():
        pair = host_outputs(project(control), control, weights)
        for name, y in pair.items():
            for offset in (0, 1):
                selected = y[offset::2]
                if not torch.equal(
                    selected, weights[name][:, 7 + offset].cpu().expand_as(selected)
                ):
                    raise ValueError(f"{arm}/{name}: one-hot control failed (order or transpose)")

    # Every input row equals fixture row 2, including the prefix in the shorter calls.
    x = x[2].clone().expand_as(x).contiguous()
    row = x[0].cpu().double()
    ref = {name: torch.mv(w.cpu().double(), row) for name, w in weights.items()}
    longest = {}
    cases = {}
    for m in sorted(rows, reverse=True):
        values = x[:m]
        first = {arm: host_outputs(op(values), values, weights) for arm, op in ops.items()}
        if not longest:
            longest = first
        times = {arm: [] for arm in ops}
        last = {}
        for i in range(warmups + repeats):
            order = tuple(ops) if i % 2 == 0 else tuple(reversed(ops))
            for arm in order:
                sync()
                start = time.perf_counter()
                output = ops[arm](values)
                sync()
                elapsed = (time.perf_counter() - start) * 1000
                if i >= warmups:
                    times[arm].append(elapsed)
                if i == warmups + repeats - 1:
                    last[arm] = output

        details = {}
        for arm, pair in first.items():
            again = host_outputs(last[arm], values, weights)
            details[arm] = {}
            for name, y in pair.items():
                col = y[:, 18]
                details[arm][name] = dict(
                    within_m=_bits_different(y, y[:1]),
                    vs_largest_m=_bits_different(y, longest[arm][name][:m]),
                    repeat_diff=_bits_different(y, again[name]),
                    vs_native=_bits_different(y, first["native"][name]),
                    ch18_runs=row_runs(col.tolist()),
                    ch18_vs_rounded=int((col != ref[name][18].bfloat16()).sum()),
                    row0_channels_vs_rounded=_bits_different(y[0], ref[name].bfloat16()),
                    max_abs_vs_fp64=float((y.double() - ref[name]).abs().max()),
                )
        p50 = {arm: median(ms) for arm, ms in times.items()}
        cases[m] = dict(details=details, pair_ms=times, pair_p50_ms=p50)
        print("GROUPED_ROW", json.dumps(dict(m=m, pair_p50_ms=p50, **details)), flush=True)

    summary = {}
    for arm in ops:
        summary[arm] = {}
        for name in PROJECTIONS:
            entries = [case["details"][arm][name] for case in cases.values()]
            summary[arm][name] = dict(
                invariant=all(not (e["within_m"] or e["vs_largest_m"]) for e in entries),
                repeat_stable=all(not e["repeat_diff"] for e in entries),
                ch18_matches_rounded=all(not e["ch18_vs_rounded"] for e in entries),
            )
    return dict(
        fp64_ch18={name: float(v[18]) for name, v in ref.items()},
        rounded_bf16_ch18={name: float(v[18].bfloat16()) for name, v in ref.items()},
        summary=summary,
        cases=cases,
    )


def original_cases(cases, weights, ops):
    """Compare the original shared prefix across full/chunk calls or cloned C4 slots."""
    results = {}
    for name in (C1, C4):
        if name not in cases:
            continue
        x, info = cases[name]["x"], cases[name]["info"]
        prefix = info["prefix_rows"]
        if info["controls"]:
            raise ValueError(f"{name}: inconsistent operands: {info['controls']}")
        full = {arm: host_outputs(op(x), x, weights) for arm, op in ops.items()}
        result = {}
        for arm, op in ops.items():
            if name == C1:
                short = host_outputs(op(x[:prefix]), x[:prefix], weights)
            result[arm] = {}
            for projection, y in full[arm].items():
                if name == C1:
                    diffs = [_bits_different(y[:prefix], short[projection])]
                else:
                    slots = y.reshape(4, info["slot_rows"], -1)[:, :prefix]
                    diffs = [_bits_different(slot, slots[info["focus_slot"]]) for slot in slots]
                result[arm][projection] = dict(
                    prefix_diffs=diffs,
                    full_vs_native=_bits_different(y, full["native"][projection]),
                )
        results[name] = result
        print("GROUPED_FIXTURE", name, json.dumps(result), flush=True)
    return results


def profile_calls(directory, x, ops):
    import torch_npu.profiler as p

    data = {}
    for arm, project in ops.items():
        folder = directory / arm
        folder.mkdir(parents=True, exist_ok=False)
        torch.npu.synchronize()
        with p.profile(
            activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
            experimental_config=p._ExperimentalConfig(profiler_level=p.ProfilerLevel.Level1),
            record_shapes=True,
            on_trace_ready=p.tensorboard_trace_handler(str(folder)),
        ):
            for m in ROWS:
                with torch.profiler.record_function(f"RLT/{arm}/M{m}"):
                    project(x[:m])
                    torch.npu.synchronize()
        counts = Counter()
        files = sorted(folder.rglob("kernel_details.csv"))
        for path in files:
            with path.open(encoding="utf-8-sig", newline="") as f:
                for record in csv.DictReader(f):
                    counts[(record.get("Type", ""), record.get("Name", ""))] += 1
        data[arm] = dict(
            files=[str(f) for f in files],
            device_rows_available=any(t or n for t, n in counts),
            kernels=[dict(type=t, name=n, count=k) for (t, n), k in counts.items()],
        )
        print("GROUPED_DEVICE", arm, json.dumps(data[arm]), flush=True)
    return data


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path, default=Path("results/linear-inputs-20261119T092602.pt")
    )
    parser.add_argument("--out", type=Path, help="new JSON path (default: timestamped in results/)")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--profile", action="store_true", help="capture device kernels after timing"
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups nonnegative")
    import torch_npu
    from torch_npu.utils.collect_env import get_cann_version

    contract, weights, cases = _load_fixture(args.fixture)
    x = cases[C4]["x"]
    if (
        contract["device"] != "npu"
        or x.shape != (576, 2048)
        or x.dtype != torch.bfloat16
        or any(weights[n].shape != (2048, 2048) or weights[n].dtype != x.dtype for n in PROJECTIONS)
    ):
        parser.error("expected the original 910B4 c4_h4 BF16 fixture and K/V weights")
    ops = operators(weights, torch_npu.npu_grouped_matmul)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = args.out or Path(f"results/grouped-kv-{now}.json")
    trace = path.with_name(path.name + ".profile")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or (args.profile and trace.exists()):
        parser.error("choose a new output path")
    print("RESULTS", path.resolve(), flush=True)
    report = measure(x, weights, ops, warmups=args.warmups, repeats=args.repeats)
    report["original_cases"] = original_cases(cases, weights, ops)
    try:
        cann = get_cann_version() if os.environ.get("ASCEND_HOME_PATH") else None
    except (OSError, RuntimeError):
        cann = None
    report.update(
        fixture=str(args.fixture),
        device=torch.npu.get_device_name(0),
        torch=str(torch.__version__),
        torch_npu=torch_npu.__version__,
        cann=cann,
        ignore_infer_error=os.environ.get("IGNORE_INFER_ERROR"),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    packed_invariant = all(
        not any(p["prefix_diffs"])
        for result in report["original_cases"].values()
        for p in result["grouped"].values()
    )
    report["review"] = dict(
        **report["summary"],
        original_grouped_prefix_invariant=packed_invariant,
        rounded_bf16_ch18=report["rounded_bf16_ch18"],
        pair_p50_ms={m: c["pair_p50_ms"] for m, c in report["cases"].items()},
    )
    with path.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print("GROUPED_REVIEW", json.dumps(report["review"]), flush=True)
    if args.profile:
        repeated = x[2].clone().expand_as(x).contiguous()
        evidence = profile_calls(trace, repeated, ops)
        with (trace / "device-kernels.json").open("x") as f:
            json.dump(evidence, f, indent=2)
    return int(
        not packed_invariant
        or any(
            not (s["invariant"] and s["repeat_stable"])
            for s in report["summary"]["grouped"].values()
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
