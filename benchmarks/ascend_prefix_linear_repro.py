"""Isolate first-layer Ouro BF16 projections on the original C1 and C4 operands."""

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from benchmarks.ascend_prefix_linear_ops import measure_linear

C1 = "total512_p384"
C4 = "c4_h4"


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


@contextmanager
def _new_file(path, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as handle:
        try:
            yield handle
            handle.flush()
        except BaseException:
            path.unlink(missing_ok=True)
            raise


def _model_operands(parser, args):
    from benchmarks.ascend_prefix_fix_trial import load_replay_model
    from benchmarks.ascend_prefix_linear_inputs import prepare_linear_inputs
    from benchmarks.ascend_prefix_scenarios import suites

    reports = {
        name: json.loads(path.read_text())
        for name, path in ((C1, args.boundary), (C4, args.writer))
        if path
    }
    contract = next(iter(reports.values()))["contract"]
    if any(report["contract"] != contract for report in reports.values()):
        parser.error("the C1 boundary and C4 writer reports must share the execution contract")
    specs = {c.name: c for c in suites()["full"]}
    model = load_replay_model(parser, args, contract)
    operands = {}
    for name, report in reports.items():
        x, weights, info = prepare_linear_inputs(model, specs[name], contract)
        if name == C1:
            summary = report["summary"]
            if summary["case"] != name or not summary["comparable"]:
                parser.error("expected a comparable total512_p384 boundary report")
            references = [report["modes"][m]["fingerprint"] for m in ("reuse", "aligned")]
        else:
            clones = report["runs"]["clones"]
            if clones["summary"]["case"] != name or not clones["summary"]["comparable"]:
                parser.error("expected a comparable c4_h4 clones writer report")
            rows = clones["run"]["requests"]
            if len(rows) != 4 or any(row["prompt_sha256"] != info["prompt_sha256"] for row in rows):
                parser.error("the archived C4 clone prompts differ from the rebuilt target-2")
            references = [clones["run"]["prefill_fingerprints"][f"target-{i}"] for i in range(4)]
        if any(ref["prefix_tokens"] != info["prefix_rows"] for ref in references):
            parser.error("archived KV does not cover the comparison range")
        # V is stored directly. Archived K is post-RoPE and cannot verify a raw K projection.
        info["engine_v_sha256"] = [ref["sha256"][0][0]["v"] for ref in references]
        operands[name] = dict(x=x, info=info)
    return contract, weights, operands


def _load_fixture(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["schema"] != 1:
        raise ValueError("unsupported linear operand file")
    contract, weights, operands = data["contract"], data["weights"], data["cases"]
    if any(_digest(t) != data["weight_sha256"][k] for k, t in weights.items()) or any(
        _digest(c["x"]) != c["info"]["input_sha256"] for c in operands.values()
    ):
        raise ValueError("saved operands failed their content checks")
    device = contract["device"]
    if device == "npu":
        import torch_npu  # noqa: F401 -- registers the device

        torch.npu.set_device(0)
        device = "npu:0"
    elif device != "cpu":
        raise ValueError("expected a CPU or Ascend operand file")
    return (
        contract,
        {k: t.to(device) for k, t in weights.items()},
        {name: dict(x=c["x"].to(device), info=c["info"]) for name, c in operands.items()},
    )


@torch.inference_mode()
def _trace_native(directory, operands, weights, device):
    """Record one additional native call per shape, outside all timed measurements."""
    if device == "npu":
        import torch_npu.profiler as profiler

        activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
        pattern = "trace_view.json"
        torch.npu.synchronize()
    else:
        profiler = torch.profiler
        activities = [profiler.ProfilerActivity.CPU]
        pattern = "*.pt.trace.json"
    directory.mkdir(parents=True, exist_ok=False)
    with profiler.profile(
        activities=activities,
        record_shapes=True,
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
    ):
        for name, case in operands.items():
            x = case["x"]
            shapes = {"full": x}
            if name == C1:
                shapes["prefix"] = x[: case["info"]["prefix_rows"]]
            for projection, w in weights.items():
                for shape, values in shapes.items():
                    with torch.profiler.record_function(f"RLT/{name}/{projection}/{shape}"):
                        torch.nn.functional.linear(values, w)
        if device == "npu":
            torch.npu.synchronize()
    traces = [str(p) for p in sorted(directory.rglob(pattern)) if p.stat().st_size]
    if not traces:
        raise RuntimeError(f"no {pattern} exported in {directory}; the measurement JSON is saved")
    print("LINEAR_TRACE", json.dumps(traces), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="local Ouro checkpoint")
    parser.add_argument("--boundary", type=Path, help="C1 total512_p384 BOUNDARY JSON")
    parser.add_argument("--writer", type=Path, help="C4_WRITE JSON containing the clones run")
    parser.add_argument(
        "--save-fixture", type=Path, help="export actual X and first-layer weights to a new .pt"
    )
    parser.add_argument(
        "--fixture", type=Path, help="replay an exported .pt without the model or engine"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--trace-dir", type=Path, help="new directory for a separate native F.linear trace"
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        choices=("linear", "mm", "tile64", "npu_linear"),
        default=["linear", "mm", "tile64"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    if args.fixture:
        if any((args.model, args.boundary, args.writer, args.save_fixture)):
            parser.error("--fixture takes the place of model and archive inputs")
    elif not args.model or not (args.boundary or args.writer):
        parser.error("supply --model with --boundary and/or --writer, or use --fixture")
    if args.repeats < 2 or args.warmups < 0:
        parser.error("use at least two repeats and nonnegative warmups")
    outputs = [p.resolve() for p in (args.out, args.save_fixture, args.trace_dir) if p]
    if len(outputs) != len(set(outputs)) or any(p.exists() for p in outputs):
        parser.error("outputs must be distinct new paths")
    contract, weights, operands = (
        _load_fixture(args.fixture) if args.fixture else _model_operands(parser, args)
    )
    hashes = {k: _digest(w) for k, w in weights.items()}
    if args.save_fixture:
        with _new_file(args.save_fixture, "xb") as handle:
            torch.save(
                dict(
                    schema=1,
                    contract=contract,
                    weight_sha256=hashes,
                    weights={k: w.detach().cpu() for k, w in weights.items()},
                    cases={
                        n: dict(x=c["x"].detach().cpu(), info=c["info"])
                        for n, c in operands.items()
                    },
                ),
                handle,
            )

    runtime = dict(
        torch=str(torch.__version__), threads=torch.get_num_threads(), device=contract["device"]
    )
    if contract["device"] == "npu":
        import torch_npu

        runtime.update(torch_npu=str(torch_npu.__version__), name=torch.npu.get_device_name(0))
    report = dict(contract=contract, runtime=runtime, weight_sha256=hashes, cases={})
    comparable = True
    for name, case in operands.items():
        info = case["info"]
        results = {}
        report["cases"][name] = dict(inputs=info, ops=results)
        for op in dict.fromkeys(["linear", *args.ops]):
            projections = {}
            results[op] = projections
            for projection, weight in weights.items():
                projections[projection] = measure_linear(
                    case["x"],
                    weight,
                    case=name,
                    prefix_rows=info["prefix_rows"],
                    slot_rows=info["slot_rows"],
                    op=op,
                    repeats=args.repeats,
                    warmups=args.warmups,
                )
            controls = info["controls"] + [
                f"{k}:{issue}" for k, p in projections.items() for issue in p["controls"]
            ]
            summary = dict(
                case=name,
                op=op,
                comparable=not controls,
                controls=controls,
                different_elements={k: p["different_elements"] for k, p in projections.items()},
                matches_archived_v=projections["v_proj"]["prefix_sha256"]
                == info["engine_v_sha256"],
                matches_native_full={
                    k: p["first_output_sha256"]["full"]
                    == results["linear"][k]["first_output_sha256"]["full"]
                    for k, p in projections.items()
                },
                p50_ms={
                    k: {g: round(t["p50"], 4) for g, t in p["timing_ms"].items()}
                    for k, p in projections.items()
                },
            )
            report["cases"][name].setdefault("summaries", {})[op] = summary
            comparable = comparable and not controls
            print("LINEAR", json.dumps(summary, separators=(",", ":"), allow_nan=False), flush=True)
    with _new_file(args.out, "x") as handle:
        json.dump(report, handle, allow_nan=False)
    if args.trace_dir:
        _trace_native(args.trace_dir, operands, weights, contract["device"])
    return int(not comparable)


if __name__ == "__main__":
    raise SystemExit(main())
