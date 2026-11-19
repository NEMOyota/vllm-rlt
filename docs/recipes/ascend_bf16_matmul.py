"""Replay one real BF16 V projection using only PyTorch and torch_npu."""

import argparse
from itertools import groupby
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def replay(x, weight):
    import torch
    import torch.nn.functional as F

    if x.shape != (576, 2048) or weight.shape != (2048, 2048):
        raise ValueError(f"Unexpected X or W shape: {x.shape}, {weight.shape}")
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype or not x.is_contiguous():
        raise ValueError("Expected contiguous BF16 X and BF16 weight")
    row, channel = 2, 18

    # Same full weight in every call. Slicing its rows would change matmul N.
    original = F.linear(x, weight).cpu()
    print(
        "ORIGINAL slot0_vs_slot2_v_diffs",
        (original[:128] != original[288:416]).sum().item(),
    )
    print("ORIGINAL rows_2_and_290", original[[2, 290], channel].tolist())
    reference_dot = torch.dot(x[row].cpu().double(), weight[channel].cpu().double())
    print(
        "FP64_DOT",
        reference_dot.item(),
        "rounded_bf16",
        reference_dot.bfloat16().item(),
    )
    x.copy_(x[row].clone().expand_as(x))
    reference = F.linear(x, weight)[:, channel].cpu().tolist()
    for m in (576, 512, 384):
        # All three inputs share the base address and only differ in M.
        values = F.linear(x[:m], weight)[:, channel].cpu().tolist()
        repeat = F.linear(x[:m], weight)[:, channel].cpu().tolist()
        runs = []
        start = 0
        for value, group in groupby(values):
            end = start + sum(1 for _ in group)
            runs.append((start, end - 1, value))
            start = end
        changes = sum(a != b for a, b in zip(values, reference[:m]))
        print(
            "ROW_MAP",
            m,
            "stable",
            values == repeat,
            "changed_vs_576",
            changes,
            "runs",
            runs,
        )


def pack(fixture, output, trace_dir):
    if not fixture.is_file():
        raise FileNotFoundError(fixture)
    traces = sorted(trace_dir.rglob("trace_view.json"))
    if len(traces) != 1:
        raise ValueError(f"Expected one trace_view.json under {trace_dir}, found {len(traces)}")
    details = sorted(trace_dir.rglob("kernel_details.csv"))
    instructions = (
        f"Run: python repro.py {fixture.name}\n"
        "Environment: Ascend 910B2, Ubuntu 22.04, CANN 9.0.0, torch 2.9.0, torch_npu 2.9.0.\n"
        "Case: BF16 F.linear without bias, K=N=2048, same input row repeated 576 times.\n"
        "Observed on the original host: channel 18 changes by one BF16 ULP. "
        "For M=512 and 576, rows 128..383 differ from rows 0..127. "
        "For M=384 all rows have the same value. Indices are zero-based.\n"
        "Expected changed_vs_576 counts at M=576/512/384: 0/0/256.\n"
        "Trace is from the original K/V projections before filling the rows. "
        "Device 0 stream 46: Tasks 44/45 = V 512/384, Task 47 = V 576.\n"
    )
    with output.open("xb") as stream:
        try:
            with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
                archive.write(__file__, "repro.py")
                archive.writestr("README.txt", instructions)
                archive.write(fixture, fixture.name)
                archive.write(traces[0], "trace_view.json")
                for path in details:
                    archive.write(path, str(Path("profiler") / path.relative_to(trace_dir)))
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    print(output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="original exported linear-inputs .pt")
    parser.add_argument("--pack", type=Path, metavar="ZIP", help="create a new local handoff ZIP")
    parser.add_argument("--trace-dir", type=Path, help="profiler directory for this fixture run")
    args = parser.parse_args()
    if args.pack:
        if args.trace_dir is None:
            parser.error("--pack requires --trace-dir")
        pack(args.fixture, args.pack, args.trace_dir)
        return
    if args.trace_dir is not None:
        parser.error("--trace-dir is only used with --pack")

    import torch
    import torch_npu

    data = torch.load(args.fixture, map_location="cpu", weights_only=True)
    torch.npu.set_device(0)
    print(
        "DEVICE",
        torch.npu.get_device_name(0),
        "torch",
        torch.__version__,
        "torch_npu",
        torch_npu.__version__,
    )
    weight = data["weights"]["v_proj"].to("npu:0")
    x = data["cases"]["c4_h4"]["x"].to("npu:0")
    with torch.inference_mode():
        replay(x, weight)


if __name__ == "__main__":
    main()
