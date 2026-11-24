#!/usr/bin/env python3
"""Replay the five 910B3 native BF16 failures from the PR1 cache matrix."""

import argparse
import os
import sys
from pathlib import Path

CASES = ("c4_h3", "c4_h4", "p128_s128", "p384_s8", "p512_s8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=os.environ.get("MODEL"),
        help="local Ouro-1.4B checkpoint (default: $MODEL)",
    )
    parser.add_argument(
        "--card",
        type=int,
        default=os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0"),
        help="single card index visible to the container",
    )
    parser.add_argument("--out-dir", type=Path, help="new result directory (default: timestamped)")
    args = parser.parse_args()
    if args.model is None or not args.model.is_dir():
        parser.error("provide the local Ouro-1.4B directory with --model or MODEL")
    if args.card < 0:
        parser.error("--card must be nonnegative")
    if args.out_dir is not None and args.out_dir.exists():
        parser.error("--out-dir already exists; choose a new directory")

    repo = Path.cwd()
    if not (repo / "benchmarks/ascend_main_prefix_matrix.py").is_file():
        parser.error("run from the root of the vllm-rlt PR1 test checkout (dca7633)")
    sys.path.insert(0, str(repo))
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    os.environ["RLT_BENCH_DTYPE"] = "bfloat16"
    os.environ.pop("RLT_BENCH_CHECKPOINT_JSON", None)  # Fingerprint the supplied weights.

    try:
        import torch
        import torch_npu

        from benchmarks.ascend_main_prefix_matrix import main as run_matrix
        from benchmarks.ascend_prefix_pr1_hardware import identify

        hardware = identify(torch_npu.npu, str(args.card), expected="910B3")
        if hardware["verification"] != "matched":
            parser.exit(2, f"INCOMPLETE: {hardware['detail']}\n")
        torch.npu.matmul.allow_hf32 = False
    except (ImportError, OSError, RuntimeError, AttributeError) as error:
        parser.exit(2, f"INCOMPLETE: {error}\n")

    print(
        f"{hardware['runtime_device_name']} | torch {torch.__version__} | "
        f"torch_npu {torch_npu.__version__} | Python {sys.version.split()[0]}",
        flush=True,
    )
    options = [
        "bf16",
        "--model",
        str(args.model),
        "--main-only",
        "--warmups",
        "0",
        "--repeats",
        "2",
        "--correctness-rounds",
        "2",
    ]
    for case in CASES:
        options.extend(["--case", case])
    if args.out_dir is not None:
        options.extend(["--out-dir", str(args.out_dir)])
    return run_matrix(options)


if __name__ == "__main__":
    raise SystemExit(main())
