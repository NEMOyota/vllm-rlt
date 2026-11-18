"""Capture separate CPU/NPU timelines for selected prefix-cache benchmark arms."""

import argparse
import json
from contextlib import ExitStack, contextmanager
from functools import partial, wraps
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.ascend_prefix_scenarios import ARMS, suites
from benchmarks.ascend_prefix_workload import Case, run_trial
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.kernels.ascend_eager import AscendEagerAttention
from vllm_rlt.models import OuroForCausalLM


def marked(method):
    name = f"RLT/{method.__qualname__}"

    @wraps(method)
    def call(*args, **kwargs):
        with torch.profiler.record_function(name):
            return method(*args, **kwargs)

    return call


@contextmanager
def source_ranges():
    with ExitStack() as stack:
        for owner, name in (
            (KVCacheManager, "lookup_prefix"),
            (AscendEagerAttention, "_batched_prefill"),
            (AscendEagerAttention, "_forward"),
        ):
            stack.enter_context(patch.object(owner, name, marked(getattr(owner, name))))
        yield


def npu_profiler():
    try:
        import torch_npu.profiler as profiler

        required = {profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU}
        if not required.issubset(profiler.supported_activities()) or not torch.npu.is_available():
            raise RuntimeError("CPU and NPU profiler activities and an allocated NPU are required")
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError(
            "Prefix tracing requires torch_npu.profiler with CANN profiling support"
        ) from exc
    return profiler


@contextmanager
def trace_target(profiler, directory):
    torch.npu.synchronize()
    with (
        profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        ),
        source_ranges(),
    ):
        yield
        torch.npu.synchronize()


def capture(model, case: Case, output_tokens, seed, out, profiler, arms):
    for arm in dict.fromkeys(arms):
        directory = out / arm
        directory.mkdir()
        trial = partial(run_trial, model, "ascend_eager", case, arm, output_tokens, seed)
        trial()  # Warm the same shapes and kernels outside the profiler, using a fresh engine.
        sample = trial(target_context=partial(trace_target, profiler, directory))
        traces = sorted(path for path in directory.rglob("trace_view.json") if path.stat().st_size)
        if not traces:
            raise RuntimeError(
                f"{arm} produced no trace_view.json; check CANN profiler logs in {directory}"
            )
        print(
            "TRACE",
            json.dumps(
                {
                    "case": case.name,
                    "arm": arm,
                    "seed": seed,
                    "output_tokens_per_request": output_tokens,
                    "hit_tokens": [r["hit_tokens"] for r in sample["requests"]],
                    "traces": [str(p) for p in traces],
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cases = {case.name: case for case in suites()["full"]}
    parser.add_argument("--model", required=True, type=Path, help="Local Ouro checkpoint directory")
    parser.add_argument("--case", required=True, choices=cases)
    parser.add_argument(
        "--out", required=True, type=Path, help="New directory for the selected arm traces"
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        default=("miss", "reuse"),
        help="Trace each selected arm once (default: miss reuse)",
    )
    parser.add_argument(
        "--output-tokens", type=int, default=1, help="Generated tokens per target (default 1)"
    )
    parser.add_argument("--seed", type=int, default=168)
    args = parser.parse_args()
    if not args.model.is_dir() or args.output_tokens < 1:
        parser.error("a local checkpoint directory and a positive output length are required")
    try:
        profiler = npu_profiler()
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    out = args.out.resolve()
    try:
        out.mkdir(parents=True)
    except FileExistsError:
        parser.error("--out must name a new directory")
    torch.npu.set_device(0)
    model = OuroForCausalLM.from_pretrained(args.model, device="npu:0", dtype=torch.bfloat16)
    capture(model, cases[args.case], args.output_tokens, args.seed, out, profiler, args.arms)


if __name__ == "__main__":
    main()
