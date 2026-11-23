"""Separate numerical and device-profile replays of the native BF16 PR1 path."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks import ascend_main_prefix_combo as combo
from benchmarks import ascend_main_prefix_matrix as matrix
from benchmarks.ascend_main_prefix_results import ids_digest
from benchmarks.ascend_prefix_combo_checks import require
from benchmarks.preemption import checkpoint_identity
from vllm_rlt import SamplingParams
from vllm_rlt.device import resolve_device
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Operation, RequestPhase
from vllm_rlt.worker.model_runner import ModelRunner

KNOWN = ("p1024_s8", "total512_p384", "c4_h3", "c4_h4")


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def trace_logits(model, case, inputs, arm):
    """Bind LM-head rows to the actual CODA spans in this synchronous replay only."""
    records = defaultdict(list)
    execute = ModelRunner._execute_batch

    def observed(runner, batch, prepared):
        if batch.operation != Operation.CODA:
            return execute(runner, batch, prepared)
        ids = [span.request.request_id for span in batch.spans]

        def capture(_module, arguments, values):
            hidden = arguments[0]
            require(len(values) == len(hidden) == len(ids), "unexpected CODA row layout")
            for rid, vector, scores in zip(ids, hidden, values, strict=True):
                if rid in inputs[2]:
                    records[rid].append(
                        (vector.detach().cpu().clone(), scores.detach().cpu().clone())
                    )

        with model.lm_head.register_forward_hook(capture):
            return execute(runner, batch, prepared)

    with patch.object(ModelRunner, "_execute_batch", observed):
        result = matrix.trial(model, case, "main", arm, inputs)
    for rid, row in result["requests"].items():
        selected = [int(scores.argmax()) for _, scores in records[rid]]
        require(selected == row["tokens"], f"logits do not match returned IDs for {rid}")
    return result, records


def margin(scores, tokens):
    values, ids = scores.float().topk(2)
    return dict(
        top2_ids=ids.tolist(),
        top2_scores=values.tolist(),
        margin=float(values[0] - values[1]),
        pair_scores=scores[tokens].tolist(),
    )


def diagnose_case(model, name):
    case = matrix.FULL[name]
    inputs = combo.workload(case, model.config.vocab_size)
    baseline, reference = trace_logits(model, case, inputs, "off")
    differences = []
    for arm in ("miss", "reuse"):
        current, observed = trace_logits(model, case, inputs, arm)
        for diff in combo.token_diffs(baseline, current, f"off_vs_{arm}"):
            rid, pos = diff["request"], diff["first_diff_1based"] - 1
            history = baseline["requests"][rid]["tokens"][:pos]
            require(current["requests"][rid]["tokens"][:pos] == history, "different history")
            left_hidden, left_scores = reference[rid][pos]
            right_hidden, right_scores = observed[rid][pos]
            tokens = [diff["expected"], diff["actual"]]
            # Independent dot products for just the two selected tokens and both
            # native hidden rows. This isolates the head, not all upstream layers.
            head = model.lm_head
            idx = torch.tensor(tokens, device=head.weight.device)
            weights = head.weight.index_select(0, idx).detach().cpu().double()
            bias = (
                0 if head.bias is None else head.bias.index_select(0, idx).detach().cpu().double()
            )
            differences.append(
                dict(
                    **diff,
                    prompt_sha256=ids_digest(inputs[2][rid]),
                    prior_generated_ids=history,
                    cached_tokens=current["hit_tokens"][rid],
                    off=margin(left_scores, tokens),
                    candidate=margin(right_scores, tokens),
                    max_abs_hidden_delta=float(
                        (left_hidden.float() - right_hidden.float()).abs().max()
                    ),
                    off_head_fp64_pair=(weights @ left_hidden.double() + bias).tolist(),
                    candidate_head_fp64_pair=(weights @ right_hidden.double() + bias).tolist(),
                )
            )
    return dict(case=name, differences=differences)


def checked_profile_files(folder, full):
    traces = sorted(folder.rglob("trace_view.json"))
    require(len(traces) == 1, f"expected one decoded NPU trace under {folder}")
    data = json.loads(traces[0].read_bytes())
    events = data if isinstance(data, list) else data["traceEvents"]
    operators = Counter()
    shapes = stacks = 0
    for event in events:
        name = event.get("name", "")
        if event.get("ph") == "X" and name.startswith("aten::"):
            operators[name] += 1
            fields = event.get("args", {})
            shapes += bool(fields.get("Input Dims"))
            stacks += bool(fields.get("Call stack"))
    require(bool(operators), "no CPU ATen events in the trace")

    def csv_rows(file):
        with file.open(encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)

    kernels = Counter()
    for path in folder.rglob("kernel_details.csv"):
        for row in csv_rows(path):
            name = row.get("Name") or row.get("Op Name")
            if name:
                kernels[name] += 1
    require(bool(kernels), "no decoded NPU kernels in kernel_details.csv")
    memory = [
        str(p.relative_to(folder))
        for p in folder.rglob("*.csv")
        if "memory" in p.name.lower() and next(csv_rows(p), None)
    ]
    if full:
        require(
            shapes and stacks and memory,
            "full capture missing shapes, call stacks or memory records",
        )
    return dict(
        trace=str(traces[0].relative_to(folder)),
        events=len(events),
        cpu_operators=dict(operators),
        device_kernels=dict(kernels),
        shape_events=shapes,
        stack_events=stacks,
        memory_tables=memory,
    )


def capture_context(folder, full):
    from torch_npu import profiler

    return profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        record_shapes=full,
        with_stack=full,
        profile_memory=full,
        experimental_config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level1
        ),
        on_trace_ready=profiler.tensorboard_trace_handler(str(folder), async_mode=False),
    )


def prefill_window(model, case, inputs, arm, capture):
    donor, miss, targets, _ = inputs
    engine = combo.open_engine(model, case, arm)
    try:
        require(
            engine.execution_config.packed_prefill and engine.cache_manager.dtype == torch.bfloat16,
            "expected native packed BF16",
        )
        params = SamplingParams(
            max_tokens=1,
            temperature=0,
            min_loops=4,
            max_loops=4,
            exit_threshold=1.0,
            ignore_eos=True,
        )
        engine.add_request("donor", miss if arm == "miss" else donor, params)
        combo.drain(engine, ["donor"], 1)
        rid, prompt = next(iter(targets.items()))
        engine.add_request(rid, prompt, params)
        engine.model_runner.synchronize()
        # Only the target PREFILL submission, through completion on its stream.
        with capture:
            first = engine.step()
            engine.model_runner.synchronize()
        schedule = engine.last_schedule
        start = case.starts(arm)[rid]
        require(
            schedule.phase == RequestPhase.PREFILL
            and schedule.num_tokens == len(prompt) - start
            and schedule.requests[0].request.prefill_window_origin == start,
            "capture does not cover the expected target prefill",
        )
        output = combo.drain(engine, [rid], 1, first)
        return dict(
            hit_tokens=start,
            prefill_rows=schedule.num_tokens,
            prompt_sha256=ids_digest(prompt),
            token_ids=output[rid]["tokens"],
        )
    finally:
        engine.close()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("env", "drift", "profile"))
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--matrix", type=Path, help="Also diagnose failures found by this BF16 matrix")
    p.add_argument("--case", choices=matrix.FULL, action="append")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    env = combo.environment(device)
    env.update(
        python=platform.python_version(),
        system=platform.platform(),
        allow_hf32=getattr(getattr(torch.npu, "matmul", None), "allow_hf32", None),
    )
    save(args.out / "environment.json", env)
    if args.task == "env":
        identity = checkpoint_identity(args.model)
        for path in sorted(args.model.glob("*token*.json")):
            identity[path.name] = dict(
                size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
        save(args.out / "checkpoint.json", identity)
        if shutil.which("npu-smi"):
            with (args.out / "npu-smi.txt").open("w") as log:
                subprocess.run(
                    ["npu-smi", "info"], stdout=log, stderr=subprocess.STDOUT, timeout=30
                )
        return 0
    model = OuroForCausalLM.from_pretrained(args.model, device=device, dtype=torch.bfloat16).eval()
    require(
        all(param.dtype == torch.bfloat16 for param in model.parameters()), "expected BF16 weights"
    )
    if args.task == "drift":
        names = list(args.case or KNOWN)
        if args.matrix and args.matrix.is_file():
            names.extend(json.loads(args.matrix.read_text()).get("failed", []))
        count = 0
        for name in dict.fromkeys(names):
            row = diagnose_case(model, name)
            save(args.out / f"{name}.json", row)
            count += len(row["differences"])
        save(args.out / "review.json", dict(cases=list(dict.fromkeys(names)), differences=count))
        return int(count > 0)

    name = (args.case or ["p512_s8"])[0]
    case = matrix.FULL[name]
    require(
        case.count == case.hot == 1 and case.starts("reuse")["target-0"] > 0,
        "profile requires a single request with an actual cache hit",
    )
    inputs = combo.workload(case, model.config.vocab_size)
    rows = {}
    for arm in ("miss", "reuse"):
        prefill_window(model, case, inputs, arm, nullcontext())
    for full in (False, True):
        for arm in ("miss", "reuse"):
            label = ("full-" if full else "ops-") + arm
            folder = args.out / label
            folder.mkdir()
            row = prefill_window(model, case, inputs, arm, capture_context(folder, full))
            row["artifacts"] = checked_profile_files(folder, full)
            save(folder / "evidence.json", row)
            rows[label] = row
    same_ids = len({tuple(r["token_ids"]) for r in rows.values()}) == 1
    save(args.out / "review.json", dict(case=name, captures=list(rows), same_token_ids=same_ids))
    return int(not same_ids)


if __name__ == "__main__":
    raise SystemExit(main())
