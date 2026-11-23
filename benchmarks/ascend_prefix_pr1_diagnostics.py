"""Numerical replay of PR1 cache arms and device profiles of native BF16."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import traceback
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from benchmarks import ascend_main_prefix_combo as combo
from benchmarks import ascend_main_prefix_matrix as matrix
from benchmarks.ascend_main_prefix_linear import (
    add_precision_arg,
    padded_fp32_linears,
    precision_settings,
)
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


def trace_logits(model, case, inputs, arm, *, linear_work=None):
    """Bind LM-head rows to the actual CODA spans in this synchronous replay only."""
    records = defaultdict(list)
    execute = ModelRunner._execute_batch
    dtype = model.lm_head.weight.dtype
    sdpa = F.scaled_dot_product_attention
    attention_dtypes = Counter()

    def observed_sdpa(q, k, v, *args, **kwargs):
        output = sdpa(q, k, v, *args, **kwargs)
        attention_dtypes[q.dtype, k.dtype, v.dtype, output.dtype] += 1
        return output

    def observed(runner, batch, prepared):
        if batch.operation != Operation.CODA:
            return execute(runner, batch, prepared)
        ids = [span.request.request_id for span in batch.spans]

        def capture(_module, arguments, values):
            hidden = arguments[0]
            require(len(values) == len(hidden) == len(ids), "unexpected CODA row layout")
            require(hidden.dtype == values.dtype == dtype, "unexpected LM head activation dtype")
            for rid, vector, scores in zip(ids, hidden, values, strict=True):
                if rid in inputs[2]:
                    records[rid].append(
                        (vector.detach().cpu().clone(), scores.detach().cpu().clone())
                    )

        with model.lm_head.register_forward_hook(capture):
            return execute(runner, batch, prepared)

    with (
        patch.object(ModelRunner, "_execute_batch", observed),
        patch.object(F, "scaled_dot_product_attention", observed_sdpa),
    ):
        result = matrix.trial(
            model, case, "main", arm, inputs, linear_work=linear_work, dtype=dtype
        )
    require(set(attention_dtypes) == {(dtype,) * 4}, f"unexpected SDPA dtypes: {attention_dtypes}")
    (q, k, v, output), calls = next(iter(attention_dtypes.items()))
    result["observed_dtypes"] = dict(
        model_weights=str(dtype),
        key_cache=result["key_cache_dtype"],
        value_cache=result["value_cache_dtype"],
        sdpa_q=str(q),
        sdpa_k=str(k),
        sdpa_v=str(v),
        sdpa_output=str(output),
        sdpa_calls=calls,
    )
    for rid, row in result["requests"].items():
        require(
            row["prefill_rows"] == len(inputs[2][rid]) - row["hit_tokens"],
            f"{arm} {rid}: first prefill did not cover the expected suffix",
        )
        # A device can select a different tied maximum from a CPU argmax. Use
        # the device's returned IDs for parity and its captured scores as evidence.
        steps = zip(records[rid], row["tokens"], strict=True)
        for pos, ((hidden, scores), token) in enumerate(steps, 1):
            if dtype == torch.float16 and not (
                torch.isfinite(hidden).all() and torch.isfinite(scores).all()
            ):
                raise FloatingPointError(
                    f"{arm} {rid} token {pos}: non-finite FP16 hidden or logits"
                )
            require(
                scores[token] == scores.max(),
                f"{arm} {rid} token {pos}: captured logits do not support the returned greedy ID",
            )
    return result, records


def margin(scores, tokens):
    values, ids = scores.float().topk(2)
    return dict(
        top2_ids=ids.tolist(),
        top2_scores=values.tolist(),
        margin=float(values[0] - values[1]),
        pair_scores=scores[tokens].tolist(),
    )


def diagnose_case(model, name, *, linear_work=None):
    case = matrix.FULL[name]
    inputs = combo.workload(case, model.config.vocab_size)
    baseline, reference = trace_logits(model, case, inputs, "off", linear_work=linear_work)
    trials = {"off": baseline}
    differences = []
    for arm in ("miss", "reuse"):
        current, observed = trace_logits(model, case, inputs, arm, linear_work=linear_work)
        trials[arm] = current
        for diff in combo.token_diffs(baseline, current, f"off_vs_{arm}"):
            rid, pos = diff["request"], diff["first_diff_1based"] - 1
            history = baseline["requests"][rid]["tokens"][:pos]
            require(current["requests"][rid]["tokens"][:pos] == history, "different history")
            left_hidden, left_scores = reference[rid][pos]
            right_hidden, right_scores = observed[rid][pos]
            tokens = [diff["expected"], diff["actual"]]
            # Independent dot products for just the two selected tokens and both
            # observed hidden rows. This isolates the head, not all upstream layers.
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
    arms = {}
    for arm, trial in trials.items():
        requests = {}
        for rid, row in trial["requests"].items():
            requests[rid] = {key: row[key] for key in ("hit_tokens", "prefill_rows", "tokens")}
        arms[arm] = {"requests": requests, "observed_dtypes": trial["observed_dtypes"]}
        if linear_work is not None:
            arms[arm]["linear_work"] = trial["linear_work"]
    return dict(
        case=name,
        primer_ids={"off_and_reuse": inputs[0], "miss": inputs[1]},
        target_prompt_ids=inputs[2],
        arms=arms,
        differences=differences,
    )


def run_drift(model, names, out, chunk):
    names = list(dict.fromkeys(names))
    settings = precision_settings(chunk, dtype=next(model.parameters()).dtype)
    results = {}
    count = 0
    with padded_fp32_linears(model, chunk) as work:
        for name in names:
            try:
                row = diagnose_case(model, name, linear_work=work)
            except (AssertionError, ValueError, RuntimeError, FloatingPointError) as exc:
                traceback.print_exc()
                status = "FAIL" if isinstance(exc, FloatingPointError) else "INCOMPLETE"
                summary = dict(status=status, detail=f"{type(exc).__name__}: {exc}")
                row = {"case": name}
            else:
                diffs = row["differences"]
                count += len(diffs)
                by_pair = Counter(diff["comparison"] for diff in diffs)
                summary = dict(
                    status="FAIL" if diffs else "PASS",
                    requests=len(row["target_prompt_ids"]),
                    off_vs_miss=by_pair["off_vs_miss"],
                    off_vs_reuse=by_pair["off_vs_reuse"],
                )
            results[name] = summary
            save(out / f"{name}.json", {**settings, **row, **summary})
            print(name, summary, flush=True)
    statuses = {row["status"] for row in results.values()}
    status = "INCOMPLETE" if "INCOMPLETE" in statuses else "FAIL" if "FAIL" in statuses else "PASS"
    save(
        out / "review.json",
        dict(**settings, cases=names, results=results, differences=count, status=status),
    )
    return {"INCOMPLETE": 2, "FAIL": 1, "PASS": 0}[status]


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
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    add_precision_arg(p)
    args = p.parse_args()
    if args.linear_fp32_chunk is not None and args.task != "drift":
        p.error("--linear-fp32-chunk applies to drift replays")
    if args.dtype == "float16" and (args.task != "drift" or args.linear_fp32_chunk is not None):
        p.error("--dtype float16 requires a native drift replay with no FP32 Linear override")
    dtype = getattr(torch, args.dtype)
    args.out.mkdir(parents=True, exist_ok=False)
    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    env = combo.environment(device, dtype=None if args.task == "env" else dtype)
    env.update(
        python=platform.python_version(),
        system=platform.platform(),
        allow_hf32=getattr(getattr(torch.npu, "matmul", None), "allow_hf32", None),
    )
    if args.task != "env":
        env.update(precision_settings(args.linear_fp32_chunk, dtype=dtype))
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
    model = OuroForCausalLM.from_pretrained(args.model, device=device, dtype=dtype).eval()
    require(all(param.dtype == dtype for param in model.parameters()), f"expected {dtype} weights")
    if args.task == "drift":
        names = list(args.case or KNOWN)
        if args.matrix and args.matrix.is_file():
            names.extend(json.loads(args.matrix.read_text()).get("failed", []))
        return run_drift(model, names, args.out, args.linear_fp32_chunk)

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
