"""Replay the three historical token divergences with joint K/V in the synchronous Ouro engine."""

import argparse
import json
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.ascend_grouped_kv import operators
from benchmarks.ascend_prefix_evidence import _arm_checks, _first_diff
from benchmarks.ascend_prefix_logits_probe import LogitsProbe
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_scenarios import ARMS, _mismatches, suites
from benchmarks.ascend_prefix_workload import run_trial
from vllm_rlt.models import OuroForCausalLM

# target index, first differing generated token (1-based), cold ID, reuse ID.
HISTORICAL = {
    "p1024_s8": (0, 2, 198, 1116),
    "total512_p384": (0, 2, 198, 79),
    "c4_h4": (2, 8, 30, 93),
}


@contextmanager
def grouped_layer(attn, grouped_matmul, calls):
    # OuroAttention calls k_proj and then v_proj with the identical hidden tensor.
    # Keep the second output only until the paired v_proj call on this layer.
    k, v = attn.k_proj, attn.v_proj
    if k.bias is not None or v.bias is not None:
        raise ValueError("expected bias-free Ouro K/V")
    joint = operators({"k_proj": k.weight, "v_proj": v.weight}, grouped_matmul)["grouped"]
    pending = None

    def key(hidden):
        nonlocal pending
        if pending is not None:
            raise RuntimeError("K/V projection order changed")
        try:
            key_out, value_out = joint(hidden)
        except RuntimeError as exc:
            raise RuntimeError(
                f"grouped K/V failed for layer {attn.layer_idx}, X={tuple(hidden.shape)}"
            ) from exc
        pending = hidden, value_out
        return key_out

    def value(hidden):
        nonlocal pending
        if pending is None or pending[0] is not hidden:
            raise RuntimeError("K/V projection inputs differ")
        output = pending[1]
        pending = None
        calls[hidden.shape[0]] += 1
        return output

    with patch.object(k, "forward", key), patch.object(v, "forward", value):
        yield
        if pending is not None:
            raise RuntimeError("grouped V was not consumed")


def compare_outputs(reference, candidate):
    diffs = []
    for i, (a, b) in enumerate(zip(reference["requests"], candidate["requests"], strict=True)):
        if a["prompt_sha256"] != b["prompt_sha256"]:
            raise ValueError(f"target-{i} prompts differ across runs")
        if len(a["tokens"]) != len(b["tokens"]):
            raise ValueError(f"target-{i} output lengths differ across fixed-length runs")
        pos = _first_diff(a["tokens"], b["tokens"])
        if pos is not None:
            diffs.append((i, pos, a["tokens"][pos - 1], b["tokens"][pos - 1]))
    return diffs


def replay(model, backend, case, grouped_matmul):
    trials, calls_by_arm = {}, {}
    for mode in ("native", "grouped"):
        with ExitStack() as stack:
            calls = Counter()
            if mode == "grouped":
                for layer in model.model.layers:
                    stack.enter_context(grouped_layer(layer.self_attn, grouped_matmul, calls))
            trials[mode] = {}
            for arm in ARMS:
                calls.clear()
                trials[mode][arm] = run_trial(model, backend, case, arm, 16, 168)
                if mode == "grouped":
                    if not calls:
                        raise RuntimeError(f"{arm}: no grouped calls")
                    calls_by_arm[arm] = dict(calls)
    native, grouped = trials["native"], trials["grouped"]
    native_diff = compare_outputs(native["off"], native["reuse"])
    issues = {mode: list(_mismatches(case, [trial])) for mode, trial in trials.items()}
    native_controls = [
        p for p in issues["native"] if not (p["kind"] == "output_vs_off" and p["arm"] == "reuse")
    ]
    summary = dict(
        case=case.name,
        native_reproduced=native_diff == [HISTORICAL[case.name]] and not native_controls,
        native_off_vs_reuse=native_diff,
        native_controls=native_controls,
        grouped_issues=issues["grouped"],
        grouped_off_vs_reuse=compare_outputs(grouped["off"], grouped["reuse"]),
        native_off_vs_grouped_off=compare_outputs(native["off"], grouped["off"]),
        grouped_hits=[r["hit_tokens"] for r in grouped["reuse"]["requests"]],
    )
    return dict(summary=summary, grouped_calls_by_rows=calls_by_arm, trials=trials)


def replay_split(model, backend, case, grouped_matmul):
    """Replay grouped C4 with full cold, split cold and shared prefix at the same boundary."""
    runs, issues, prefill = {}, [], {}
    prefix, length = case.reusable_tokens, case.prefix + case.suffix
    spans_expected = {
        "off": [(0, length)],
        "split": [(0, prefix), (prefix, length - prefix)],
        "reuse": [(prefix, length - prefix)],
    }
    calls = Counter()
    with ExitStack() as stack:
        for layer in model.model.layers:
            stack.enter_context(grouped_layer(layer.self_attn, grouped_matmul, calls))
        for mode, expected in spans_expected.items():
            calls.clear()
            probe = None

            def observe(engine, targets, donor):
                nonlocal probe
                probe = LogitsProbe(engine.model_runner)
                return probe

            run = replay_mode(model, backend, case, mode, 16, 168, target_context=observe)
            run.update(
                logits=probe.report(),
                grouped_calls_by_rows=dict(calls),
                output_tokens_per_request=16,
            )
            checks = _arm_checks(run)
            issues.extend(f"{mode}:{msg}" for msg in checks["issues"] + checks["probe_issues"])
            if not calls or len(run["requests"]) != case.concurrency:
                issues.append(f"{mode}:incomplete_burst")
            spans = {r["request_id"]: [] for r in run["requests"]}
            for step in run["steps"]:
                for s in step["spans"]:
                    if s["phase"] == "prefill":
                        spans[s["request_id"]].append((s["token_start"], s["token_count"]))
            if any(actual != expected for actual in spans.values()):
                issues.append(f"{mode}:prefill_spans")
            prefill[mode] = spans["target-2"]
            runs[mode] = run
    focus = {m: r["requests"][2]["tokens"] for m, r in runs.items()}
    diffs = {
        f"{a}_vs_{b}": compare_outputs(runs[a], runs[b])
        for a, b in (("off", "reuse"), ("off", "split"), ("split", "reuse"))
    }
    same_history = all(tokens[:6] == focus["off"][:6] for tokens in focus.values())
    reproduced = diffs["off_vs_reuse"] == [(2, 7, 302, 726)] and focus["off"][7] == 93
    return dict(
        summary=dict(
            case=case.name,
            controls=issues,
            prior_grouped_focus_reproduced=reproduced,
            same_first_six=same_history,
            comparable_at_7=not issues and reproduced and same_history,
            focus_prefill=prefill,
            focus_at_7={m: r["logits"]["target-2"][6] for m, r in runs.items()},
            hits={m: [row["hit_tokens"] for row in r["requests"]] for m, r in runs.items()},
            **diffs,
        ),
        runs=runs,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="new JSONL file; default under results/")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--c4-split", action="store_true", help="Compare grouped full, split and cached C4"
    )
    args = parser.parse_args()
    if args.repeats < 1 or not args.model.is_dir():
        parser.error("use a local Ouro checkpoint directory and positive --repeats")
    import torch_npu

    torch.npu.set_device(0)
    model = OuroForCausalLM.from_pretrained(
        args.model, device=torch.device("npu:0"), dtype=torch.bfloat16
    )
    model.eval()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = args.out or Path(f"results/grouped-kv-cases-{stamp}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = {case.name: case for case in suites()["full"]}
    success = True
    with path.open("x") as f:
        print("RESULTS", path.resolve(), flush=True)
        print(
            json.dumps(
                dict(
                    model=str(args.model.resolve()),
                    device=torch.npu.get_device_name(0),
                    torch=str(torch.__version__),
                    torch_npu=torch_npu.__version__,
                    seed=168,
                    output_tokens=16,
                    repeats=args.repeats,
                )
            ),
            file=f,
            flush=True,
        )
        for name in ("c4_h4",) if args.c4_split else HISTORICAL:
            for repeat in range(1, args.repeats + 1):
                experiment = replay_split if args.c4_split else replay
                result = experiment(
                    model, "ascend_eager", cases[name], torch_npu.npu_grouped_matmul
                )
                info = result["summary"]
                info["repeat"] = repeat
                print(json.dumps(result), file=f, flush=True)
                if args.c4_split:
                    print("GROUPED_C4_SPLIT", json.dumps(info), flush=True)
                    success &= info["comparable_at_7"]
                    continue
                print("GROUPED_CASE", json.dumps(info), flush=True)
                success &= (
                    info["native_reproduced"]
                    and not info["grouped_issues"]
                    and not info["native_off_vs_grouped_off"]
                )
    return int(not success)


if __name__ == "__main__":
    raise SystemExit(main())
