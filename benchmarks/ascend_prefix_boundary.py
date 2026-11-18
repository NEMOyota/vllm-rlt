"""Compare the same logical prefix across full, split and shared C1 prefill paths."""

import argparse
import json
from pathlib import Path

import torch

from benchmarks.ascend_prefix_evidence import _first_diff
from benchmarks.ascend_prefix_logical_kv import LogicalKVProbe, compare_logical_kv
from benchmarks.ascend_prefix_logits_probe import LogitsProbe
from benchmarks.ascend_prefix_matrix import git_info, save_report
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_scenarios import suites
from benchmarks.preemption import checkpoint_identity
from vllm_rlt.device import resolve_device
from vllm_rlt.models import OuroForCausalLM


def _observed(model, backend, case, mode, output_tokens, seed):
    probes = {}
    request = "donor" if mode in ("reuse", "aligned") else "target-0"

    def lifetime(engine, targets, donor):
        probes["kv"] = LogicalKVProbe(engine.cache_manager, case.reusable_tokens, [request])
        return probes["kv"]

    def target(engine, targets, donor):
        probes["scores"] = LogitsProbe(engine.model_runner)
        return probes["scores"]

    run = replay_mode(
        model,
        backend,
        case,
        mode,
        output_tokens,
        seed,
        engine_context=lifetime,
        target_context=target,
    )
    (row,) = run["requests"]
    spans = [
        [span["token_start"], span["token_count"]]
        for step in run["steps"]
        for span in step["spans"]
        if span["phase"] == "prefill"
    ]
    return dict(
        request=row,
        donor_spans=[[s["token_start"], s["token_count"]] for s in run["donor_prefill_spans"]],
        target_spans=spans,
        prefix_hits=run["prefix_hits"],
        prefix_queries=run["prefix_queries"],
        fingerprint=probes["kv"].report()[request],
        logits=probes["scores"].report()["target-0"],
    )


def boundary_experiment(model, backend, case, output_tokens, seed, previous_passes):
    p, n = case.reusable_tokens, case.prefix + case.suffix
    if case.concurrency != 1 or case.hits == 0 or case.churn or not 0 < n - p <= p:
        raise ValueError("use a hot C1 case with one aligned prefix and one suffix chunk")
    modes = {
        mode: _observed(model, backend, case, mode, output_tokens, seed)
        for mode in ("off", "split", "reuse", "aligned")
    }
    matches = []
    for index, previous in enumerate(previous_passes, 1):
        for mode in ("off", "split", "reuse"):
            (recorded,) = previous["runs"][mode]["requests"]
            if any(
                recorded[k] != modes[mode]["request"][k]
                for k in ("prompt_sha256", "tokens", "hit_tokens", "prefill_start")
            ):
                break
        else:
            matches.append(index)
    issues = []
    for mode, data in modes.items():
        row = data["request"]
        hot = mode in ("reuse", "aligned")
        if (
            row["prompt_sha256"] != modes["off"]["request"]["prompt_sha256"]
            or row["finish_reason"] != "length"
            or len(row["tokens"]) != output_tokens
            or row["exit_depths"] != [4] * output_tokens
            or row["tokens"] != [score["selected_id"] for score in data["logits"]]
        ):
            issues.append(f"{mode}:target")
        if (data["prefix_queries"], data["prefix_hits"], row["hit_tokens"]) != (
            1,
            p if hot else 0,
            p if hot else 0,
        ):
            issues.append(f"{mode}:admission")
        split = [[0, p], [p, n - p]]
        target_spans = [[p, n - p]] if hot else split if mode == "split" else [[0, n]]
        donor_spans = split if mode in ("aligned", "split") else [[0, n]]
        if data["target_spans"] != target_spans or data["donor_spans"] != donor_spans:
            issues.append(f"{mode}:geometry")
    kv = {
        key: compare_logical_kv(modes[left]["fingerprint"], modes[right]["fingerprint"])
        for key, left, right in (
            ("donor__off_target", "reuse", "off"),
            ("donor__split_target", "reuse", "split"),
            ("off_target__split_target", "off", "split"),
            ("aligned_donor__split_target", "aligned", "split"),
        )
    }
    outputs = {}
    for left, right in (
        ("off", "split"),
        ("off", "reuse"),
        ("split", "aligned"),
        ("reuse", "aligned"),
        ("off", "aligned"),
    ):
        a, b = (modes[name]["request"]["tokens"] for name in (left, right))
        pos = _first_diff(a, b)
        outputs[f"{left}__{right}"] = (
            None
            if pos is None
            else dict(
                position=pos, ids=[seq[pos - 1] if pos <= len(seq) else None for seq in (a, b)]
            )
        )
    summary = dict(
        case=case.name,
        matched_prior_replays=matches,
        control_issues=issues,
        comparable=bool(matches) and not issues,
        kv=kv,
        output_first_diff=outputs,
    )
    return {"summary": summary, "modes": modes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag", required=True, type=Path, help="Saved prefix diagnosis JSON")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--case", choices=("total512_p384", "p1024_s8"), default="total512_p384")
    parser.add_argument("--out", required=True, type=Path, help="New detailed JSON")
    args = parser.parse_args()
    prior = json.loads(args.diag.read_text())
    if prior.get("state") != "complete" or not prior.get("cases", {}).get(args.case):
        parser.error("expected a completed diagnostic report containing the selected case")
    if args.out.exists() or not args.model.is_dir():
        parser.error("supply a new output file and a local checkpoint directory")
    contract = prior["contract"]
    if checkpoint_identity(args.model) != contract["checkpoint"]:
        parser.error("checkpoint files differ from the diagnostic run")
    backend = contract["backend"]
    device = resolve_device("npu:0" if contract["device"] == "npu" else "cpu")
    if (backend, contract["loops"], contract["block_size"], contract["dtype"]) != (
        "ascend_eager" if device.type == "npu" else "torch",
        4,
        16,
        "bfloat16",
    ):
        parser.error("unsupported original execution contract")
    if device.type == "npu":
        torch.npu.set_device(0)
    case = next(c for c in suites()["full"] if c.name == args.case)
    model = OuroForCausalLM.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
    result = boundary_experiment(
        model, backend, case, contract["output_tokens"], contract["seed"], prior["cases"][args.case]
    )
    result.update(
        git_commit=git_info("rev-parse", "HEAD"),
        diagnostic_commit=prior.get("diagnostic_commit"),
        contract=contract,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=False)
    save_report(args.out, result)
    print("BOUNDARY", json.dumps(result["summary"], separators=(",", ":")), flush=True)
    return int(not result["summary"]["comparable"])


if __name__ == "__main__":
    raise SystemExit(main())
