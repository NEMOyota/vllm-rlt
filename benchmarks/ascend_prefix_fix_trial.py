"""Try fixed-row K/V projections against both the original and candidate cold outputs."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from benchmarks.ascend_prefix_cache import BLOCK_SIZE
from benchmarks.ascend_prefix_evidence import _arm_checks, _first_diff
from benchmarks.ascend_prefix_head import HeadProbe
from benchmarks.ascend_prefix_layer0 import _prefix_stages
from benchmarks.ascend_prefix_matrix import git_info, save_report
from benchmarks.ascend_prefix_projection import fixed_kv_rows
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_scenarios import suites
from benchmarks.ascend_prefix_workload import _prompts
from benchmarks.preemption import checkpoint_identity
from vllm_rlt.device import resolve_device
from vllm_rlt.models import OuroForCausalLM

CASES = ("total512_p384", "p1024_s8", "c4_h4")
ARMS = ("off", "reuse")


def _differences(left, right):
    differences = []
    for a, b in zip(left["requests"], right["requests"], strict=True):
        if (a["request_id"], a["prompt_sha256"]) != (b["request_id"], b["prompt_sha256"]):
            raise ValueError("replay comparison requires identical ordered target requests")
        pos = _first_diff(a["tokens"], b["tokens"])
        if pos is not None:
            ids = [r["tokens"][pos - 1] if len(r["tokens"]) >= pos else None for r in (a, b)]
            differences.append(dict(request_id=a["request_id"], position=pos, ids=ids))
    return differences


def _identity(run):
    return [
        tuple(
            r[k] for k in ("request_id", "prompt_sha256", "tokens", "prefill_start", "hit_tokens")
        )
        for r in run["requests"]
    ]


def _matches_recorded(run, native, decision):
    if _identity(run) != _identity(native):
        return False
    if not decision or "logits" not in run:
        return True
    scores = run["logits"][decision["request_id"]][decision["position"] - 1]
    recorded = dict(zip(scores["top_ids"], scores["top_logits"], strict=True))
    return all(
        token not in recorded or recorded[token] == value
        for token, value in zip(decision["ids"], decision["sampler_logits"], strict=True)
    )


def _schedule(run):
    return [(s["phase"], s["loop_count"], s["spans"]) for s in run["steps"]]


def _archived_point(priors):
    points = []
    for prior in priors:
        runs = prior["runs"]
        try:
            diffs = _differences(runs["off"], runs["reuse"])
        except ValueError:
            continue  # Only paired, identical target prompts can nominate a decision.
        points.append(min(diffs, key=lambda d: d["position"], default=None))
    return points[0] if points and all(p == points[0] for p in points) else None


def _check_producer(model, case, contract, rows):
    # C1 only: the archived cold donor actually used these two shapes. A C4
    # packed target does not have the same projection geometry as a lone donor.
    if case.concurrency != 1:
        return None
    p = case.reusable_tokens
    donor, _, _ = _prompts(
        case, "reuse", contract["output_tokens"], model.config.vocab_size, contract["seed"]
    )
    stages = ("k_proj", "v_proj")
    differences = {}
    for name, scope in (("native", nullcontext()), ("candidate", fixed_kv_rows(model, rows))):
        with scope:
            full = _prefix_stages(model, donor, p, names=stages)
            split = _prefix_stages(model, donor[:p], p, names=stages)
            differences[name] = {
                k: int((full[k].view(torch.int16) != split[k].view(torch.int16)).sum())
                for k in stages
            }
    return differences


def fix_trial(model, backend, case, contract, priors, rows):
    point = _archived_point(priors)
    native, candidate, head, work = {}, {}, {}, {}
    controls = []
    for mode in ARMS:
        probe = None

        def observe(engine, targets, donor):
            nonlocal probe
            probe = HeadProbe(
                engine.model_runner, model, point["request_id"], point["position"], point["ids"]
            )
            return probe

        args = (model, backend, case, mode, contract["output_tokens"], contract["seed"])
        native[mode] = replay_mode(*args, target_context=observe if point else None)
        if probe is not None:
            head[mode] = probe.report()
        with fixed_kv_rows(model, rows) as counters:
            candidate[mode] = replay_mode(*args)
        work[mode] = counters
        if (
            not counters["calls"]
            or counters["tiles"] * rows != counters["input_rows"] + counters["padded_rows"]
        ):
            controls.append(f"{mode}:projection_work")
        for variant, run in (("native", native[mode]), ("candidate", candidate[mode])):
            run["output_tokens_per_request"] = contract["output_tokens"]
            controls.extend(f"{variant}_{mode}:{issue}" for issue in _arm_checks(run)["issues"])
        different_donor = (
            native[mode]["donor_prefill_spans"] != candidate[mode]["donor_prefill_spans"]
        )
        if different_donor or _schedule(native[mode]) != _schedule(candidate[mode]):
            controls.append(f"{mode}:schedule")
    matches = [
        i
        for i, prior in enumerate(priors, 1)
        if all(_matches_recorded(prior["runs"][m], native[m], head.get(m)) for m in ARMS)
    ]
    native_diffs = _differences(native["off"], native["reuse"])
    if point and point not in native_diffs:
        controls.append("native:first_difference_changed")
    if point and point in native_diffs:
        for i, mode in enumerate(ARMS):
            if head[mode]["selected_id"] != point["ids"][i]:
                controls.append(f"{mode}:head_selection")
    within = _differences(candidate["off"], candidate["reuse"])
    cold = _differences(native["off"], candidate["off"])
    hot = _differences(native["off"], candidate["reuse"])
    comparable = bool(matches) and not controls
    return dict(
        case=case.name,
        comparable=comparable,
        matched_prior_replays=matches,
        controls=controls,
        native_off_vs_reuse=native_diffs,
        candidate_off_vs_reuse=within,
        native_off_vs_candidate_off=cold,
        native_off_vs_candidate_reuse=hot,
        preserves_baseline_and_parity=comparable and not (within or cold or hot),
        head=head,
        layer0_different_elements=_check_producer(model, case, contract, rows),
        projection_work=work,
        hit_tokens={m: [r["hit_tokens"] for r in candidate[m]["requests"]] for m in ARMS},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--rows", type=int, default=64)
    args = parser.parse_args()
    if args.rows < 1 or args.out.exists() or not args.model.is_dir():
        parser.error("supply a positive row count, a new output path and a local checkpoint")
    diag = json.loads(args.diag.read_text())
    names = list(dict.fromkeys(args.case or CASES))
    if diag.get("state") != "complete" or any(not diag.get("cases", {}).get(n) for n in names):
        parser.error("expected a completed diagnosis with replay records for all selected cases")
    contract = diag["contract"]
    device_type = contract["device"]
    if device_type not in ("cpu", "npu") or (
        contract["dtype"],
        contract["backend"],
        contract["loops"],
        contract["block_size"],
    ) != ("bfloat16", "ascend_eager" if device_type == "npu" else "torch", 4, BLOCK_SIZE):
        parser.error("unsupported original execution contract")
    if checkpoint_identity(args.model) != contract["checkpoint"]:
        parser.error("checkpoint differs from the diagnostic replay")
    device = resolve_device("npu:0" if device_type == "npu" else "cpu")
    if device.type == "npu":
        torch.npu.set_device(0)
    model = OuroForCausalLM.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
    cases = {c.name: c for c in suites()["full"]}
    report = dict(
        state="running",
        cases={},
        rows=args.rows,
        contract=contract,
        git_commit=git_info("rev-parse", "HEAD"),
        diagnostic_commit=diag.get("diagnostic_commit"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=False)
    try:
        for name in names:
            result = fix_trial(
                model, contract["backend"], cases[name], contract, diag["cases"][name], args.rows
            )
            report["cases"][name] = result
            save_report(args.out, report)
            line = {k: v for k, v in result.items() if k not in ("head", "projection_work")}
            line["head_gap_reuse_id_minus_off_id"] = {
                mode: {
                    key: observation[key][1] - observation[key][0]
                    for key in ("head_logits", "fp64_logits")
                }
                for mode, observation in result["head"].items()
            }
            print("FIX_TRIAL", json.dumps(line, separators=(",", ":"), allow_nan=False), flush=True)
        report["state"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(state="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_report(args.out, report)
    return int(any(not r["comparable"] for r in report["cases"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
