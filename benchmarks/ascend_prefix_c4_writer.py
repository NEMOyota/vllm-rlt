"""Locate the first C4 slot difference along the actual depth-zero V write path."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from benchmarks.ascend_prefix_c4_slot import cold_slot_run
from benchmarks.ascend_prefix_c4_writer_probe import FirstWriteProbe
from benchmarks.ascend_prefix_evidence import _arm_checks
from benchmarks.ascend_prefix_fix_trial import _identity, _schedule, load_replay_model
from benchmarks.ascend_prefix_layer0 import _compare
from benchmarks.ascend_prefix_matrix import git_info, save_report
from benchmarks.ascend_prefix_scenarios import suites

V_PATH = ("hidden", "norm", "v_input", "v_proj", "v_write", "v_read")


def _bit_differences(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def writer_experiment(model, backend, case, contract, previous, *, variant="clones"):
    ids = [f"target-{i}" for i in range(4)]
    focus = ids.index("target-2")
    probe = None

    def capture(engine, targets, donor):
        nonlocal probe
        probe = FirstWriteProbe(
            model, engine.cache_manager, ids, case.reusable_tokens, case.prefix + case.suffix
        )
        return probe

    run = cold_slot_run(model, backend, case, contract, variant, target_context=capture)
    buffers = probe.stages
    controls = list(_arm_checks(run)["issues"])
    if _identity(run) != _identity(previous):
        controls.append("archived_inputs_or_outputs")
    if _schedule(run) != _schedule(previous):
        controls.append("archived_schedule")
    for field in ("prefill_fingerprints", "fingerprints", "donor_sha256", "prefix_sha256"):
        if run[field] != previous[field]:
            controls.append(f"archived_{field}")
    if variant == "clones" and len({r["prompt_sha256"] for r in run["requests"]}) != 1:
        controls.append("cloned_prompts")

    # Inter-slot comparisons cover only the shared prefix. All counts are BF16 elements.
    stages = {
        name: {rid: _compare(values[focus], values[i]) for i, rid in enumerate(ids)}
        for name, values in buffers.items()
    }
    first, write_read, read_to_prefill, connections = {}, {}, {}, {}
    for i, rid in enumerate(ids):
        first[rid] = next(
            (name for name in V_PATH if stages[name][rid]["different_elements"]), None
        )
        connections[rid] = {
            "norm_to_v_input": _bit_differences(buffers["norm"][i], buffers["v_input"][i]),
            "v_proj_to_write": _bit_differences(buffers["v_proj"][i], buffers["v_write"][i]),
        }
        write_read[rid] = {}
        read_to_prefill[rid] = {}
        for kind in ("k", "v"):
            actual = buffers[kind + "_read"][i]
            write_read[rid][kind] = _bit_differences(buffers[kind + "_write"][i], actual)
            digest = hashlib.sha256(actual.contiguous().view(torch.uint8).numpy()).hexdigest()
            end = run["prefill_fingerprints"][rid]["sha256"][0][0][kind]
            read_to_prefill[rid][kind] = digest != end

    # Re-evaluate a differing projection element with the stored BF16 weights on CPU.
    # Only compare to one reference if the entire input row matches bit for bit.
    linear = model.model.layers[0].self_attn.v_proj
    for i, rid in enumerate(ids):
        diff = stages["v_proj"][rid]
        if diff["worst_index"] is None or diff["nonfinite_pairs"] or linear.bias is not None:
            continue
        token, channel = divmod(diff["worst_index"], linear.out_features)
        a, b = buffers["v_input"][focus, token], buffers["v_input"][i, token]
        if _bit_differences(a, b):
            continue
        w = linear.weight[channel].detach().cpu().double()
        reference = float(torch.dot(a.double(), w))
        if math.isfinite(reference):
            diff["sample"] = {
                "token_0based": token,
                "channel": channel,
                "fp64_dot": reference,
                "focus": float(buffers["v_proj"][focus, token, channel]),
                "slot": float(buffers["v_proj"][i, token, channel]),
            }

    summary = dict(
        case=case.name,
        variant=variant,
        focus=ids[focus],
        comparable=not controls,
        controls=controls,
        first_v_difference=first,
        address_issue=probe.address_issue,
        v_proj_different_elements={
            rid: s["different_elements"] for rid, s in stages["v_proj"].items()
        },
        write_read_mismatch=write_read,
        connection_mismatches={rid: v for rid, v in connections.items() if any(v.values())},
        changed_after_immediate_read={
            rid: v for rid, v in read_to_prefill.items() if any(v.values())
        },
    )
    return {"summary": summary, "stages": stages, "run": run}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    previous = json.loads(args.consumer.read_text())
    summary = previous.get("summary", {})
    if (
        summary.get("case") != "c4_h4"
        or not summary.get("comparable")
        or any(
            "prefill_fingerprints" not in previous.get("runs", {}).get(n, {})
            for n in ("clones", "base")
        )
    ):
        parser.error("expected a comparable c4_consumer JSON containing the base and clones runs")
    contract = previous["contract"]
    model = load_replay_model(parser, args, contract)
    case = next(c for c in suites()["full"] if c.name == "c4_h4")
    result = dict(
        commit=git_info("rev-parse", "HEAD"),
        consumer_commit=previous.get("commit"),
        contract=contract,
        runs={},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=False)
    try:
        for variant in ("clones", "base"):
            trial = writer_experiment(
                model,
                contract["backend"],
                case,
                contract,
                previous["runs"][variant],
                variant=variant,
            )
            result["runs"][variant] = trial
            save_report(args.out, result)
            print(
                "C4_WRITE",
                json.dumps(trial["summary"], separators=(",", ":"), allow_nan=False),
                flush=True,
            )
    except (Exception, KeyboardInterrupt) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        save_report(args.out, result)
        raise
    return int(any(not t["summary"]["comparable"] for t in result["runs"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
