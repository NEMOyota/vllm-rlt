"""Test whether changing only the four hit targets' suffix V projection restores cold output."""

import argparse
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch

from benchmarks.ascend_prefix_evidence import _arm_checks
from benchmarks.ascend_prefix_fix_trial import _differences, _identity, _schedule, load_replay_model
from benchmarks.ascend_prefix_layer0 import _compare
from benchmarks.ascend_prefix_logical_kv import LogicalKVProbe
from benchmarks.ascend_prefix_logits_probe import LogitsProbe
from benchmarks.ascend_prefix_matrix import git_info, save_report
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_scenarios import suites
from benchmarks.ascend_prefix_suffix_projection import SuffixVProjection


def _run_projection(model, backend, case, contract, mode, ids, reference):
    prefix, total = case.reusable_tokens, case.prefix + case.suffix
    arm = "off" if mode == "off" else "reuse"
    probes = {}

    def donor_context(engine, targets, donor):
        probes["fingerprints"] = LogicalKVProbe(engine.cache_manager, prefix, ["donor"])
        return probes["fingerprints"]

    @contextmanager
    def target_context(engine, targets, donor):
        with ExitStack() as stack:
            probes["logits"] = stack.enter_context(LogitsProbe(engine.model_runner))
            projection = SuffixVProjection(
                model,
                engine.cache_manager,
                ids,
                prefix,
                total,
                mode,
                reference=reference,
            )
            probes["projection"] = stack.enter_context(projection)
            yield

    run = replay_mode(
        model,
        backend,
        case,
        arm,
        contract["output_tokens"],
        contract["seed"],
        engine_context=donor_context if arm == "reuse" else None,
        target_context=target_context,
    )
    run.update({k: v.report() for k, v in probes.items()})
    run["output_tokens_per_request"] = contract["output_tokens"]
    return run, probes["projection"].inputs, probes["projection"].values


def suffix_trial(model, backend, case, contract, archive):
    if case.concurrency != 4 or case.hits != 4 or not case.reusable_tokens or case.churn:
        raise ValueError("expected four targets sharing a resident prefix")
    ids = [f"target-{i}" for i in range(case.concurrency)]
    runs, controls, first_layer = {}, [], {}
    cold_inputs = cold_values = hot_inputs = padded_values = None
    traversals = model.config.num_hidden_layers * model.config.total_ut_steps
    for mode, modifications in (
        ("off", 0),
        ("reuse", 0),
        ("first", 1),
        ("all", traversals),
        ("oracle", 1),
    ):
        reference = cold_values if mode == "oracle" else None
        run, inputs, values = _run_projection(model, backend, case, contract, mode, ids, reference)
        runs[mode] = run
        checks = _arm_checks(run)
        controls.extend(f"{mode}:{i}" for i in checks["issues"] + checks["probe_issues"])
        counts = run["projection"]
        if counts["observed_calls"] != traversals or counts["modified_calls"] != modifications:
            controls.append(f"{mode}:projection_coverage")

        if mode in ("off", "reuse"):
            prior = archive["runs"][mode]
            if _identity(run) != _identity(prior) or _schedule(run) != _schedule(prior):
                controls.append(f"{mode}:archived_inputs_outputs_or_schedule")
            if (
                "logits" in prior
                and prior["logits"]["target-2"][6:8] != run["logits"]["target-2"][6:8]
            ):
                controls.append(f"{mode}:archived_focus_logits")
            if (
                mode == "reuse"
                and "fingerprints" in prior
                and run["fingerprints"] != prior["fingerprints"]
            ):
                controls.append("reuse:archived_donor_kv")
        else:
            hot = runs["reuse"]
            if (
                _schedule(run) != _schedule(hot)
                or run["donor_prefill_spans"] != hot["donor_prefill_spans"]
            ):
                controls.append(f"{mode}:schedule")
            if run["fingerprints"] != hot["fingerprints"]:
                controls.append(f"{mode}:donor_kv")

        if mode == "off":
            cold_inputs, cold_values = inputs, values
            continue
        if mode == "reuse":
            hot_inputs = inputs
        elif not torch.equal(inputs.view(torch.int16), hot_inputs.view(torch.int16)):
            controls.append(f"{mode}:input_changed_before_intervention")
        if mode == "first":
            padded_values = values
        if mode == "all" and not torch.equal(
            values.view(torch.int16), padded_values.view(torch.int16)
        ):
            controls.append("all:first_layer_differs_from_first")
        first_layer[mode] = {}
        for i, rid in enumerate(ids):
            v_input = _compare(cold_inputs[i], inputs[i])
            v_output = _compare(cold_values[i], values[i])
            first_layer[mode][rid] = dict(
                v_input_different_elements=v_input["different_elements"],
                v_output_different_elements=v_output["different_elements"],
                v_input_max_abs=v_input["max_abs"],
                v_output_max_abs=v_output["max_abs"],
            )
            if v_input["nonfinite_pairs"] or v_output["nonfinite_pairs"]:
                controls.append(f"{mode}:{rid}:nonfinite")
            if mode == "oracle" and v_output["different_elements"]:
                controls.append(f"oracle:{rid}:cold_suffix_copy")

    summary = dict(
        case=case.name,
        comparable=not controls,
        controls=controls,
        first_layer=first_layer,
        hit_tokens={mode: [r["hit_tokens"] for r in run["requests"]] for mode, run in runs.items()},
        native_off_vs_reuse=_differences(runs["off"], runs["reuse"]),
        off_vs_first=_differences(runs["off"], runs["first"]),
        off_vs_all=_differences(runs["off"], runs["all"]),
        off_vs_oracle=_differences(runs["off"], runs["oracle"]),
    )
    return {"summary": summary, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True, help="previous C4_ORACLE JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    archive = json.loads(args.oracle.read_text())
    baseline = archive["summary"]
    if baseline["case"] != "c4_h4" or not baseline["comparable"]:
        parser.error("supply a comparable c4_h4 oracle report")
    contract = archive["contract"]
    model = load_replay_model(parser, args, contract)
    case = next(c for c in suites()["full"] if c.name == "c4_h4")
    result = suffix_trial(model, contract["backend"], case, contract, archive)
    result.update(
        contract=contract, commit=git_info("rev-parse", "HEAD"), oracle_commit=archive.get("commit")
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=False)
    save_report(args.out, result)
    summary = result["summary"].copy()
    first_layer = summary.pop("first_layer")
    summary["first_v_input_diff_elements_reuse_vs_off"] = [
        row["v_input_different_elements"] for row in first_layer["reuse"].values()
    ]
    summary["first_v_diff_elements_vs_off"] = {
        mode: [row["v_output_different_elements"] for row in targets.values()]
        for mode, targets in first_layer.items()
    }
    print("C4_SUFFIX", json.dumps(summary, separators=(",", ":"), allow_nan=False), flush=True)
    return int(not summary["comparable"])


if __name__ == "__main__":
    raise SystemExit(main())
