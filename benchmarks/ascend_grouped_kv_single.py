"""Run every single-target preset on native and grouped Ouro K/V projections."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.ascend_grouped_kv_cases import compare_outputs, run_projection_trials
from benchmarks.ascend_prefix_scenarios import _mismatches, suites
from benchmarks.ascend_prefix_workload import expected_hits
from vllm_rlt.models import OuroForCausalLM


def single_cases():
    return [case for case in suites()["full"] if case.concurrency == 1]


def check_trial(case, trials, reference, *, candidate="grouped", candidate_reference=None):
    native, tested = trials["native"], trials[candidate]
    controls, parity_diffs, native_diffs = [], [], []
    for mode, trial in trials.items():
        for mismatch in _mismatches(case, [trial]):
            # The enclosing JSONL record carries the actual repeat index.
            del mismatch["trial"]
            if mismatch["kind"] == "output_vs_off" and mode == candidate:
                parity_diffs.append(mismatch)
            elif mismatch["kind"] == "output_vs_off" and mismatch["arm"] == "reuse":
                native_diffs.append(mismatch)
            else:
                controls.append(dict(mode=mode, **mismatch))
    cold_diffs = []
    try:
        cold_diffs = compare_outputs(native["off"], tested["off"])
        if drift := compare_outputs(reference, native["off"]):
            controls.append(dict(kind="native_cold_changed_between_repeats", differences=drift))
        if candidate_reference is not None and (
            drift := compare_outputs(candidate_reference, tested["off"])
        ):
            controls.append(dict(kind="candidate_cold_changed_between_repeats", differences=drift))
    except ValueError as exc:
        controls.append(dict(kind="incomparable_reference", detail=str(exc)))
    return {
        "passed": not (parity_diffs or cold_diffs or controls),
        f"{candidate}_parity_diffs": parity_diffs,
        f"native_off_vs_{candidate}_off": cold_diffs,
        "controls": controls,
        "native_off_vs_reuse": native_diffs,
        f"{candidate}_reuse_hits": [r["hit_tokens"] for r in tested["reuse"]["requests"]],
    }


def write_record(stream, **fields):
    print(json.dumps(fields, allow_nan=False), file=stream, flush=True)


def run_suite(model, backend, grouped_matmul, cases, repeats, output_tokens, seed, stream):
    failed = []
    cases_with = {
        field: []
        for field in (
            "grouped_parity_diffs",
            "native_off_vs_grouped_off",
            "controls",
            "native_off_vs_reuse",
        )
    }
    for index, case in enumerate(cases, 1):
        reference = None
        expected = expected_hits(case, "reuse")[1][0]
        print(f"[{index}/{len(cases)}] {case.name}  expected reuse={expected}", flush=True)
        case_ok = True
        for repeat in range(1, repeats + 1):
            try:
                trials, calls = run_projection_trials(
                    model, backend, case, grouped_matmul, output_tokens, seed
                )
                if reference is None:
                    reference = trials["native"]["off"]
                checks = check_trial(case, trials, reference)
            except (Exception, KeyboardInterrupt) as exc:
                write_record(
                    stream, kind="incomplete", case=case.name, repeat=repeat, error=repr(exc)
                )
                raise
            write_record(
                stream,
                kind="trial",
                case=case.name,
                repeat=repeat,
                **checks,
                grouped_calls_by_rows=calls,
                trials=trials,
            )
            for field, names in cases_with.items():
                if checks[field] and case.name not in names:
                    names.append(case.name)
            if not checks["passed"]:
                case_ok = False
                print(
                    "GROUPED_C1_FAIL",
                    json.dumps(dict(case=case.name, repeat=repeat, **checks)),
                    flush=True,
                )
        if not case_ok:
            failed.append(case.name)
        print("GROUPED_C1_CASE", case.name, "PASS" if case_ok else "FAIL", flush=True)

    hit_cases = sum(expected_hits(c, "reuse")[1][0] > 0 for c in cases)
    review = dict(
        state="complete",
        cases=len(cases),
        passed=len(cases) - len(failed),
        failed=failed,
        grouped_parity_failed=cases_with["grouped_parity_diffs"],
        cold_baseline_changed=cases_with["native_off_vs_grouped_off"],
        control_failed=cases_with["controls"],
        expected_hit_cases=hit_cases,
        expected_zero_hit_cases=len(cases) - hit_cases,
        native_reuse_diverged=cases_with["native_off_vs_reuse"],
    )
    write_record(stream, kind="review", **review)
    print("GROUPED_C1_REVIEW", json.dumps(review), flush=True)
    return review


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="list cases and expected hits without loading a model"
    )
    parser.add_argument("--model", type=Path, help="prepared local Ouro checkpoint directory")
    parser.add_argument(
        "--out", type=Path, help="new JSONL file; defaults to a timestamped file in results/"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=168)
    args = parser.parse_args()
    cases = single_cases()
    if args.list:
        for case in cases:
            print(
                json.dumps(
                    dict(**asdict(case), expected_reuse_hits=expected_hits(case, "reuse")[1][0])
                )
            )
        return 0
    if not args.model or not args.model.is_dir() or min(args.repeats, args.output_tokens) < 1:
        parser.error("provide a local --model and positive --repeats and --output-tokens")
    import torch_npu

    torch.npu.set_device(0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = args.out or Path(f"results/grouped-kv-single-{stamp}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        print("RESULTS", path.resolve(), flush=True)
        write_record(
            stream,
            kind="config",
            model=str(args.model.resolve()),
            device=torch.npu.get_device_name(0),
            torch=str(torch.__version__),
            torch_npu=torch_npu.__version__,
            seed=args.seed,
            output_tokens=args.output_tokens,
            repeats=args.repeats,
            cases=[asdict(case) for case in cases],
        )
        model = OuroForCausalLM.from_pretrained(
            args.model, device=torch.device("npu:0"), dtype=torch.bfloat16
        )
        model.eval()
        review = run_suite(
            model,
            "ascend_eager",
            torch_npu.npu_grouped_matmul,
            cases,
            args.repeats,
            args.output_tokens,
            args.seed,
            stream,
        )
    return int(bool(review["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
