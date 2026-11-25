"""Replay historical native BF16 prefix-cache failures on CANN 9.0.0.

Run from the test checkout with python -m benchmarks.ascend_pr1_failure_seeds.
Each card gets its own process, model, and disjoint set of prompt seeds.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

SEEDS = tuple(range(169, 179))
# One random draw per equal-width stratum of [0, 2**31), shuffled once before
# either chip was tested. Freeze the order so both hosts replay the same inputs.
BF16_SEEDS_30 = (
    1109648318,
    1727668039,
    1059945226,
    1443766105,
    1699062105,
    1593979132,
    1258657849,
    1893393689,
    273728779,
    797543261,
    700954282,
    1406478784,
    1332156500,
    942743421,
    891809316,
    207493098,
    2081658344,
    421559278,
    641794792,
    482563560,
    1995089171,
    114281796,
    2016684385,
    1560366951,
    1849591969,
    770929007,
    16672712,
    561128055,
    355234819,
    1179450028,
)
CASES = {
    "910B3": ("c4_h3", "c4_h4", "p128_s128", "p384_s8", "p512_s8"),
    "910B4": ("c4_h3", "c4_h4", "p128_s128", "p1024_s8", "p256_s8", "total512_p384"),
}
ROUNDS = 2
CANN = "9.0.0"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_card(chip, card, seeds, cases, precision, model_dir, out):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(card)
    os.environ["RLT_BENCH_DTYPE"] = "float16" if precision == "fp16" else "bfloat16"
    state = {
        "card": card,
        "chip": chip,
        "seeds": seeds,
        "cases": cases,
        "precision": precision,
        "status": "INCOMPLETE",
    }
    state_path = out / f"card-{card}.json"
    write_json(state_path, state)
    try:
        import torch
        import torch_npu

        from benchmarks import ascend_main_prefix_matrix as matrix
        from benchmarks.ascend_main_prefix_combo import environment
        from benchmarks.ascend_main_prefix_linear import benchmark_dtype, padded_fp32_linears
        from benchmarks.ascend_main_prefix_results import Report
        from benchmarks.ascend_prefix_pr1_hardware import identify
        from benchmarks.preemption import checkpoint_identity
        from vllm_rlt.models import OuroForCausalLM

        hardware = identify(torch_npu.npu, str(card), expected=chip)
        state["hardware"] = hardware
        if hardware["verification"] != "matched":
            raise RuntimeError(hardware["detail"])
        device = torch.device("npu:0")
        torch.npu.matmul.allow_hf32 = False
        dtype = benchmark_dtype()
        env = environment(device, dtype=dtype)
        env.pop("seed")  # This process tests multiple prompt seeds.
        env["native_allow_hf32"] = bool(torch.npu.matmul.allow_hf32)
        state["environment"] = env
        if env["cann"] != CANN or env["native_allow_hf32"]:
            raise RuntimeError(f"expected CANN {CANN} and HF32 off, observed {env}")
        state["checkpoint"] = checkpoint_identity(model_dir)
        write_json(state_path, state)
        model = OuroForCausalLM.from_pretrained(model_dir, device=device, dtype=dtype).eval()
        if model.lm_head.weight.dtype != dtype:
            raise RuntimeError(f"unexpected LM head dtype: {model.lm_head.weight.dtype}")
        args = argparse.Namespace(
            experiment=precision, warmups=0, repeats=ROUNDS, correctness_rounds=ROUNDS
        )
        chunk = 256 if precision == "fp32-linear" else None
        with padded_fp32_linears(model, chunk) as work:
            for seed in seeds:
                state["current_seed"] = seed
                print(f"{chip} card {card}, {precision}, seed {seed}", flush=True)
                report = Report("CORRECTNESS", out / f"seed-{seed}", linear_fp32_chunk=chunk)
                try:
                    report.record(
                        "ENV", **env, seed=seed, hardware=hardware, checkpoint=state["checkpoint"]
                    )
                    if work is not None:
                        report.record("LINEAR_MODULES", modules=work[0])
                    matrix.run_cases(
                        args,
                        model,
                        report,
                        cases,
                        modes=("main",),
                        seed=seed,
                        linear_work=work,
                    )
                finally:
                    report.stream.close()
        state.pop("current_seed")
        state["status"] = "COMPLETED"
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise
    finally:
        write_json(state_path, state)


def summarize(out, chip, precision, assignments, workers_ok, *, cases=None):
    cases = CASES[chip] if cases is None else cases
    rows = []
    for card, seeds in assignments.items():
        for seed in seeds:
            folder = out / f"seed-{seed}"
            details = {}
            trials = {}
            target_counts = {}
            path = folder / "raw.jsonl"
            if path.exists():
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.endswith("\n"):  # Interrupted write; keep completed cases.
                            break
                        event = json.loads(line)
                        if event["kind"] == "CASE_DETAILS":
                            details[event["case"]] = event["all_issues"]
                        elif event["kind"] == "TRIAL":
                            trials[event["case"], event["round"], event["arm"]] = set(
                                event["requests"]
                            )
                        elif event["kind"] == "INPUTS":
                            target_counts[event["case"]] = len(event["prompts"])
            for case in cases:
                issues = details.get(case)
                counts = Counter(item.get("comparison", "control") for item in issues or [])
                compared = {}
                for arm in ("miss", "reuse"):
                    compared[arm] = sum(
                        len(trials.get((case, i, "off"), set()) & trials.get((case, i, arm), set()))
                        for i in range(1, ROUNDS + 1)
                    )
                expected = ROUNDS * target_counts.get(case, 0)
                status = "INCOMPLETE"
                if issues:
                    status = "FAIL"
                elif issues == [] and expected and all(n == expected for n in compared.values()):
                    status = "PASS"
                rows.append(
                    dict(
                        chip=chip,
                        card=card,
                        precision=precision,
                        seed=seed,
                        case=case,
                        status=status,
                        miss_compared_requests=compared["miss"],
                        reuse_compared_requests=compared["reuse"],
                        off_vs_miss=counts["main.off_vs_main.miss"]
                        if issues is not None and compared["miss"]
                        else None,
                        off_vs_reuse=counts["main.off_vs_main.reuse"]
                        if issues is not None and compared["reuse"]
                        else None,
                        repeat_drift=counts["same_arm_first_run"] if issues is not None else None,
                        control_errors=counts["control"] if issues is not None else None,
                        first_issue=json.dumps(issues[0]) if issues else "",
                    )
                )
    rows.sort(key=lambda row: row["seed"])
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["status"] for row in rows)
    status = (
        "INCOMPLETE"
        if counts["INCOMPLETE"] or not workers_ok
        else ("FAIL" if counts["FAIL"] else "PASS")
    )
    result = dict(
        status=status,
        chip=chip,
        precision=precision,
        planned=len(rows),
        passed=counts["PASS"],
        failed=counts["FAIL"],
        incomplete=counts["INCOMPLETE"],
    )
    result["cases"] = {
        case: {
            status_name.lower(): [
                row["seed"] for row in rows if row["case"] == case and row["status"] == status_name
            ]
            for status_name in ("PASS", "FAIL", "INCOMPLETE")
        }
        for case in cases
    }
    write_json(out / "summary.json", result)
    print(json.dumps(result), flush=True)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[status]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", required=True, choices=CASES)
    parser.add_argument(
        "--bf16-30-seeds",
        action="store_true",
        help="run 30 stratified prompt seeds on this chip's historical native BF16 failures",
    )
    parser.add_argument(
        "--seed", type=int, action="append", help="prompt seed; repeat to select several"
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(set().union(*CASES.values())),
        help="select a historical failure from either chip; repeat to select several",
    )
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp16", "fp32-linear"),
        default="bf16",
        help="native BF16 (default), experimental FP16, or fixed 256-row FP32 Linear",
    )
    parser.add_argument(
        "--cards",
        default=os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0"),
        help="comma-separated device indices allocated on this host",
    )
    parser.add_argument("--model", type=Path, default=os.environ.get("MODEL"))
    parser.add_argument("--out-dir", type=Path, help="new output directory")
    parser.add_argument("--plan", action="store_true", help="print the plan without using a device")
    parser.add_argument("--worker-card", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        cards = [int(part) for part in args.cards.split(",")]
    except ValueError:
        parser.error("--cards requires comma-separated integers")
    if len(set(cards)) != len(cards) or min(cards) < 0 or len(cards) > len(SEEDS):
        parser.error("provide at most ten distinct nonnegative card indices")
    if args.bf16_30_seeds and (args.precision != "bf16" or args.seed or args.case):
        parser.error(
            "--bf16-30-seeds selects native BF16, the fixed seeds and this chip's failure cases"
        )
    seeds = list(BF16_SEEDS_30) if args.bf16_30_seeds else list(dict.fromkeys(args.seed or SEEDS))
    cases = list(dict.fromkeys(args.case or CASES[args.chip]))
    cards = cards[: len(seeds)]
    assignments = {card: seeds[index :: len(cards)] for index, card in enumerate(cards)}
    plan = dict(
        chip=args.chip,
        cann=CANN,
        precision=args.precision,
        rounds=ROUNDS,
        output_tokens=16,
        loops=4,
        cases=cases,
        seeds=seeds,
        arms=["off", "miss", "reuse"],
        assignments=assignments,
    )
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.model or not args.model.is_dir():
        parser.error("provide a local checkpoint directory with --model or MODEL")
    if args.worker_card is not None:
        run_card(
            args.chip,
            args.worker_card,
            assignments[args.worker_card],
            cases,
            args.precision,
            args.model,
            args.out_dir,
        )
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = "pr1-failure-seeds30" if args.bf16_30_seeds else "pr1-failure-seeds"
    out = args.out_dir or Path("results") / f"{prefix}-{args.chip}-{args.precision}-{stamp}"
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"result directory already exists: {out}")
    write_json(out / "plan.json", plan)
    print("RESULTS", out.resolve(), flush=True)
    repo = Path(__file__).resolve().parents[1]
    success = False
    try:
        with ExitStack() as stack:
            tree = repo
            if args.precision == "fp16":
                # Keep the experimental FP16 patch in a disposable checkout.
                tree = Path(stack.enter_context(tempfile.TemporaryDirectory())) / "fp16"
                sha = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip()
                subprocess.run(
                    ["git", "clone", "-q", "--no-checkout", str(repo), str(tree)], check=True
                )
                subprocess.run(["git", "checkout", "-q", "--detach", sha], cwd=tree, check=True)
                patch = tree / "benchmarks/ascend_prefix_fp16_replay.patch"
                (out / "fp16-runtime.patch").write_bytes(patch.read_bytes())
                subprocess.run(["git", "apply", str(patch)], cwd=tree, check=True)
            workers = []
            for card in cards:
                log = stack.enter_context((out / f"card-{card}.log").open("w", encoding="utf-8"))
                cmd = [
                    sys.executable,
                    "-m",
                    "benchmarks.ascend_pr1_failure_seeds",
                    "--chip",
                    args.chip,
                    "--cards",
                    args.cards,
                    "--precision",
                    args.precision,
                    "--model",
                    str(args.model.resolve()),
                    "--out-dir",
                    str(out.resolve()),
                    "--worker-card",
                    str(card),
                ]
                if args.bf16_30_seeds:
                    cmd.append("--bf16-30-seeds")
                for seed in args.seed or []:
                    cmd.extend(["--seed", str(seed)])
                for case in args.case or []:
                    cmd.extend(["--case", case])
                worker = stack.enter_context(
                    subprocess.Popen(cmd, cwd=tree, stdout=log, stderr=subprocess.STDOUT)
                )
                workers.append(worker)
            codes = [worker.wait() for worker in workers]
            success = all(code == 0 for code in codes)
    except (OSError, subprocess.CalledProcessError) as exc:
        (out / "setup-error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        print(exc, file=sys.stderr)
    return summarize(out, args.chip, args.precision, assignments, success, cases=cases)


if __name__ == "__main__":
    sys.exit(main())
