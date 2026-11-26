"""One baseline (two original rotating rounds), or one instrumented arm in an isolated process."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import (
    ARMS,
    CANDIDATE,
    CASES,
    HARNESS,
    HERE,
    git,
    identity,
    read,
    sha,
    source_revision,
    verdict,
    write,
)


def load_inputs(matrix, acceptance, name, vocab, history, override=None):
    spec = CASES[name]
    data = matrix.combo.workload(matrix.FULL[name], vocab, seed=spec["seed"])
    paths = [history / "inputs.json", history / "result.json"]
    expected = None
    supplied = None
    for p in ([override] if override else []) + paths:
        obj = read(p)
        if not obj:
            continue
        obj = obj.get("inputs", obj)
        if not isinstance(obj, dict) or "donor" not in obj:
            continue
        if isinstance(obj["donor"], list):
            current = (obj["donor"], obj["miss"], obj["targets"], obj["fillers"])
            if supplied is not None and current != supplied:
                raise ValueError("conflicting archived request IDs")
            supplied = current
        else:
            if expected and obj != expected:
                raise ValueError("conflicting original input hashes")
            expected = obj
    raw = history / "matrix/raw.jsonl"
    if raw.exists():
        with raw.open() as f:
            for line in f:
                item = json.loads(line)
                if item.get("kind") == "INPUTS":
                    if item["case"] != name or item["seed"] != spec["seed"]:
                        raise ValueError("wrong raw trial")
                    raw_hash = dict(
                        donor=item["donor_sha256"],
                        miss=item["miss_sha256"],
                        targets=item["prompts"],
                        fillers=[],
                    )
                    if expected and raw_hash != expected:
                        raise ValueError("raw input hashes disagree")
                    expected = raw_hash
                    break
    if supplied is not None:
        if list(supplied[2]) != list(data[2]):
            raise ValueError("wrong archived request order")
        data = supplied
    hashes = acceptance.ids_fingerprint(data)
    validation = (
        ("MATCH" if hashes == expected else "MISMATCH") if expected else "ORIGINAL_HASH_UNAVAILABLE"
    )
    return data, dict(
        source="archived_ids" if supplied is not None else "pinned_seed_generator",
        seed=spec["seed"],
        validation=validation,
        original=expected,
        actual=hashes,
    )


def main(args):
    spec = CASES[args.case]
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(spec["card"])
    os.environ["RLT_BENCH_DTYPE"] = "bfloat16"
    import torch
    import torch_npu
    from probes import Probes
    from torch_npu.utils.collect_env import get_cann_version

    import vllm_rlt
    from benchmarks import ascend_main_prefix_matrix as matrix
    from benchmarks import ascend_pr1_acceptance as acceptance
    from benchmarks.ascend_main_prefix_linear import padded_fp32_linears
    from benchmarks.ascend_prefix_pr1_hardware import identify
    from benchmarks.preemption import checkpoint_identity
    from vllm_rlt.models import OuroForCausalLM

    tree = Path.cwd()
    if (
        git(tree, "rev-parse", "HEAD") != CANDIDATE
        or Path(vllm_rlt.__file__).resolve().parents[1] != tree
    ):
        raise RuntimeError("launch inside the prepared 606a60c runtime with PYTHONPATH=$PWD")
    cfg = read(args.report / "run.json")
    if (
        not cfg
        or cfg["candidate_sha"] != CANDIDATE
        or cfg["harness"] != HARNESS
        or cfg["chip"] != args.chip
    ):
        raise ValueError("--report must point to the original matching results-910Bx directory")
    if cpus := cfg.get("cpu_affinity", {}).get(str(spec["card"])):
        os.sched_setaffinity(0, cpus)
    base = args.out / args.chip / args.case / args.precision
    folder = base / "baseline" if args.mode == "baseline" else base / args.mode / args.arm
    folder.mkdir(parents=True, exist_ok=False)
    torch.npu.matmul.allow_hf32 = False
    hw = identify(torch.npu, str(spec["card"]), args.chip)
    cann = str(get_cann_version())
    weights = checkpoint_identity(args.model)
    tools = HERE.parents[1]
    diffs = {}
    for label, src in (("runtime", tree), ("scripts", tools)):
        path = folder / (label + ".diff")
        if label == "scripts" and not (src / ".git").exists():
            diffs[label] = dict(
                head=source_revision(src), patch=None, source="archive; see file hashes"
            )
            continue
        path.write_bytes(
            subprocess.check_output(["git", "-C", str(src), "diff", "--binary", "HEAD"])
        )
        diffs[label] = dict(
            head=git(src, "rev-parse", "HEAD"),
            patch=path.name,
            sha256=sha(path),
            status=git(src, "status", "--porcelain", "--untracked-files=normal"),
        )
    env = dict(
        hardware=hw,
        cann=cann,
        torch=str(torch.__version__),
        torch_npu=str(torch_npu.__version__),
        model=str(args.model.resolve()),
        checkpoint=weights,
        physical_card=spec["card"],
        logical_card=torch.npu.current_device(),
        allow_hf32=torch.npu.matmul.allow_hf32,
        revisions=diffs,
        prepared=read(tree / "diagnostic-runtime.json"),
        diagnostic_files={p.name: sha(p) for p in HERE.glob("*.py")},
        executable=sys.executable,
        cann_home=os.getenv("ASCEND_HOME_PATH"),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
    )
    write(folder / "environment.json", env)
    if (
        hw["verification"] != "matched"
        or torch.npu.matmul.allow_hf32
        or cann != args.cann
        or weights != cfg["checkpoint"]
    ):
        raise RuntimeError("environment or checkpoint mismatch; see environment.json")
    if args.cann == "9.0.0" and (
        str(torch.__version__).split("+")[0] != "2.9.0"
        or str(torch_npu.__version__).split("+")[0] != "2.9.0.post2"
    ):
        raise RuntimeError("original run requires torch 2.9.0 / torch_npu 2.9.0.post2")
    if args.precision == "fp32-linear" and args.case != "total512_p256":
        raise ValueError("unsupported FP32 case")
    history = args.report / "bf16" / args.case / f"seed-{spec['seed']}"
    vocab = read(args.model / "config.json")["vocab_size"]
    inputs, proof = load_inputs(matrix, acceptance, args.case, vocab, history, args.input_ids)
    write(
        folder / "inputs.json",
        dict(zip(("donor", "miss", "targets", "fillers"), inputs, strict=True)),
    )
    write(folder / "fingerprints.json", proof)
    if proof["validation"] == "MISMATCH":
        raise ValueError("input fingerprint mismatch; original and actual saved")
    orig_commit = git(tree, "diff", "HEAD", "--", "vllm_rlt")
    if orig_commit:
        raise RuntimeError("runtime differs from candidate; patch saved")
    case = matrix.FULL[args.case]
    with torch.inference_mode():
        model = OuroForCausalLM.from_pretrained(
            args.model, device=torch.device("npu:0"), dtype=torch.bfloat16
        ).eval()
        with padded_fp32_linears(model, 256 if args.precision == "fp32-linear" else None) as work:

            def trial(arm):
                return matrix.trial(model, case, "main", arm, inputs, linear_work=work)

            if args.mode == "baseline":
                rounds = []
                for i in range(2):
                    t = {}
                    for _, arm in matrix.trial_order(i, ("main",)):
                        try:
                            t[arm] = trial(arm)
                        except Exception as exc:
                            write(
                                folder / "error.json",
                                dict(
                                    round=i + 1, arm=arm, type=type(exc).__name__, message=str(exc)
                                ),
                            )
                            raise
                        write(folder / f"round-{i + 1}.json", t)
                    rounds.append(t)
                    write(folder / f"round-{i + 1}.json", t)
                reports = [verdict(args.case, t) for t in rounds]
                write(
                    folder / "result.json",
                    dict(
                        rounds=reports,
                        stable=all(identity(rounds[0][a]) == identity(rounds[1][a]) for a in ARMS),
                    ),
                )
                return
            baseline = read(base / "baseline/result.json")
            if (
                not baseline
                or not baseline["stable"]
                or any(r["reproduction"] != "REPRODUCED" for r in baseline["rounds"])
            ):
                raise RuntimeError(
                    "baseline marker mismatch; inspect both uninstrumented rounds before collecting"
                )
            selection = read(base / "selection.json") or {}
            if args.mode == "dump" and not selection.get("call"):
                raise RuntimeError("run compare first")
            window = None
            if args.mode == "window":
                from torch_npu import profiler

                window = profiler.profile(
                    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                    record_shapes=True,
                    on_trace_ready=profiler.tensorboard_trace_handler(str(folder / "profile")),
                    experimental_config=profiler._ExperimentalConfig(
                        profiler_level=profiler.ProfilerLevel.Level1
                    ),
                )
            obs = Probes(
                model,
                spec,
                inputs,
                "profile" if args.mode == "window" else args.mode,
                folder,
                selection,
                window,
            )
            with obs.installed():
                t = trial(args.arm)
            ok = identity(t) == baseline["rounds"][0]["arms"][args.arm]
            write(
                folder / "result.json",
                dict(
                    matches_baseline=ok,
                    trial=t,
                    linear_modules=dict(work[1]) if work else {},
                    head=obs.head,
                ),
            )
            if args.mode in ("scan", "dump"):
                scores = torch.load(folder / "head.pt", weights_only=True)["logits"]
                sampled = t["requests"][spec["target"]]["tokens"][spec["step"] - 1]
                if scores[sampled] != scores.max():
                    raise RuntimeError(
                        "captured head row does not support the returned greedy token"
                    )
            if not ok or (args.mode == "dump" and not obs.saved):
                raise RuntimeError(
                    "probe changed tokens/hits or selected call missing; evidence saved"
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("report", "model", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--chip", choices=["910B3", "910B4"], required=True)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument(
        "--mode", choices=["baseline", "scan", "profile", "window", "dump"], required=True
    )
    p.add_argument("--arm", choices=ARMS, default="off")
    p.add_argument("--precision", choices=["bf16", "fp32-linear"], default="bf16")
    p.add_argument("--input-ids", type=Path)
    p.add_argument("--cann", default="9.0.0")
    main(p.parse_args())
