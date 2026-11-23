"""Run the paired PR1/PR2 release experiments on explicitly assigned Ascend cards."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from benchmarks import ascend_prefix_pr1_suite as suite
from benchmarks.ascend_main_prefix_matrix import FULL
from benchmarks.preemption import checkpoint_identity

PRECISIONS = ("native_bf16", "native_fp16", "linear_fp32_r256")
GEOMETRIES = ("near", "exact", "mixed", "outlier", "cold_grid", "single")
CORRECTNESS = 2
PERFORMANCE = 3
PY = sys.executable
ORACLE = "tests/test_ascend_prefix_prefill_groups.py"
STOCK_HTTP = suite.STOCK + suite.HTTP_CASE


@dataclass(frozen=True)
class Job:
    project: str
    precision: str
    phase: str
    card: int
    name: str
    args: tuple
    cases: tuple = ()


def plan(projects, cards):
    jobs = []
    for project in projects:
        pr1 = project == "pr1"
        entry_card = cards[(6 if pr1 else 7) % len(cards)]
        jobs.append(
            Job(
                project,
                "stock",
                "engine",
                entry_card,
                "upstream-npu",
                (
                    "pytest",
                    "-m",
                    "npu",
                    "--ignore=" + suite.CONTRACT,
                    "--deselect=" + STOCK_HTTP,
                    *(("--ignore=" + ORACLE,) if not pr1 else ()),
                    "tests",
                ),
            )
        )
        for precision in PRECISIONS:
            chunk = ("--linear-fp32-chunk", "256") if precision == PRECISIONS[2] else ()
            scenarios = tuple(FULL) if pr1 else GEOMETRIES
            shards = 4 if pr1 else 2
            for i in range(shards):
                cases = scenarios[i::shards]
                switches = tuple(arg for case in cases for arg in ("--case", case))
                common = (
                    "--repeats",
                    str(PERFORMANCE),
                    "--correctness-rounds",
                    str(CORRECTNESS),
                    "--warmups",
                    "0",
                    *chunk,
                    *switches,
                )
                cmd = (
                    (suite.MATRIX, "benefit", "--main-only")
                    if pr1
                    else ("benchmarks.ascend_prefix_pr2",)
                )
                stage = "cache-matrix-benefit" if pr1 else "geometry"
                jobs.append(
                    Job(
                        project,
                        precision,
                        "engine",
                        cards[(i if pr1 else i + 4) % len(cards)],
                        f"{stage}/shard-{i + 1}",
                        cmd + common,
                        cases,
                    )
                )
            for n in range(1, CORRECTNESS + 1):
                tests = (
                    [suite.CONTRACT, STOCK_HTTP] if pr1 else [suite.CONTRACT, STOCK_HTTP, ORACLE]
                )
                jobs.append(
                    Job(
                        project,
                        precision,
                        "engine",
                        entry_card,
                        f"entrypoint/round-{n}",
                        ("pytest", "-m", "npu", *tests),
                    )
                )
                if pr1:
                    jobs.append(
                        Job(
                            project,
                            precision,
                            "engine",
                            entry_card,
                            f"cache-replay/round-{n}",
                            ("benchmarks.ascend_prefix_pr1_replay", *chunk),
                        )
                    )
            for i, traffic in enumerate(("burst", "paced")):
                cmd = (
                    suite.SERVICE,
                    "--requests",
                    "32",
                    "--concurrency",
                    "4",
                    "--seed",
                    "168",
                    "--rps",
                    ("0", "3")[i],
                    "--rounds",
                    str(PERFORMANCE),
                    "--correctness-rounds",
                    str(CORRECTNESS),
                    *chunk,
                )
                cmd += ("--main-only",) if pr1 else ("--pr2-slack", "4", "--collect-opportunities")
                jobs.append(
                    Job(project, precision, "http", cards[i % len(cards)], f"http-{traffic}", cmd)
                )
    return jobs


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def check_rounds(job, folder):
    """Check evidence cardinality even when the benchmark reported token divergence."""
    if not job.cases and job.phase != "http":
        return
    expected = Counter()
    if job.phase == "http":
        modes = ("off", "main") if job.project == "pr1" else ("main", "exact", "similar")
        kind, cases = "SERVICE_RUN", ("",)
    else:
        modes = ("off", "miss", "reuse") if job.project == "pr1" else ("main", "exact", "similar")
        kind, cases = "TRIAL", job.cases
    for case in cases:
        for n in range(1, PERFORMANCE + 1):
            expected.update((case, n, mode) for mode in modes)
    actual = Counter()
    with (folder / "raw.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == kind and not row.get("warmup"):
                actual[row.get("case", ""), row["round"], row.get("arm", row["mode"])] += 1
    if expected != actual:
        raise ValueError(
            f"round coverage mismatch: missing={expected - actual}, extra={actual - expected}"
        )


def execute(job, tree, root, cpu_ids, port, common_env, manifests):
    folder = root / job.precision / job.name
    folder.mkdir(parents=True)
    env = dict(
        common_env,
        ASCEND_RT_VISIBLE_DEVICES=str(job.card),
        PYTHONPATH=str(tree),
        RLT_BENCH_DTYPE="float16" if job.precision == "native_fp16" else "bfloat16",
        RLT_BENCH_LINEAR_FP32_CHUNK="256" if job.precision == PRECISIONS[2] else "",
        VLLM_RLT_ASCEND_RESULTS=str(folder / "http-entry"),
        RLT_BENCH_PORT=str(port),
    )
    args = list(job.args)
    if job.phase == "http" and job.name in manifests:
        args.extend(["--reference-traffic", str(manifests[job.name])])
    if args[0] == "pytest":
        artifact = folder / "junit.xml"
        command = [PY, "-m", "pytest", "--run-npu", "-q", "--junitxml", str(artifact), *args[1:]]
    else:
        data = folder / "data"
        artifact = data / "review.json"
        command = [
            PY,
            "-m",
            *args,
            "--model",
            env["VLLM_RLT_OURO_CHECKPOINT"],
            "--out-dir",
            str(data),
        ]
    command = ["taskset", "-c", ",".join(map(str, cpu_ids)), *command]
    stage = suite.Stage("worker", command, artifact)
    result = suite.run_stage(stage, tree, env, folder)
    if result["status"] in ("PASS", "FAIL") and args[0] != "pytest":
        try:
            check_rounds(job, artifact.parent)
        except (OSError, ValueError, KeyError) as exc:
            result.update(status="INCOMPLETE", detail=str(exc))
    if job.phase == "http" and (artifact.parent / "traffic.json").is_file():
        manifests.setdefault(job.name, artifact.parent / "traffic.json")
    write_json(folder / "result.json", dict(job=asdict(job), cpus=cpu_ids, **result))
    return result["status"]


def port_map(cards):
    sockets = []
    try:
        ports = {}
        for card in cards:
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(("127.0.0.1", 0))
            ports[card] = sock.getsockname()[1]
        return ports
    finally:
        for sock in sockets:
            sock.close()


def run_phase(jobs, run_lane, cpus):
    lanes = defaultdict(list)
    for job in jobs:
        lanes[job.card].append(job)
    if not lanes:
        return
    workers = min(len(lanes), len(cpus))
    # With a smaller container cpuset, process waves of cards using disjoint CPU sets.
    items = list(lanes.items())
    for start in range(0, len(items), workers):
        wave = items[start : start + workers]
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            pending = [
                pool.submit(run_lane, card, lane, cpus[i :: len(wave)])
                for i, (card, lane) in enumerate(wave)
            ]
            for future in as_completed(pending):
                future.result()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pr1-repo", type=Path)
    p.add_argument("--pr2-repo", type=Path)
    p.add_argument("--pr1-ref", default="HEAD")
    p.add_argument("--pr2-ref", default="HEAD")
    p.add_argument(
        "--cards", required=True, help="ordered physical selectors, e.g. 0,1,2,3,4,5,6,7"
    )
    p.add_argument("--chip", required=True, choices=("910B3", "910B4"))
    p.add_argument("--model", type=Path, default=os.getenv("MODEL"))
    p.add_argument("--out", type=Path)
    p.add_argument("--list", action="store_true", help="print assignments without running")
    p.add_argument(
        "--device-only", action="store_true", help="use existing CPU regression evidence"
    )
    args = p.parse_args(argv)
    selectors = args.cards.split(",")
    if not all(s.isascii() and s.isdecimal() for s in selectors):
        p.error("--cards requires numeric physical card selectors")
    cards = list(map(int, selectors))
    if len(cards) != len(set(cards)) or len(cards) > 8:
        p.error("select one to eight distinct cards")
    projects = {
        pr: repo.resolve() for pr, repo in (("pr1", args.pr1_repo), ("pr2", args.pr2_repo)) if repo
    }
    if not projects:
        p.error("choose --pr1-repo and/or --pr2-repo")
    refs = {
        pr: suite.git(
            repo,
            "rev-parse",
            "--verify",
            "--end-of-options",
            getattr(args, pr + "_ref") + "^{commit}",
        )
        for pr, repo in projects.items()
    }
    jobs = plan(projects, cards)
    if args.list:
        print(
            json.dumps(
                dict(
                    candidates=refs,
                    correctness_rounds=CORRECTNESS,
                    performance_rounds=PERFORMANCE,
                    jobs=[asdict(j) for j in jobs],
                ),
                indent=2,
            )
        )
        return 0
    if not args.model or not args.model.is_dir():
        p.error("--model or MODEL must identify a local Ouro checkpoint")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (args.out or Path("results") / f"ascend-prefix-{args.chip}-{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    print("RESULTS", out, flush=True)
    roots = {pr: out / pr / sha / args.chip for pr, sha in refs.items()}
    env = dict(os.environ, VLLM_RLT_OURO_CHECKPOINT=str(args.model.resolve()))
    for key in (
        "PYTEST_ADDOPTS",
        "SKIP",
        "RLT_BENCH_PORT",
        "RLT_BENCH_DTYPE",
        "RLT_BENCH_LINEAR_FP32_CHUNK",
    ):
        env.pop(key, None)
    report = dict(
        candidates=refs,
        chip=args.chip,
        cards=cards,
        hardware={},
        cpu={},
        correctness_rounds=CORRECTNESS,
        performance_rounds=PERFORMANCE,
        stages={},
        status="INCOMPLETE",
    )

    def key(j):
        return "/".join((j.project, j.precision, j.name))

    report["stages"] = {key(j): dict(card=j.card, status="NOT_RUN") for j in jobs}
    write_json(out / "plan.json", [asdict(j) for j in jobs])
    write_json(out / "review.json", report)
    lock = Lock()
    with tempfile.TemporaryDirectory(prefix="rlt-prefix-parallel-") as tmp:
        trees = {}
        for pr, repo in projects.items():
            tree = Path(tmp) / pr
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--no-hardlinks",
                    "--no-checkout",
                    str(repo),
                    str(tree),
                ],
                check=True,
            )
            suite.git(tree, "checkout", "--quiet", "--detach", refs[pr])
            trees[pr, False] = tree
            fp16 = Path(tmp) / (pr + "-fp16")
            suite.git(tree, "worktree", "add", "--quiet", "--detach", str(fp16), "HEAD")
            trees[pr, True] = fp16
            manifest = roots[pr] / "environment"
            manifest.mkdir(parents=True)
            for name in ("ascend_prefix_fp16_replay.patch", "ascend_prefix_fp16_http.patch"):
                patch = tree / "benchmarks" / name
                (manifest / name).write_bytes(patch.read_bytes())
                suite.git(fp16, "apply", str(patch))
            write_json(
                manifest / "fp16-patches.json",
                {
                    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in manifest.glob("*.patch")
                },
            )
            if not args.device_only:
                cpu_dir = roots[pr].parent / "cpu-regression"
                proc = subprocess.run(
                    [PY, "-m", "benchmarks.ascend_prefix_pr1_suite", "cpu", "--out", str(cpu_dir)],
                    cwd=tree,
                    env=dict(env, PYTHONPATH=str(tree)),
                )
                report["cpu"][pr] = {0: "PASS", 1: "FAIL"}.get(proc.returncode, "INCOMPLETE")
        first_tree = trees[next(iter(projects)), False]
        for card in cards:
            path = out / f"hardware-card{card}.json"
            with (out / f"hardware-card{card}.log").open("w") as log:
                proc = subprocess.run(
                    [PY, suite.PROBE, str(path), args.chip],
                    cwd=first_tree,
                    env=dict(env, ASCEND_RT_VISIBLE_DEVICES=str(card)),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            report["hardware"][card] = (
                json.loads(path.read_text())
                if proc.returncode == 0
                else {"verification": "unavailable"}
            )
        write_json(out / "review.json", report)
        usable = [
            c for c in cards if report["hardware"][c]["verification"] in ("matched", "unverified")
        ]
        if usable:
            identity = out / "checkpoint.json"
            write_json(identity, checkpoint_identity(args.model))
            env["RLT_BENCH_CHECKPOINT_JSON"] = str(identity)
        ports = port_map(usable)

        def lane(card, tasks, cpus):
            manifests = {}
            for j in tasks:
                try:
                    status = execute(
                        j,
                        trees[j.project, j.precision == "native_fp16"],
                        roots[j.project],
                        cpus,
                        ports[card],
                        env,
                        manifests,
                    )
                    data = dict(status=status)
                except Exception as exc:
                    data = dict(status="INCOMPLETE", detail=f"{type(exc).__name__}: {exc}")
                with lock:
                    report["stages"][key(j)].update(data)
                    write_json(out / "review.json", report)

        for phase in ("engine", "http"):
            run_phase(
                [j for j in jobs if j.phase == phase and j.card in usable],
                lane,
                sorted(os.sched_getaffinity(0)),
            )
    statuses = set(report["cpu"].values()) | {s["status"] for s in report["stages"].values()}
    if statuses <= {"FAIL", "PASS"} and all(
        h["verification"] == "matched" for h in report["hardware"].values()
    ):
        report["status"] = "FAIL" if "FAIL" in statuses else "PASS"
    write_json(out / "review.json", report)
    lines = [
        "# Ascend prefix cache",
        "",
        f"**{report['status']}** · {args.chip}",
        "",
        "正确性取前两轮，性能取三轮。第三轮的控制或输出分歧会使性能对照失效。",
        "",
        "| 候选 / 精度 / 阶段 | 物理卡 | 状态 | 记录 |",
        "|---|---:|---|---|",
    ]
    for j in jobs:
        folder = roots[j.project] / j.precision / j.name
        data = folder.relative_to(out).as_posix()
        status = report["stages"][key(j)]["status"]
        links = [
            f"[{label}]({data}/{path})"
            for label, path in (
                ("结果", "result.json"),
                ("汇总", "data/review.json"),
                ("原始数据", "data/raw.jsonl"),
                ("服务指标", "data/service.csv"),
                ("引擎指标", "data/cases.csv"),
            )
            if (folder / path).is_file()
        ]
        lines.append(f"| {key(j)} | {j.card} | {status} | {' · '.join(links)} |")
    (out / "PR_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(report["status"], out / "PR_RESULTS.md", flush=True)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
