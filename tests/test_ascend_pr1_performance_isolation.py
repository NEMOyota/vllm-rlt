import json
from threading import Barrier, Lock
from types import SimpleNamespace

import torch

from benchmarks import ascend_pr1_acceptance as acceptance
from tests.test_ascend_main_prefix_experiments import cpu_model  # noqa: F401


def test_performance_workers_finish_one_workload_before_the_next(monkeypatch):
    seeds = acceptance.FIXED_SEEDS[:3]
    jobs = acceptance.make_jobs(seeds, [0, 1], [3])
    jobs = [job for job in jobs if job.case in acceptance.WORKLOADS]
    seen = set()
    stages = list(acceptance.WORKLOADS)
    ready = Barrier(2, timeout=10)
    lock = Lock()

    def run_lane(cfg, tree, precision, lane):
        (job,) = lane
        if job.case == "W1" and job.seed in seeds[:2]:
            ready.wait()
        with lock:
            for previous in stages[: stages.index(job.case)]:
                assert all((previous, s) in seen for s in seeds)
            assert job.card == seeds.index(job.seed) % 2
            assert (job.case, job.seed) not in seen
            seen.add((job.case, job.seed))

    monkeypatch.setattr(acceptance, "run_lane", run_lane)
    acceptance.execute({}, None, "bf16", jobs)
    assert len(seen) == len(jobs)


def test_replay_releases_kv_after_every_arm(cpu_model, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setitem(acceptance.WORKLOADS, "W1", (48, 4, 2, 2, 32))
    job = acceptance.Job((3,), "W1", acceptance.FIXED_SEEDS[0], 0)
    with torch.inference_mode():
        result = acceptance.engine_performance(
            cpu_model, job, {"stream_size": 24, "window": 0.001}, tmp_path
        )
    log = [json.loads(line) for line in (tmp_path / "replay-memory.jsonl").read_text().splitlines()]
    assert len(result["controlled"]) == len(result["measurements"]) == 9
    assert len(log) >= 18 * 3
    for before, created, after in zip(log[::3], log[1::3], log[2::3], strict=True):
        assert [before["stage"], created["stage"], after["stage"]] == [
            "before_create",
            "after_create",
            "after_cleanup",
        ]
        assert after["kv_released"]
        for key in ("pid", "phase", "round", "arm"):
            assert before[key] == created[key] == after[key]
    assert all(
        row["hits"] == {32 if row["arm"] == "reuse" else 0: 2} for row in result["controlled"]
    )


def test_npu_cleanup_collects_before_emptying_allocator(monkeypatch):
    steps = []
    device = SimpleNamespace(type="npu")
    model = SimpleNamespace(lm_head=SimpleNamespace(weight=SimpleNamespace(device=device)))
    npu = SimpleNamespace(
        synchronize=lambda d: steps.append(d), empty_cache=lambda: steps.append("empty")
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(acceptance, "gc", SimpleNamespace(collect=lambda: steps.append("collect")))
    acceptance.release_replay_memory(model)
    assert steps == [device, "collect", "empty"]
