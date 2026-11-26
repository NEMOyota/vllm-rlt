"""File formats and immutable reproduction markers for the two PR1 cases."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

CANDIDATE = "606a60ce9bacdd5df1d25168d38665b3a0ef106d"
HARNESS = "136a775e02c8c63ac5335970189a1d50b6c3d4f2"
CASES = {
    "c4_h1": dict(
        seed=421559278, card=3, target="target-3", step=1, pair=[532, 30], hits=[128, 0, 0, 0]
    ),
    "total512_p256": dict(
        seed=114281796, card=0, target="target-0", step=9, pair=[302, 447], hits=[256]
    ),
}
ARMS = ("off", "miss", "reuse")
HERE = Path(__file__).resolve().parent


def read(p):
    return Path(p).is_file() and json.loads(Path(p).read_text())


def write(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(p)


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(trial):
    return {rid: [r["hit_tokens"], r["tokens"]] for rid, r in trial["requests"].items()}


def first(a, b):
    for i, (x, y) in enumerate(zip(a, b, strict=True), 1):
        if x != y:
            return dict(step=i, off=x, other=y)
    return None


def verdict(case, trials):
    spec = CASES[case]
    full = {}
    comparisons = {}
    for arm in ARMS:
        full[arm] = identity(trials[arm])
        if arm != "off":
            comparisons[arm] = {r: first(x[1], full[arm][r][1]) for r, x in full["off"].items()}
    hits_ok = all(
        [full[a][f"target-{i}"][0] for i in range(len(spec["hits"]))]
        == (spec["hits"] if a == "reuse" else [0] * len(spec["hits"]))
        for a in ARMS
    )
    expected = dict(step=spec["step"], off=spec["pair"][0], other=spec["pair"][1])
    match = (
        hits_ok
        and not any(comparisons["miss"].values())
        and comparisons["reuse"][spec["target"]] == expected
    )
    return dict(
        reproduction="REPRODUCED" if match else "MARKER_MISMATCH",
        correctness="FAIL"
        if any(d for arm in comparisons.values() for d in arm.values()) or not hits_ok
        else "PASS",
        hits_match=hits_ok,
        arms=full,
        first_differences=comparisons,
    )


def source_revision(src):
    # GitHub's pinned source archive expands this field with export-subst.
    if (src / ".git").exists():
        return git(src, "rev-parse", "HEAD")
    revision = (HERE / "revision.txt").read_text().strip()
    return revision if len(revision) == 40 else "archive-without-git-metadata"
