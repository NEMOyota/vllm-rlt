"""Create the candidate checkout without modifying the working benchmark repository."""

import argparse
import sys
from pathlib import Path

from common import CANDIDATE, HARNESS, HERE, git, sha, source_revision, write

repo = HERE.parents[1]
sys.path.insert(0, str(repo))

if __name__ == "__main__":
    from benchmarks import ascend_pr1_acceptance as acceptance

    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    tree = p.parse_args().directory.resolve()
    tree.mkdir(parents=True, exist_ok=False)
    git(tree, "init", "-q")
    git(
        tree,
        "bundle",
        "unbundle",
        str(repo / "benchmarks/ascend_pr1_package/pr1-fixed10-tests.bundle"),
    )
    git(tree, "checkout", "-q", "--detach", CANDIDATE)
    acceptance.prepare_harness(tree, tree)
    paths = [tree / "benchmarks" / (n + ".py") for n in acceptance.HELPERS]
    write(
        tree / "diagnostic-runtime.json",
        dict(
            candidate=git(tree, "rev-parse", "HEAD"),
            harness=HARNESS,
            tools=source_revision(repo),
            helpers={p.name: sha(p) for p in paths if p.exists()},
        ),
    )
    print(tree)
