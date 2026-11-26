"""Pack portable replay inputs, source and derived evidence. Raw PROF trees remain at source."""

import argparse
import tarfile
from pathlib import Path

from common import HERE, sha, write

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("case", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    files = []
    for f in sorted(a.case.rglob("*")):
        if not f.is_file() or f.name == "package-manifest.json":
            continue
        if any(n.startswith("PROF_") for n in f.relative_to(a.case).parts):
            if f.name.startswith("op_summary") and f.suffix == ".csv":
                files.append(f)
            continue
        if f.suffix in (".json", ".pt", ".diff", ".log", ".txt", ".csv"):
            files.append(f)
    write(
        a.case / "package-manifest.json",
        dict(
            files={str(f.relative_to(a.case)): sha(f) for f in files},
            raw_profiles=[str(p.relative_to(a.case)) for p in a.case.rglob("PROF_*") if p.is_dir()],
        ),
    )
    with tarfile.open(a.output, "w:gz") as tar:
        for f in files + [a.case / "package-manifest.json"]:
            tar.add(f, arcname=str(f.relative_to(a.case)))
        for f in HERE.iterdir():
            if f.is_file():
                tar.add(f, arcname="scripts/" + f.name)
    print(a.output)
