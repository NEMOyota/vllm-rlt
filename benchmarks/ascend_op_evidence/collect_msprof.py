"""Run the installed msprof and retain raw exports, trace and an auditable kernel index."""

import argparse
import csv
import json
import shlex
import shutil
import subprocess
from pathlib import Path

from common import read, write


def run_logged(cmd, folder, name):
    (folder / (name + ".argv.json")).write_text(json.dumps(cmd, indent=2))
    with (folder / (name + ".log")).open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def helptext(cmd, path):
    r = subprocess.run([*cmd, "--help"], capture_output=True, text=True)
    s = r.stdout + r.stderr
    path.write_text(s)
    if not s.strip():
        raise RuntimeError("msprof help unavailable")
    return s


def index(folder, observations, selected):
    calls = (observations or {}).get("calls", [])
    labels = {
        e["label"]
        for e in calls
        if e["kind"] == "linear"
        and e["call"] == selected.get("call")
        and any(pos == selected.get("pos") for pos, row in e["target_rows"])
    }

    def field(row, *names):
        norm = {
            k.lower().replace(" ", "").replace("_", ""): v.strip()
            for k, v in row.items()
            if k and v
        }
        return next((norm[k] for k in names if k in norm), "")

    # Use only explicit device stream/time fields on MSTX exports. Never
    # bind a host timestamp, kernel substring or same-named op by itself.
    ranges = []
    for file in folder.rglob("*.csv"):
        if not any(s in file.name.lower() for s in ("mstx", "msprof_tx", "msproftx")):
            continue
        with file.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                name = field(r, "message", "markname", "rangename", "name")
                stream = field(r, "streamid")
                start = field(r, "starttime(us)", "timestamp(us)")
                duration = field(r, "duration(us)")
                if name in labels and stream and start and duration:
                    ranges.append((name, stream, float(start), float(start) + float(duration)))
    fields = [
        "file",
        "csv_row",
        "kernel",
        "type",
        "stream",
        "task",
        "start_us",
        "block_dim",
        "label",
    ]
    rows = []
    for file in sorted(folder.rglob("op_summary*.csv")):
        with file.open(encoding="utf-8-sig") as f:
            for n, r in enumerate(csv.DictReader(f), 2):
                kernel = field(r, "opname", "kernelname", "taskname")
                typ = field(r, "optype", "tasktype")
                stream, task = field(r, "streamid"), field(r, "taskid")
                stamp = field(r, "taskstarttime(us)", "starttime(us)")
                label = ""
                if stream and stamp:
                    matches = {
                        label for label, s, a, b in ranges if s == stream and a <= float(stamp) <= b
                    }
                    if len(matches) == 1:
                        label = matches.pop()
                if label or any(s in (kernel + typ).lower() for s in ("matmul", "gemm", "linear")):
                    rows.append(
                        [
                            str(file.relative_to(folder)),
                            n,
                            kernel,
                            typ,
                            stream,
                            task,
                            stamp,
                            field(r, "blockdim"),
                            label,
                        ]
                    )
    with (folder / "kernels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)
    linked = [r for r in rows if r[-1] and "matmul" in (r[2] + r[3]).lower()]
    write(
        folder / "correlation.json",
        dict(
            labels=sorted(labels),
            matched_matmul=linked,
            status="UNIQUE_STREAM_RANGE_MATCH" if len(linked) == 1 else "INSPECT_TIMELINE",
            timestamps="microseconds in same device stream",
            profiles=[str(p.relative_to(folder)) for p in folder.rglob("PROF_*") if p.is_dir()],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["app", "op"])
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--arm", choices=["off", "miss", "reuse"], required=True)
    # argparse.REMAINDER after a positional would swallow named options;
    # split at the explicit -- delimiter instead.
    import sys

    i = sys.argv.index("--")
    a = p.parse_args(sys.argv[1:i])
    cmd = sys.argv[i + 1 :]
    dest = a.folder.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    exe = shutil.which("msprof")
    if not exe:
        raise RuntimeError("msprof missing from active CANN PATH")
    tool = [exe] if a.kind == "app" else [exe, "op"]
    if a.kind == "op":
        replay = read(a.base / "replay.json")
        if (
            not replay
            or replay["independent_reproduction"] != "REPRODUCED"
            or not replay["device"].startswith("npu")
        ):
            raise RuntimeError("requires independent NPU reproduction for this case")
    h = helptext(tool, dest / "msprof-help.txt")
    opts = [f"--output={dest}"]
    if a.kind == "app":
        for key, val in (
            ("--task-time", "l1"),
            ("--ai-core", "on"),
            ("--aic-mode", "task-based"),
            ("--aic-metrics", "PipeUtilization"),
            ("--msproftx", "on"),
        ):
            if key in h:
                opts.append(f"{key}={val}")
    argv = tool + opts + ([f"--application={shlex.join(cmd)}"] if "--application" in h else cmd)
    run_logged(argv, dest, "capture")
    for n, prof in enumerate(sorted(dest.rglob("PROF_*"))):
        if a.kind == "app" and prof.is_dir() and "--export" in h:
            run_logged([exe, "--export=on", f"--output={prof}"], dest, f"export-{n}")
    index(
        dest,
        read(a.base / "profile" / a.arm / "observations.json"),
        read(a.base / "selection.json") or {},
    )
    logs = [p for p in dest.rglob("*.log") if p.stat().st_size < 30_000_000]
    write(
        dest / "exports.json",
        dict(
            op_summary=[str(p.relative_to(dest)) for p in dest.rglob("op_summary*.csv")],
            timelines=[str(p.relative_to(dest)) for p in dest.rglob("msprof*.json")],
            debug_line_messages=[
                line.strip()
                for p in logs
                for line in p.read_text(errors="replace").splitlines()
                if "debug_line" in line
            ],
        ),
    )
