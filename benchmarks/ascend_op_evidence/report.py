"""Compare matched logical coordinates and summarize each case."""

import argparse
from pathlib import Path

from common import ARMS, CASES, read, write


def differences(left, right):
    def key(e):
        return e["call"], e["kind"], e["pos"]

    index = {key(e): e for e in left}
    if len(index) != len(left) or len({key(e) for e in right}) != len(right):
        raise ValueError("ambiguous observation key")
    output = []
    missing = []
    for e in right:
        k = key(e)
        a = index.get(k)
        if a is None:
            missing.append(k)
        elif a["digest"] != e["digest"]:
            output.append(dict(call=e["call"], kind=e["kind"], pos=e["pos"], left=a, right=e))
    return output, missing, list(index.keys() - {key(e) for e in right})


def summarize(base, case):
    data = dict(
        baseline=read(base / "baseline/result.json") or None,
        comparisons={},
        same_history_logits={},
        earliest_observed_difference=None,
        independent_replay=read(base / "replay.json") or None,
    )
    if not data["baseline"]:
        data["status"] = "INCOMPLETE" if (base / "baseline").exists() else "NOT_RUN"
        data["error"] = read(base / "baseline/error.json") or None
        return data
    rounds = data["baseline"]["rounds"]
    data["status"] = (
        "REPRODUCED"
        if data["baseline"]["stable"] and all(r["reproduction"] == "REPRODUCED" for r in rounds)
        else "MARKER_MISMATCH"
    )
    data["cache_correctness"] = (
        "FAIL" if any(r["correctness"] == "FAIL" for r in rounds) else "PASS"
    )
    obs = {a: read(base / "scan" / a / "observations.json") for a in ARMS}
    checked = {a: read(base / "scan" / a / "result.json") for a in ARMS}
    spec = CASES[case]
    for arm in ARMS:
        if not checked[arm] or not checked[arm]["matches_baseline"] or not obs[arm]:
            continue
        h = obs[arm]["head"]
        ids = checked[arm]["trial"]["requests"][spec["target"]]["tokens"]
        prefix = ids[: spec["step"] - 1]
        ref = rounds[0]["arms"]["off"][spec["target"]][1]
        data["same_history_logits"][arm] = dict(
            **h,
            same_history=prefix == ref[: len(prefix)],
            history=prefix,
            sampled=ids[spec["step"] - 1],
            tensor=f"scan/{arm}/head.pt",
        )
        if arm == "off" or not checked["off"] or not checked["off"]["matches_baseline"]:
            continue
        diffs, extra, missing = differences(obs["off"]["events"], obs[arm]["events"])
        data["comparisons"][arm] = dict(
            first=diffs[0] if diffs else None,
            differing=len(diffs),
            only_other=len(extra),
            only_off=len(missing),
        )
        if arm == "reuse":
            data["earliest_observed_difference"] = diffs[0] if diffs else None
            selected = next((d for d in diffs if d["kind"] == "linear_output"), None)
            requested = read(base / "requested.json")
            if requested:
                matches = []
                for side in ("off", "reuse"):
                    matches.append(
                        next(
                            e
                            for e in obs[side]["events"]
                            if e["kind"] == "linear_output"
                            and e["call"] == requested["call"]
                            and e["pos"] == requested["pos"]
                        )
                    )
                selected = dict(
                    call=requested["call"], pos=requested["pos"], left=matches[0], right=matches[1]
                )
            if selected:
                evidence = dict(
                    call=selected["call"],
                    pos=selected["pos"],
                    off_row=selected["left"]["row"],
                    reuse_row=selected["right"]["row"],
                    reason="requested cross-card comparison"
                    if requested
                    else "first unequal observed F.linear output in reuse execution order",
                    preceding_difference=data["earliest_observed_difference"],
                )
                evidence["sides"] = {}
                for side in ("off", "reuse"):
                    rows = [
                        e
                        for e in obs[side]["events"]
                        if e["call"] == selected["call"] and e["pos"] == selected["pos"]
                    ]
                    evidence["sides"][side] = {
                        k: next(e["digest"] for e in rows if e["kind"] == "linear_" + k)
                        for k in ("input", "output")
                    }
                write(base / "selection.json", evidence)
                data["selected"] = evidence
    if case == "total512_p256" and len(data["same_history_logits"]) == 3:
        scores = data["same_history_logits"]
        data["historical_logit_marker"] = bool(
            scores["off"]["same_history"]
            and scores["reuse"]["same_history"]
            and scores["off"]["margin_other_minus_off"] == 0
            and scores["reuse"]["margin_other_minus_off"] == 0.0625
        )
        if not data["historical_logit_marker"]:
            data["status"] = "LOGIT_MARKER_MISMATCH"
    data["probe_consistency"] = {
        f"{mode}/{a}": r["matches_baseline"]
        for mode in ("scan", "dump", "profile", "window")
        for a in ARMS
        if (r := read(base / mode / a / "result.json"))
    }
    data["evidence"] = [
        str(p.relative_to(base))
        for p in sorted(base.rglob("*"))
        if p.is_file() and p.suffix in (".pt", ".json", ".csv", ".diff")
    ]
    return data


def report(root, chip, case):
    base = root / chip / case
    summary = dict(
        chip=chip,
        case=case,
        spec=CASES[case],
        precisions={
            name: summarize(base / name, case)
            for name in ("bf16", "fp32-linear")
            if name == "bf16" or (base / name).exists()
        },
    )
    write(base / "summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chip", required=True, choices=["910B3", "910B4"])
    p.add_argument("--case", required=True, choices=CASES)
    a = p.parse_args()
    d = report(a.out, a.chip, a.case)
    print(a.out / a.chip / a.case / "summary.json", d["precisions"]["bf16"]["status"])
