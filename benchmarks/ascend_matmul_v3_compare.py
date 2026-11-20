"""Compare two complete matrix reports, including cases with a native cache parity failure."""

import argparse
import json
from pathlib import Path

from benchmarks.ascend_prefix_scenarios import ARMS, suites

FOCUS = ("p1024_s8", "total512_p384", "c4_h4")


RUNTIME_KEYS = ("cann_version", "torch_version", "torch_npu_version")


def compare(native, candidate, names, route="bridge"):
    contract = candidate["contract"].copy()
    original = native["contract"].copy()
    projection = "native-dispatch-audit" if route == "public" else "MatMulV3"
    expected_override = "1" if route == "diagnostic" else None
    if contract.pop("ignore_infer_error", None) != expected_override:
        raise ValueError("unexpected GE inference override in the candidate matrix")
    if route == "public":
        for key in RUNTIME_KEYS:
            contract.pop(key, None)
            original.pop(key, None)
    if contract.pop("kv_projection", None) != projection or contract != original:
        raise ValueError("incompatible matrix contracts or missing V3 intervention")
    if native["state"] != "complete" or not set(names).issubset(native["cases"]):
        raise ValueError("native baseline does not cover the requested cases")
    if candidate["state"] != "complete" or set(candidate["cases"]) != set(names):
        raise ValueError("incomplete candidate suite or unexpected set of cases")

    cases = {}
    for name in names:
        before, after = (r["cases"][name] for r in (native, candidate))
        if before["summary"]["case"] != after["summary"]["case"]:
            raise ValueError(f"{name}: case definition changed")
        counts = after["v3_calls"]
        if not counts.get("k_proj") or counts["k_proj"] != counts.get("v_proj"):
            raise ValueError(f"{name}: missing V3 K/V dispatches")
        changes = dict.fromkeys(ARMS, 0)
        first = {}
        for bucket in ("warmups", "trials"):
            pairs = zip(before[bucket], after[bucket], strict=True)
            for repeat, (old, new) in enumerate(pairs, 1):
                for arm in ARMS:
                    targets = zip(old["off"]["requests"], new[arm]["requests"], strict=True)
                    for target, (ref, actual) in enumerate(targets):
                        if ref["prompt_sha256"] != actual["prompt_sha256"]:
                            raise ValueError(f"{name}: different prompt at {target}")
                        a, b = ref["tokens"], actual["tokens"]
                        if len(a) != contract["output_tokens"] or len(b) != len(a):
                            raise ValueError(f"{name}: incomplete token sequence")
                        if a != b:
                            changes[arm] += 1
                            if arm not in first:
                                i = next(i for i in range(len(a)) if a[i] != b[i])
                                first[arm] = dict(
                                    bucket=bucket,
                                    repeat=repeat,
                                    target=target,
                                    token_1based=i + 1,
                                    native=a[i],
                                    v3=b[i],
                                )
        delta = {}
        for arm in ARMS:
            previous, current = (e["summary"]["arms"][arm] for e in (before, after))
            delta[arm] = {
                k: round(current[k] - previous[k], 3) for k in ("ttft_p50_ms", "ttft_p95_ms")
            }
        cases[name] = dict(
            native_parity=before["summary"]["status"],
            v3_parity=after["summary"]["status"],
            v3_vs_native_off=changes,
            first_difference=first,
            v3_calls=counts,
            v3_minus_native_ms=delta,
        )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("stage", choices=("probe", "failures", "full"))
    parser.add_argument("--route", choices=("public", "bridge", "diagnostic"), default="bridge")
    args = parser.parse_args()

    def read(name):
        return json.loads((args.directory / name).read_text())

    numerical = read("v3-probe.json")
    if numerical["ignore_infer_error"] != ("1" if args.route == "diagnostic" else None):
        raise ValueError("probe GE environment disagrees with requested route")
    if numerical["impl"] != ("native" if args.route == "public" else "v3"):
        raise ValueError("probe projection disagrees with requested route")
    arms = ("v3",) if args.route == "public" else ("native", "v3")
    result = {
        "route": args.route,
        "ignore_infer_error": numerical["ignore_infer_error"],
        "probe_invariant": numerical["invariant"],
        "probe_ch18_matches_fp64": all(
            v["ch18_different_from_rounded_fp64"] == 0 for v in numerical["rows"].values()
        ),
        "dispatch": {arm: read(f"{arm}-dispatch.json")["types"] for arm in arms},
    }
    if args.stage != "probe":
        result["matrix_shape_replay_dispatch"] = read("v3-matrix-dispatch.json")["types"]
        names = FOCUS if args.stage == "failures" else [c.name for c in suites()["full"]]
        native, candidate = read("native-matrix.json"), read("v3-matrix.json")
        cases = compare(native, candidate, names, args.route)
        result["runtime_versions"] = {
            arm: {k: report["contract"].get(k) for k in RUNTIME_KEYS}
            for arm, report in (("baseline", native), ("candidate", candidate))
        }
        result.update(
            cases=cases,
            v3_parity_failed=[n for n in names if cases[n]["v3_parity"] != "PASS"],
            native_cold_changed=[n for n in names if cases[n]["v3_vs_native_off"]["off"]],
        )
    with (args.directory / "comparison.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print("V3_REVIEW", json.dumps({k: v for k, v in result.items() if k != "cases"}))
    return int(
        not result["probe_invariant"]
        or bool(result.get("v3_parity_failed"))
        or bool(result.get("native_cold_changed"))
    )


if __name__ == "__main__":
    raise SystemExit(main())
