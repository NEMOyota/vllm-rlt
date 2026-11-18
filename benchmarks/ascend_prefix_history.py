"""Compare one scenario with a saved run of the same deterministic workload."""

import math

from benchmarks.ascend_prefix_scenarios import ARMS, metric_deltas


def _check_replays(source, summary, trials, reference, expected_length):
    """Check every target and every arm against the archived cache-off outputs."""
    count = summary["case"]["concurrency"]
    if not trials or len(trials) != summary["repeats"] or len(reference) != count:
        raise ValueError(f"incomplete {source} measurements")
    for repeat, trial in enumerate(trials, 1):
        for arm in ARMS:
            requests = trial[arm]["requests"]
            if len(requests) != count:
                raise ValueError(f"incomplete {source} request list")
            for index, (row, ref) in enumerate(zip(requests, reference, strict=True)):
                location = dict(source=source, repeat=repeat, arm=arm, target=index)
                if not ref["prompt_sha256"] or not row["prompt_sha256"]:
                    raise ValueError(f"missing {source} prompt digest")
                if row["prompt_sha256"] != ref["prompt_sha256"]:
                    return dict(status="INCOMPATIBLE", reason="target_prompt_changed", **location)
                expected, actual = ref["tokens"], row["tokens"]
                if not expected or not isinstance(expected, list) or not isinstance(actual, list):
                    raise ValueError(f"missing {source} token IDs")
                if expected_length is not None and len(expected) != expected_length:
                    raise ValueError("archived output length disagrees with contract")
                if expected != actual:
                    first = next(
                        (i for i, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]),
                        min(len(expected), len(actual)),
                    )
                    return dict(
                        status="FAIL",
                        reason="output_changed",
                        first_diff_1based=first + 1,
                        expected=expected[first] if first < len(expected) else None,
                        actual=actual[first] if first < len(actual) else None,
                        **location,
                    )
    return None


def compare_baseline(current_contract, previous_report, summary, trials):
    """Compare pooled arm metrics only for complete, correct runs of identical target prompts.

    Commit, file paths and repeat counts belong in report metadata. Differences in the measured
    model, execution contract or scenario invalidate the historical comparison.
    """
    if previous_report is None:
        return {"status": "NO_BASELINE", "reason": "no_report"}
    if not isinstance(previous_report, dict) or previous_report.get("schema") != 1:
        return {"status": "INCOMPATIBLE", "reason": "unsupported_schema"}
    try:
        prior_contract = previous_report["contract"]
        if prior_contract != current_contract:
            keys = sorted(current_contract.keys() | prior_contract.keys())
            missing = object()
            changed = [
                key
                for key in keys
                if prior_contract.get(key, missing) != current_contract.get(key, missing)
            ]
            return {"status": "INCOMPATIBLE", "reason": "contract_changed", "fields": changed}
        name = summary["case"]["name"]
        cases = previous_report["cases"]
        if name not in cases:
            return {"status": "NO_BASELINE", "reason": "case_absent"}
        archived = cases[name]
        old = archived["summary"]
        if old["case"] != summary["case"]:
            return {"status": "INCOMPATIBLE", "reason": "case_changed"}
        for source, result in (("archived", old), ("current", summary)):
            if result["status"] != "PASS":
                return {"status": "FAIL", "reason": "case_not_passed", "source": source}
        previous_trials = archived["trials"]
        reference = previous_trials[0]["off"]["requests"]
        for source, result, runs in (
            ("archived", old, previous_trials),
            ("current", summary, trials),
        ):
            failure = _check_replays(
                source, result, runs, reference, current_contract.get("output_tokens")
            )
            if failure:
                return failure

        arms = {arm: metric_deltas(old["arms"][arm], summary["arms"][arm]) for arm in ARMS}
        if any(
            value is None or not math.isfinite(value)
            for arm in arms.values()
            for value in arm.values()
        ):
            raise ValueError("invalid baseline metrics")
        return {"status": "PASS", "baseline_repeats": len(previous_trials), "arms": arms}
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        return {"status": "INCOMPATIBLE", "reason": f"incomplete or invalid report: {exc}"}
