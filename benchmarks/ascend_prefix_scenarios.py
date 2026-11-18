"""Workloads and paired comparisons for the synchronous Ouro prefix-cache benchmark."""

from dataclasses import asdict

from benchmarks.ascend_prefix_workload import Case
from benchmarks.preemption import percentile

ARMS = ("off", "miss", "reuse")


def suites():
    lengths = [Case(f"p{p}_s8", "prefix_length", p, 8) for p in (128, 256, 384, 512, 1024)]
    boundaries = [Case(f"edge_p{p}", "page_boundary", p, 8) for p in (0, 15, 16, 17, 127, 129)]
    exact = Case("repeat_p128", "whole_prompt", 128, 0)
    ratios = [Case(f"total512_p{p}", "fixed_total", p, 512 - p) for p in (0, 128, 256, 384, 512)]
    suffixes = [Case(f"p128_s{s}", "suffix_length", 128, s) for s in (1, 32, 128)]
    bursts = [
        Case(f"c{n}_h{hits}", "concurrency", 128, 16, concurrency=n, hits=hits)
        for n, hits in ((1, 1), (2, 1), (2, 2), (4, 0), (4, 1), (4, 2), (4, 4))
    ]
    # The churn case adds two unrelated waves to a capacity-constrained pool.
    resident = Case("resident_p64", "retention", 64, 8, tight_pool=True)
    evicted = Case("evicted_p64", "retention", 64, 8, churn=2, tight_pool=True)
    full = [*lengths, *boundaries, exact, *ratios, *suffixes, *bursts, resident, evicted]
    quick = [lengths[0], lengths[3], exact, bursts[1], evicted]
    return {"quick": quick, "full": full}


def _latencies(rows):
    result = {}
    for metric in ("ttft", "e2e"):
        values = [row[f"{metric}_ms"] for row in rows]
        result[f"{metric}_p50_ms"] = percentile(values, 0.5)
        result[f"{metric}_p95_ms"] = percentile(values, 0.95)
    tpot = [
        (row["e2e_ms"] - row["ttft_ms"]) / (len(row["tokens"]) - 1)
        for row in rows
        if len(row["tokens"]) > 1
    ]
    result["tpot_p50_ms"] = percentile(tpot, 0.5) if tpot else None
    result["tpot_p95_ms"] = percentile(tpot, 0.95) if tpot else None
    return result


def _arm_metrics(case, arm, trials):
    runs = [trial[arm] for trial in trials]
    rows = [row for run in runs for row in run["requests"]]
    total_ms = sum(run["wall_ms"] for run in runs)
    hit_tokens = sum(row["hit_tokens"] for row in rows)
    prompt_tokens = len(rows) * (case.prefix + case.suffix)
    output_tokens = sum(run["output_tokens"] for run in runs)
    result = {
        "request_count": len(rows),
        "hit_requests": sum(row["hit_tokens"] > 0 for row in rows),
        "hit_tokens": hit_tokens,
        "prompt_tokens": prompt_tokens,
        "hit_token_pct": 100 * hit_tokens / prompt_tokens,
        "output_tokens": output_tokens,
        "wall_ms": total_ms,
        "requests_per_s": 1000 * len(rows) / total_ms,
        "output_tok_s": 1000 * output_tokens / total_ms,
        **_latencies(rows),
    }
    for key in ("prefix_hits", "prefix_queries", "churn_requests"):
        if key in runs[0]:
            result[key] = sum(run[key] for run in runs)
    for key in ("peak_used_blocks", "num_blocks"):
        if key in runs[0]:
            result[key] = max(run[key] for run in runs)
    if runs[0].get("primed_hit_tokens") is not None:
        result["pre_churn_hit_tokens_by_trial"] = [run["primed_hit_tokens"] for run in runs]

    if case.hits is not None and 0 < case.hits < case.concurrency:
        # Keep the same target cohort on all arms, including when cache is disabled.
        for cohort, targets in (
            ("shared_prefix_targets", slice(0, case.hits)),
            ("unrelated_targets", slice(case.hits, None)),
        ):
            cohort_rows = [row for run in runs for row in run["requests"][targets]]
            result[cohort] = {"request_count": len(cohort_rows), **_latencies(cohort_rows)}
    return result


def _mismatches(case, trials):
    for iteration, trial in enumerate(trials, start=1):
        for arm in ARMS:
            run = trial[arm]
            location = {"trial": iteration, "arm": arm}
            requests = run["requests"]
            if len(requests) != case.concurrency:
                yield dict(
                    location, kind="request_count", expected=case.concurrency, actual=len(requests)
                )
            observed_sum = sum(row["hit_tokens"] for row in requests)
            if "prefix_queries" in run and run["prefix_queries"] != len(requests):
                yield dict(
                    location,
                    kind="query_counter",
                    expected=len(requests),
                    actual=run["prefix_queries"],
                )
            if "prefix_hits" in run and run["prefix_hits"] != observed_sum:
                yield dict(
                    location, kind="hit_counter", expected=observed_sum, actual=run["prefix_hits"]
                )
            if run.get("primed_hit_tokens") != run.get("expected_primed_hit_tokens"):
                yield dict(
                    location,
                    kind="pre_churn_lookup",
                    expected=run.get("expected_primed_hit_tokens"),
                    actual=run.get("primed_hit_tokens"),
                )
            for index, row in enumerate(requests):
                expected = row["expected_hit_tokens"]
                if row["hit_tokens"] != expected or row["prefill_start"] != expected:
                    yield dict(
                        location,
                        kind="hit_or_offset",
                        target=index,
                        expected=expected,
                        hit_tokens=row["hit_tokens"],
                        prefill_start=row["prefill_start"],
                    )
                if arm == "off" or index >= len(trial["off"]["requests"]):
                    continue
                off_request = trial["off"]["requests"][index]
                if row.get("prompt_sha256") != off_request.get("prompt_sha256"):
                    yield dict(location, kind="prompt_vs_off", target=index)
                reference = off_request["tokens"]
                tokens = row["tokens"]
                if tokens != reference:
                    first = next(
                        (i for i, (a, b) in enumerate(zip(reference, tokens), 1) if a != b),
                        min(len(tokens), len(reference)) + 1,
                    )
                    yield dict(
                        location,
                        kind="output_vs_off",
                        target=index,
                        first_diff_1based=first,
                        expected=reference[first - 1] if first <= len(reference) else None,
                        actual=tokens[first - 1] if first <= len(tokens) else None,
                    )


def metric_deltas(baseline, candidate):
    """Latency saved and throughput gained against one measured baseline."""
    result = {}
    for metric in ("ttft_p50", "ttft_p95", "e2e_p50", "e2e_p95"):
        base_ms = baseline[f"{metric}_ms"]
        saved = base_ms - candidate[f"{metric}_ms"]
        result[f"{metric}_saved_ms"] = saved
        result[f"{metric}_saved_pct"] = 100 * saved / base_ms if base_ms else None
    throughput_gain = candidate["output_tok_s"] - baseline["output_tok_s"]
    result["output_tok_s_gain"] = throughput_gain
    result["output_tok_s_gain_pct"] = (
        100 * throughput_gain / baseline["output_tok_s"] if baseline["output_tok_s"] else None
    )
    return result


def _comparison(arms, trials, base_arm, candidate_arm):
    result = metric_deltas(arms[base_arm], arms[candidate_arm])
    # One paired median per burst preserves run-to-run spread under rotated arm order.
    paired_savings = []
    for trial in trials:
        base_ms = percentile([r["ttft_ms"] for r in trial[base_arm]["requests"]], 0.5)
        candidate_ms = percentile([r["ttft_ms"] for r in trial[candidate_arm]["requests"]], 0.5)
        paired_savings.append(base_ms - candidate_ms)
    result["paired_trial_ttft_p50_saved_ms"] = paired_savings
    return result


def summarize(case, trials):
    """Compare measured bursts. Positive saved latency or gained throughput favors the candidate."""
    if not trials:
        raise ValueError("at least one measured trial is required")
    arms = {arm: _arm_metrics(case, arm, trials) for arm in ARMS}
    examples = []
    count = 0
    for mismatch in _mismatches(case, trials):
        count += 1
        if len(examples) < 8:
            examples.append(mismatch)
    comparisons = {}
    if not count:
        comparisons = {
            f"{candidate}_vs_{base}": _comparison(arms, trials, base, candidate)
            for candidate, base in (("reuse", "off"), ("reuse", "miss"), ("miss", "off"))
        }
    return {
        "case": asdict(case),
        "repeats": len(trials),
        "status": "FAIL" if count else "PASS",
        "arms": arms,
        "comparisons": comparisons,
        "mismatch_count": count,
        "mismatches": examples,
    }
