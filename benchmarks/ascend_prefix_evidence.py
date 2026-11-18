"""Compare archived prefix-cache trials with untimed greedy replays."""

from collections import Counter

from benchmarks.ascend_prefix_scenarios import ARMS, _mismatches

PHASES = ("warmups", "trials")


def archive_evidence(entry, case):
    """Recheck every complete archived trial, including discarded warmups."""
    phases = {}
    for phase in PHASES:
        trials = entry.get(phase, [])
        counts = Counter()
        failed, incomplete, examples = [], [], []
        for index, trial in enumerate(trials, 1):
            if not all(arm in trial for arm in ARMS):
                incomplete.append(index)
                continue
            issues = list(_mismatches(case, [trial]))
            if issues:
                failed.append(index)
                counts.update(issue["kind"] for issue in issues)
                examples.extend(dict(issue, trial=index) for issue in issues[: 8 - len(examples)])
        phases[phase] = {
            "iterations": len(trials),
            "failed_trials_1based": failed,
            "incomplete_trials_1based": incomplete,
            "counts_by_kind": dict(counts),
            "mismatch_count": sum(counts.values()),
            "examples": examples,
        }
    status = "PASS"
    if any(phase["mismatch_count"] for phase in phases.values()):
        status = "FAIL"
    elif not phases["trials"]["iterations"] or any(
        phase["incomplete_trials_1based"] for phase in phases.values()
    ):
        status = "INCOMPLETE"
    return {"status": status, "phases": phases, "last_progress": entry.get("progress")}


def _first_diff(left, right):
    for position, (a, b) in enumerate(zip(left, right), start=1):
        if a != b:
            return position
    return None if len(left) == len(right) else min(len(left), len(right)) + 1


def _pairs(runs, left, right):
    a, b = runs[left]["requests"], runs[right]["requests"]
    changed = []
    candidates = []
    for index in range(max(len(a), len(b))):
        if index >= min(len(a), len(b)):
            changed.append(index)
            continue
        pos = _first_diff(a[index]["tokens"], b[index]["tokens"])
        if pos is not None:
            changed.append(index)
            candidates.append((pos, index))
    first = None
    if candidates:
        pos, index = min(candidates)
        tokens, top2 = {}, {}
        for arm in (left, right):
            row = runs[arm]["requests"][index]
            ids = row["tokens"]
            tokens[arm] = ids[pos - 1] if pos <= len(ids) else None
            logits = runs[arm].get("logits", {}).get(row["request_id"], [])
            top2[arm] = next((r for r in logits if r["position"] == pos), None)
        first = dict(target=index, position_1based=pos, token_ids=tokens, top2=top2)
    return {"equal": not changed, "changed_targets": changed, "first_diff": first}


def _arm_checks(run):
    rows = run["requests"]
    hits = [row["hit_tokens"] for row in rows]
    starts = [row["prefill_start"] for row in rows]
    issues = []
    probe_issues = []
    if run["prefix_hits"] != sum(hits) or run["prefix_queries"] != len(rows):
        issues.append("allocation_counters")
    for target, row in enumerate(rows):
        expected = row["expected_hit_tokens"]
        if row["hit_tokens"] != expected or row["prefill_start"] != expected:
            issues.append(f"target_{target}_hit_or_offset")
        tokens = row["tokens"]
        if (
            row.get("finish_reason") != "length"
            or not tokens
            or len(tokens) != run.get("output_tokens_per_request", len(tokens))
            or row["exit_depths"] != [4] * len(tokens)
        ):
            issues.append(f"target_{target}_completion")
        logits = run.get("logits", {}).get(row["request_id"], [])
        if "logits" in run and [(r["position"], r["selected_id"]) for r in logits] != list(
            enumerate(tokens, 1)
        ):
            probe_issues.append(f"target_{target}_logits_coverage")
    kv = run.get("kv", {})
    if kv.get("ok") is False:
        issues.append("kv_witness")
    return dict(
        hits=hits,
        prefill_starts=starts,
        prefix_queries=run["prefix_queries"],
        prefix_hits=run["prefix_hits"],
        num_blocks=run["num_blocks"],
        step_count=len(run["steps"]),
        kv_ok=kv.get("ok"),
        checked_kv_pages=kv.get("checked_kv_pages"),
        issues=issues,
        probe_issues=probe_issues,
    )


def _archive_matches(requests, entry, reference_arm, compare_outputs):
    """Count matching observations for each target and each whole burst, phase by phase."""
    targets = [
        {"target": i, "known_inputs": 0, "same_inputs": 0, "same_outputs_on_same_input": 0}
        for i in range(len(requests))
    ]
    variants = [set() for _ in requests]
    phases = {}
    for phase in PHASES:
        same_inputs, same_outputs = [], []
        for index, trial in enumerate(entry.get(phase, []), 1):
            old = trial.get(reference_arm, {}).get("requests", [])
            input_matches = []
            output_matches = []
            for target, (current, previous) in enumerate(zip(requests, old)):
                old_hash = previous.get("prompt_sha256")
                same_input = bool(old_hash and old_hash == current.get("prompt_sha256"))
                input_matches.append(same_input)
                record = targets[target]
                record["known_inputs"] += bool(old_hash)
                record["same_inputs"] += same_input
                if same_input and compare_outputs and "tokens" in previous:
                    old_ids = tuple(previous["tokens"])
                    variants[target].add(old_ids)
                    equal = old_ids == tuple(current["tokens"])
                    record["same_outputs_on_same_input"] += equal
                    output_matches.append(equal)
            if len(old) == len(requests) and requests and all(input_matches):
                same_inputs.append(index)
                if len(output_matches) == len(requests) and all(output_matches):
                    same_outputs.append(index)
        phases[phase] = {"same_inputs_1based": same_inputs}
        if compare_outputs:
            phases[phase]["same_outputs_on_same_inputs_1based"] = same_outputs
    for target, observed in zip(targets, variants, strict=True):
        if compare_outputs:
            target["archived_output_variants_on_same_input"] = len(observed)
        else:
            del target["same_outputs_on_same_input"]
    return {"reference_arm": reference_arm, "targets": targets, "phases": phases}


def compare_replays(runs, entry):
    """Keep production parity, the uncached split control, and archive alignment distinct."""
    pairs = {}
    primary, diagnostic = [], []
    for left, right in (("off", "miss"), ("off", "reuse"), ("split", "reuse"), ("off", "split")):
        key = f"{left}_{right}"
        pairs[key] = _pairs(runs, left, right)
        if not pairs[key]["equal"]:
            (diagnostic if "split" in (left, right) else primary).append(f"{key}:tokens")
    alignment, checks = {}, {}
    off_targets = [(r["request_id"], r.get("prompt_sha256")) for r in runs["off"]["requests"]]
    for arm in (*ARMS, "split"):
        run = runs[arm]
        checks[arm] = _arm_checks(run)
        issues = diagnostic if arm == "split" else primary
        issues.extend(f"{arm}:{issue}" for issue in checks[arm]["issues"])
        diagnostic.extend(f"{arm}:{issue}" for issue in checks[arm]["probe_issues"])
        targets = [(r["request_id"], r.get("prompt_sha256")) for r in run["requests"]]
        if targets != off_targets or any(not digest for _, digest in targets):
            issues.append(f"{arm}:target_identity")
        reference = "off" if arm == "split" else arm
        alignment[arm] = _archive_matches(run["requests"], entry, reference, arm != "split")

    # An independently matching off/miss/reuse draw need not reproduce the same archived trial.
    joint_inputs, joint_outputs = {}, {}
    for phase in PHASES:
        for result, field in (
            (joint_inputs, "same_inputs_1based"),
            (joint_outputs, "same_outputs_on_same_inputs_1based"),
        ):
            matches = [set(alignment[arm]["phases"][phase][field]) for arm in ARMS]
            result[phase] = sorted(set.intersection(*matches))
    status = "FAIL" if primary else "PASS"
    if not any(joint_inputs.values()):
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "primary_issues": primary,
        "diagnostic_issues": diagnostic,
        "pairs": pairs,
        "arm_checks": checks,
        "archive_matches": alignment,
        "joint_archive_inputs_1based": joint_inputs,
        "joint_archive_outputs_on_same_inputs_1based": joint_outputs,
    }
