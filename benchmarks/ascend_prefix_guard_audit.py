"""Audit a first-near-tie guard using saved CODA outputs; no model or device required."""

import math


def _native_events(run, arm, count):
    """Recover the decision order from actual sampling steps, including staggered bursts."""
    rows = {row["request_id"]: row for row in run["requests"]}
    if not rows or len(rows) != len(run["requests"]) or set(run["logits"]) != set(rows):
        raise ValueError(f"{arm}:incomplete_or_duplicate_targets")
    if run["prefix_hits"] != sum(r["hit_tokens"] for r in rows.values()) or run[
        "prefix_queries"
    ] != len(rows):
        raise ValueError(f"{arm}:cache_counters")
    if run.get("kv", {}).get("ok") is False:
        raise ValueError(f"{arm}:kv_witness")
    for rid, row in rows.items():
        hit = row["hit_tokens"]
        if (
            not row["prompt_sha256"]
            or hit < 0
            or (arm == "off" and hit != 0)
            or not hit == row["expected_hit_tokens"] == row["prefill_start"]
            or row["finish_reason"] != "length"
            or len(row["tokens"]) != count
            or row["exit_depths"] != [4] * count
            or len(run["logits"][rid]) != count
        ):
            raise ValueError(f"{arm}:{rid}:invalid_native_request")

    events = []
    emitted = dict.fromkeys(rows, 0)
    for step, batch in enumerate(run["steps"], 1):
        visible = batch["visible"]
        targets = [span["request_id"] for span in batch["spans"] if span["phase"] == "coda"]
        if (
            step != batch["step"]
            or targets != [out["request_id"] for out in visible]
            or len(set(targets)) != len(targets)
        ):
            raise ValueError(f"{arm}:CODA_order_at_step_{step}")
        for out in visible:
            rid = out["request_id"]
            index = emitted[rid]
            point = run["logits"][rid][index]
            chosen = rows[rid]["tokens"][index]
            scores = point["top_logits"]
            if (
                point["position"] != index + 1
                or out["token_ids"] != [chosen]
                or len(point["top_ids"]) != 2
                or len(set(point["top_ids"])) != 2
                or point["top_ids"][0] != chosen
                or point["selected_id"] != chosen
                or len(scores) != 2
                or not all(math.isfinite(v) for v in scores)
                or scores[0] < scores[1]
                or point["margin"] != scores[0] - scores[1]
            ):
                raise ValueError(f"{arm}:{rid}:invalid_sample_at_{index + 1}")
            events.append(
                dict(
                    request_id=rid,
                    position=index + 1,
                    step=step,
                    margin=point["margin"],
                    hit_tokens=rows[rid]["hit_tokens"],
                )
            )
            emitted[rid] += 1
    if any(n != count for n in emitted.values()):
        raise ValueError(f"{arm}:incomplete_CODA_steps")
    return rows, events


def _inspect(replay, count, thresholds):
    comparison = replay["comparison"]
    issues = [issue for issue in comparison["primary_issues"] if issue != "off_reuse:tokens"]
    issues.extend(i for i in comparison["diagnostic_issues"] if i.startswith(("off:", "reuse:")))
    if comparison["status"] not in ("PASS", "FAIL") or not any(
        comparison["joint_archive_outputs_on_same_inputs_1based"].values()
    ):
        issues.append("original_matrix_replay_unconfirmed")
    if issues:
        raise ValueError(", ".join(issues))
    off, _ = _native_events(replay["runs"]["off"], "off", count)
    hot, events = _native_events(replay["runs"]["reuse"], "reuse", count)
    if list(off) != list(hot) or any(
        off[rid]["prompt_sha256"] != hot[rid]["prompt_sha256"] for rid in hot
    ):
        raise ValueError("off_reuse:target_order_or_identity")
    first = None
    for event in events:
        rid, index = event["request_id"], event["position"] - 1
        cold_id, hot_id = off[rid]["tokens"][index], hot[rid]["tokens"][index]
        if cold_id != hot_id:
            first = dict(
                request_id=rid,
                position=index + 1,
                step=event["step"],
                off_id=cold_id,
                hot_id=hot_id,
                hot_margin=event["margin"],
                hit_tokens=event["hit_tokens"],
            )
            break
    trials = []
    for threshold in thresholds:
        trigger = next((e for e in events if e["hit_tokens"] and e["margin"] <= threshold), None)
        if first is None:
            status = "unnecessary" if trigger else "unneeded"
        elif not trigger:
            status = "missed"
        elif trigger["step"] == first["step"]:
            status = "on"
        elif trigger["step"] < first["step"]:
            status = "early"
        else:
            status = "late"
        trials.append(dict(threshold=threshold, status=status, trigger=trigger))
    return first, trials


def audit_diagnosis(diag, thresholds):
    """Report the observable guard decision for every recorded burst and threshold."""
    thresholds = tuple(thresholds)
    if not thresholds or any(not math.isfinite(v) or v < 0 for v in thresholds):
        raise ValueError("expected finite, nonnegative guard thresholds")
    if diag.get("state") != "complete" or diag["contract"]["output_tokens"] < 1:
        raise ValueError("expected a completed diagnostic report with generated tokens")
    cases = {}
    for name, replays in diag["cases"].items():
        rows = cases[name] = []
        for n, replay in enumerate(replays, 1):
            controls = []
            try:
                first, trials = _inspect(replay, diag["contract"]["output_tokens"], thresholds)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                controls.append(str(exc))
                first, trials = None, []
            rows.append(
                dict(
                    replay=n,
                    comparable=not controls,
                    controls=controls,
                    first_diff=first,
                    thresholds=trials,
                )
            )
    return cases
