"""Host-side ordinary prefill observations, enabled only for separate diagnostic replays."""

from collections import Counter
from contextlib import contextmanager
from unittest.mock import patch

from benchmarks.ascend_main_prefix_results import ids_digest
from benchmarks.ascend_prefix_combo_checks import require
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.request import RequestPhase


@contextmanager
def observe_prefill(steps):
    schedule = StageEngine._schedule

    def observed(engine, *args, **kwargs):
        batch = schedule(engine, *args, **kwargs)
        if batch is not None and batch.phase == RequestPhase.PREFILL:
            members = []
            for item in batch.requests:
                length = len(item.request.prompt_token_ids)
                members.append(
                    dict(
                        prompt_sha256=ids_digest(item.request.prompt_token_ids),
                        start=item.token_start,
                        query_rows=item.token_count,
                        remaining=length - item.token_start,
                        kv_length=item.token_start + item.token_count,
                    )
                )
            steps.append(dict(depths=list(range(batch.loop_count)), members=members))
        return batch

    with patch.object(StageEngine, "_schedule", observed):
        yield


def opportunities(steps, plan, tolerances=(0, 2, 4, 8)):
    targets = {r["prompt_sha256"] for r in plan}
    seen = set()
    widths = Counter()
    with_partner = dict.fromkeys(tolerances, 0)
    total = regular = 0
    for step in steps:
        members = [m for m in step["members"] if m["prompt_sha256"] in targets]
        if not members:
            continue
        require(len(members) == len(step["members"]), "primer overlapped measured arrivals")
        require(step["depths"] == [0, 1, 2, 3], "expected synchronous four-loop prefill")
        seen.update(m["prompt_sha256"] for m in members)
        total += len(members)
        widths[len(members)] += 1
        if (
            len(members) >= 2
            and members[0]["query_rows"] > 1
            and all(
                m["start"] == 0 and m["query_rows"] == members[0]["query_rows"] for m in members
            )
        ):
            regular += len(members)
            continue  # Already handled by the mainline complete cold-grid path.
        for i, x in enumerate(members):
            if x["query_rows"] <= 1:
                continue
            distances = [
                abs(x["query_rows"] - y["query_rows"])
                for j, y in enumerate(members)
                if i != j and y["query_rows"] > 1 and y["start"] == x["start"]
            ]
            if distances:
                for slack in tolerances:
                    with_partner[slack] += min(distances) <= slack
    require(seen == targets, "prefill records missing measured targets")
    return dict(
        unique_targets=len(seen),
        request_step_observations=total,
        steps_by_size=dict(widths),
        already_regular_cold_members=regular,
        additional_members_with_partner=with_partner,
    )
