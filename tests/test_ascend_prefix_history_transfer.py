"""Exercise C4 logical-history transfers at actual synchronous token boundaries."""

import copy
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from benchmarks.ascend_prefix_history_kv import FocusHistoryKV
from benchmarks.ascend_prefix_history_transfer import history_trial
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_workload import Case
from tests.helpers import tiny_ouro_model


def test_history_trial_preserves_histories_and_checks_the_native_archive():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = Case("c4_h4", "concurrency", 32, 16, concurrency=4, hits=4)
    contract = {"seed": 168, "output_tokens": 8}
    archive = {
        "runs": {arm: replay_mode(model, "torch", case, arm, 8, 168) for arm in ("off", "reuse")}
    }
    result = history_trial(model, "torch", case, contract, archive)
    summary, runs = result["summary"], result["runs"]
    assert summary["comparable"] and not summary["controls"]
    arms = {"sham1", "sham7", "prompt", "prefix", "suffix", "history", "generated", "late_prompt"}
    assert set(runs) == {"off", "reuse"} | arms
    for arm in arms:
        run = runs[arm]
        outcomes = summary["outcomes"][arm]
        assert run["prefix_hits"] == 0
        for row, cold in zip(run["requests"], runs["off"]["requests"], strict=True):
            assert row["prompt_sha256"] == cold["prompt_sha256"]
            if row["request_id"] != "target-2" or arm in ("sham1", "sham7"):
                assert row["tokens"] == cold["tokens"]
        cold_tokens = runs["off"]["requests"][2]["tokens"]
        tokens = run["requests"][2]["tokens"]
        assert outcomes["same_focus_history_at_8"] == (tokens[:7] == cold_tokens[:7])
        assert outcomes["focus_token_8"] == tokens[7]
        expected_scores = (
            run["logits"]["target-2"][7] if outcomes["same_focus_history_at_8"] else None
        )
        assert outcomes["scores_at_8"] == expected_scores
        tap = summary["boundary"][arm]
        cut = "7" if arm in ("history", "generated", "late_prompt", "sham7") else "1"
        region = {
            "generated": "generated",
            "late_prompt": "prompt",
            "prefix": "prefix",
            "suffix": "suffix",
        }.get(arm, "all")
        assert run["kv_history"]["boundaries"] == {cut: {region: tap}}
        sham = arm in ("sham1", "sham7")
        source = runs["off" if sham else "reuse"]["kv_history"]["boundaries"][cut][region]
        assert tap["source_sha256"] == tap["after_sha256"] == source["before_sha256"]
        for other, key in (("off", "off_vs_run"), ("reuse", "reuse_vs_run")):
            expected = []
            for left, right in zip(runs[other]["requests"], run["requests"], strict=True):
                assert left["request_id"] == right["request_id"]
                for index, (a, b) in enumerate(
                    zip(left["tokens"], right["tokens"], strict=True), 1
                ):
                    if a != b:
                        expected.append(
                            {"request_id": left["request_id"], "position": index, "ids": [a, b]}
                        )
                        break
            assert outcomes[key] == expected

    stale = copy.deepcopy(archive)
    stale["runs"]["reuse"]["requests"][2]["tokens"][7] += 1
    rejected = history_trial(model, "torch", case, contract, stale)["summary"]
    assert not rejected["comparable"] and rejected["controls"]


@pytest.mark.parametrize(
    ("after", "region", "sham"), [(1, "suffix", False), (7, "generated", False), (7, "all", True)]
)
@torch.inference_mode()
def test_history_transfer_changes_only_selected_rows_of_focus_private_pages(after, region, sham):
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = Case("c4_h4", "concurrency", 32, 16, concurrency=4, hits=4)
    ids = [f"target-{i}" for i in range(4)]
    focus = ids[2]
    total = case.prefix + case.suffix
    captures = {}

    def capture(engine, targets, donor):
        probe = FocusHistoryKV(engine, ids, focus, total, case.prefix, capture=True)
        captures["hot"] = probe
        return probe

    replay_mode(model, "torch", case, "reuse", 8, 168, target_context=capture)
    source = captures["hot"].snapshots[after]
    # Make every source row distinguishable even on the tiny CPU model, on which
    # the native cold and hot KV can be identical. The last page at token 7 has
    # only six written rows; the ten remaining physical rows must be preserved.
    for depth, plane in enumerate(source):
        for page, (k, v) in enumerate(plane):
            for offset in range(k.shape[1]):
                k[:, offset].fill_(16 * page + offset + depth + 1)
                v[:, offset].fill_(-16 * page - offset - depth - 1)

    start, end = {
        "suffix": (case.prefix, total),
        "generated": (total, total + 6),
        "all": (0, total + 6),
    }[region]
    states = []

    def is_boundary(outputs):
        return any(o.request_id == focus and len(o.token_ids) == after for o in outputs)

    @contextmanager
    def transfer(engine, targets, donor):
        cache = engine.cache_manager
        original = engine.step

        def before_transfer():
            outputs = original()
            if is_boundary(outputs):
                expected = [t.clone() for t in (cache.key_cache, cache.value_cache)]
                if not sham:
                    for depth, plane in enumerate(cache.plane_block_tables(focus)):
                        for pos in range(start, end):
                            page, offset = divmod(pos, cache.block_size)
                            for pool, src in zip(expected, source[depth][page], strict=True):
                                pool[plane[page], :, offset].copy_(src[:, offset])
                states.append(expected)
            return outputs

        probe = FocusHistoryKV(
            engine,
            ids,
            focus,
            total,
            case.prefix,
            after=after,
            region=region,
            source=None if sham else source,
        )
        captures["transfer"] = probe
        with patch.object(engine, "step", before_transfer), probe:
            controlled = engine.step

            def after_transfer():
                outputs = controlled()
                if is_boundary(outputs):
                    (expected,) = states
                    for pool, wanted in zip(
                        (cache.key_cache, cache.value_cache), expected, strict=True
                    ):
                        assert torch.equal(pool.view(torch.int16), wanted.view(torch.int16))
                return outputs

            with patch.object(engine, "step", after_transfer):
                yield
        assert engine.step == original

    replay_mode(model, "torch", case, "off", 8, 168, target_context=transfer)
    assert len(states) == 1
    report = captures["transfer"].report()["boundaries"][str(after)][region]
    assert (report["start"], report["end"]) == (start, end)
    assert report["after_sha256"] == report["source_sha256"]
    assert report["changed_pages"] == (0 if sham else report["checked_pages"])
    assert report["checked_pages"] > 0
