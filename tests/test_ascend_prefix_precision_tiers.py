import io
import json
from collections import Counter
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from benchmarks import ascend_prefix_precision_tiers as bench
from benchmarks.ascend_prefix_projection import fp32_decoder_projections
from benchmarks.ascend_prefix_scenarios import ARMS
from benchmarks.ascend_prefix_workload import expected_hits
from tests.helpers import tiny_ouro_model


@pytest.mark.parametrize(
    "tier, attention, mlp",
    [(2, "qkv", ()), (3, "qkvo", ()), (4, "qkvo", ("gate", "up", "down"))],
)
@torch.inference_mode()
def test_tiers_patch_exactly_the_selected_decoder_linears(tier, attention, mlp):
    model = tiny_ouro_model(dtype=torch.bfloat16)
    modules = dict(model.named_modules())
    paths = [f"self_attn.{letter}_proj" for letter in attention] + [
        f"mlp.{name}_proj" for name in mlp
    ]
    expected = {f"model.layers.{i}.{p}" for i in range(len(model.model.layers)) for p in paths}
    with fp32_decoder_projections(model, bench.TIER_PATHS[tier], chunk_rows=4) as work:
        assert {name for name, m in modules.items() if "forward" in vars(m)} == expected
        for name in expected:
            module = modules[name]
            x = torch.zeros(5, module.in_features, dtype=torch.bfloat16)
            x[:, 0] = torch.arange(1, 6)
            reference = F.linear(x.float(), module.weight.float()).bfloat16()
            with patch.object(F, "linear", wraps=F.linear) as linear:
                output = module(x)
            assert output.dtype == module.weight.dtype == torch.bfloat16
            assert torch.equal(output, reference)
            assert len(linear.call_args_list) == 2
            for args, _ in linear.call_args_list:
                tile, weight, _ = args
                assert tile.shape == (4, module.in_features)
                assert tile.dtype == weight.dtype == torch.float32
            assert work[name.removeprefix("model.")] == dict(
                calls=1, tiles=2, input_rows=5, padded_rows=3
            )
    assert all("forward" not in vars(m) for m in modules.values())


def test_empty_screen_cannot_nominate_a_tier():
    with pytest.raises(ValueError, match="positive"):
        bench.run_sweep(None, "torch", 64, 0, io.StringIO())


@pytest.mark.parametrize("fail_every_tier", [False, True])
def test_screen_selects_only_tiers_passing_all_cases_and_repeats(monkeypatch, fail_every_tier):
    model = tiny_ouro_model(dtype=torch.bfloat16)
    calls = Counter()
    long_case, short_case, c4 = bench.DEFAULT_CASES

    def simulate(model, backend, case, output_tokens, seed, *, work=None):
        assert (output_tokens, seed) == (16, 168)
        tier = 0 if work is None else {3: 2, 4: 3, 7: 4}[len(work) // len(model.model.layers)]
        calls[tier, case.name] += 1
        repeat = calls[tier, case.name]
        arms = {}
        for arm in ARMS:
            _, hits = expected_hits(case, arm)
            requests = []
            for i, hit in enumerate(hits):
                tokens = [10] * output_tokens
                if arm == "reuse" and case.name == long_case:
                    if (tier == 2 and repeat == 2) or (tier and fail_every_tier):
                        tokens[1] = 11
                if tier == 3 and case.name == c4 and i == 2:
                    tokens[2] = 12  # Stable cold drift on all three arms.
                if tier == 4 and case.name == short_case and repeat == 2:
                    tokens[4] = 13  # Parity holds inside one repeat, but the cold output drifted.
                start = hit
                if tier == 4 and case.name == c4 and arm == "reuse" and i == 3:
                    start = 0
                requests.append(
                    dict(
                        prompt_sha256=f"p{i}",
                        tokens=tokens,
                        hit_tokens=hit,
                        prefill_start=start,
                        expected_hit_tokens=hit,
                    )
                )
            arms[arm] = dict(requests=requests, prefix_hits=sum(hits), prefix_queries=len(hits))
        return arms, {}

    monkeypatch.setattr(bench, "run_precision_arms", simulate)
    stream = io.StringIO()
    review = bench.run_sweep(model, "torch", 64, 2, stream)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert Counter(r["kind"] for r in rows) == {"native": 6, "trial": 18, "review": 1}
    assert len(calls) == 12 and set(calls.values()) == {2}
    assert review["tiers"][2]["cache_parity_failed"] == [long_case]
    assert review["tiers"][3]["native_cold_changed"] == [c4]
    assert set(review["tiers"][4]["invalid"]) == {short_case, c4}
    assert review["lowest_passing_tier"] == (None if fail_every_tier else 3)
    assert review["lowest_passing_tier_matching_native_off"] is None
    trials = {(r["tier"], r["case"], r["repeat"]): r for r in rows if r["kind"] == "trial"}
    assert trials[2, long_case, 2]["parity_diffs"][0]["first_diff_1based"] == 2
    assert trials[3, c4, 1]["native_off_vs_fp32_off"] == [[2, 3, 10, 12]]
    assert (
        trials[4, short_case, 2]["controls"][0]["kind"] == "candidate_cold_changed_between_repeats"
    )
    assert trials[4, c4, 1]["controls"][0]["kind"] == "hit_or_offset"
