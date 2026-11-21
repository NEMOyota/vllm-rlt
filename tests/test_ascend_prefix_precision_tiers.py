import csv
import io
import json
import sys
from collections import Counter
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from benchmarks import ascend_prefix_kv_precision
from benchmarks import ascend_prefix_precision_tiers as bench
from benchmarks.ascend_prefix_projection import fp32_decoder_projections
from benchmarks.ascend_prefix_scenarios import ARMS, suites
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

    def simulate(model, backend, case, output_tokens, seed, *, work=None, arm_offset=0):
        assert (output_tokens, seed, arm_offset) == (16, 168, 0)
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


def test_tier3_full_cli_covers_matrix_and_excludes_warmups_only_from_timings(tmp_path, monkeypatch):
    model = tiny_ouro_model(dtype=torch.bfloat16)
    rounds = Counter()
    output = tmp_path / "full.jsonl"
    monkeypatch.setattr(bench.OuroForCausalLM, "from_pretrained", lambda *a, **kw: model)
    monkeypatch.setattr(bench, "checkpoint_identity", lambda path: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiers",
            "--tier3-full",
            "--model",
            str(tmp_path),
            "--device",
            "cpu",
            "--out",
            str(output),
        ],
    )

    def simulate(model, backend, case, arm, output_tokens, seed):
        assert (output_tokens, seed) == (16, 168)
        attn = model.model.layers[0].self_attn
        patched = "forward" in vars(attn.q_proj)
        if patched:
            for layer in model.model.layers:
                assert "forward" not in vars(layer.mlp.gate_proj)
                for path in bench.ATTENTION:
                    module = layer.get_submodule(path)
                    module(torch.ones(1, module.in_features, dtype=torch.bfloat16))
        index, slot = divmod(rounds[patched, case.name], 3)
        assert arm == ARMS[(index + slot) % 3]
        rounds[patched, case.name] += 1
        primed, hits = expected_hits(case, arm)
        ms = 9000 if index == 0 else {"off": 10, "miss": 12, "reuse": 8}[arm]
        requests = []
        for i, hit in enumerate(hits):
            tokens = [7] * output_tokens
            start = hit
            if patched and arm == "reuse":
                if (case.name == "edge_p15" and index == 0) or (
                    case.name == "total512_p384" and index == 5
                ):
                    tokens[1] = 8
                if case.name == "c4_h2" and i == 3 and index == 4:
                    start += 16
            if patched and case.name == "c4_h4" and i == 2:
                tokens[7] = 9
            requests.append(
                dict(
                    prompt_sha256=f"p{i}",
                    tokens=tokens,
                    hit_tokens=hit,
                    prefill_start=start,
                    expected_hit_tokens=hit,
                    ttft_ms=ms,
                    e2e_ms=ms + 3,
                )
            )
        return dict(
            requests=requests,
            prefix_hits=sum(hits),
            prefix_queries=len(hits),
            primed_hit_tokens=primed,
            expected_primed_hit_tokens=primed,
            output_tokens=output_tokens * len(hits),
            wall_ms=ms + 3,
        )

    monkeypatch.setattr(ascend_prefix_kv_precision, "run_trial", simulate)
    assert bench.main() == 1
    config, *rows, review = [json.loads(s) for s in output.read_text().splitlines()]
    names = [c.name for c in suites()["full"]]
    assert len(names) == 29 and [c["name"] for c in config["cases"]] == review["cases"] == names
    assert config["projection_paths_by_tier"] == {"3": list(bench.ATTENTION)}
    assert (config["warmups"], config["repeats"], config["chunk_rows"]) == (1, 5, 64)
    assert len(rounds) == 2 * 29 and set(rounds.values()) == {6 * 3}
    assert Counter((r["kind"], r["warmup"]) for r in rows if "warmup" in r) == {
        ("native", True): 29,
        ("native", False): 145,
        ("trial", True): 29,
        ("trial", False): 145,
    }
    assert review["passing_tiers"] == [] and review["tiers"] == {
        "3": {
            "passed_cases": 26,
            "cache_parity_failed": ["edge_p15", "total512_p384"],
            "invalid": ["c4_h2"],
            "native_cold_changed": ["c4_h4"],
        }
    }
    with output.with_suffix(".csv").open() as f:
        csv_rows = {r["case"]: r for r in csv.DictReader(f)}
    assert list(csv_rows) == names
    warm_failure = csv_rows["edge_p15"]
    assert (
        warm_failure["cache_parity"] == "FAIL"
        and warm_failure["reuse_vs_miss_ttft_p50_saved_ms"] == ""
    )
    row = csv_rows["c4_h4"]
    assert row["cache_parity"] == "PASS" and row["native_cold_changed"] == "True"
    assert [
        float(row[key])
        for key in (
            "miss_ttft_p95_ms",
            "reuse_ttft_p50_ms",
            "reuse_vs_miss_ttft_p50_saved_ms",
        )
    ] == [12, 8, 4]
    assert float(csv_rows["evicted_p64"]["reuse_hit_tokens_per_request"]) == 0
    assert float(csv_rows["resident_p64"]["reuse_hit_tokens_per_request"]) == 64
