import csv
from types import SimpleNamespace

import pytest

from benchmarks import ascend_prefix_miss_control as bench


def sample(offset=0):
    return dict(
        prefix_hits=0,
        prefix_queries=4,
        requests=[
            dict(
                prompt_sha256=str(i),
                tokens=[i] * 16,
                hit_tokens=0,
                prefill_start=0,
                ttft_ms=t + offset,
            )
            for i, t in enumerate((90, 100, 102, 180))
        ],
    )


def test_miss_pairs_preserve_order_burst_median_and_delta_sign(tmp_path):
    calls = []

    def variant(mode, audit, args, kwargs, profile_target):
        assert audit is None and profile_target is None
        calls.append(mode)
        return sample(15 if mode == "framework_both" else 0)

    run = SimpleNamespace(
        modes=("original", "framework_both"),
        candidate="framework_both",
        run_variant=variant,
        reference=sample(),
        record=lambda *a, **kw: None,
    )
    rows = bench.measure(run, (), {}, 10, tmp_path)
    assert calls == ["original", "framework_both", "framework_both", "original"] * 5
    with (tmp_path / "pairs.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 10
    summary = bench.summarize(rows)
    assert summary["original_ms"] == 101 and summary["framework_both_ms"] == 116
    assert summary["paired_extra_ms"] == [15] * 10
    assert summary["slower_pairs"] == 10 and summary["by_first"]["original"]["pairs"] == 5
    assert bench.should_trace(summary, 10) and not bench.should_trace(summary, None)
    # Even ten slow pairs do not trigger a trace if the effect is small in one order.
    for row in rows:
        if row["first"] == "framework_both":
            row["extra_ms"] = 1
    assert not bench.should_trace(bench.summarize(rows), 10)


@pytest.mark.parametrize("mistake", ["prefix_counter", "prefill_start", "prompt", "tokens"])
def test_invalid_sample_cannot_enter_miss_timings(mistake):
    wrong = sample()
    if mistake == "prefix_counter":
        wrong["prefix_queries"] = 0
    elif mistake == "prefill_start":
        wrong["requests"][2]["prefill_start"] = 128
    elif mistake == "prompt":
        wrong["requests"][2]["prompt_sha256"] = "other"
    else:
        wrong["requests"][2]["tokens"][-1] += 1
    run = SimpleNamespace(
        run_variant=lambda *a, **kw: wrong, reference=sample(), record=lambda *a, **kw: None
    )
    with pytest.raises(bench.paired.GateFailure):
        bench.checked_miss(run, "framework_both", (), {}, pair=7)
