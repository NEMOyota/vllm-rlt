"""Replay cold, repeated and branched Ouro prompts on fresh cache-off and cache-on engines."""

import argparse
from itertools import zip_longest
from math import ceil
from pathlib import Path

import torch

from benchmarks import ascend_main_prefix_combo as combo
from benchmarks.ascend_main_prefix_linear import (
    add_precision_arg,
    benchmark_dtype,
    check_linear_usage,
    linear_usage,
    padded_fp32_linears,
)
from benchmarks.ascend_main_prefix_results import Report, ids_digest
from benchmarks.ascend_prefix_combo_checks import PrefillAudit, require
from vllm_rlt import SamplingParams
from vllm_rlt.models import OuroForCausalLM, load_tokenizer
from vllm_rlt.request import RequestPhase

# Same 71-token prompt with and without a restrictive per-step token budget.
# The long branch changes its last 32 tokens, starting inside a 16-token page.
SHORT = combo.Case(48, 23)
CASES = {
    "short_full": (SHORT, 71),
    "short_split": (SHORT, 16),
    "long_full": (combo.Case(1068, 32), 1100),
}
PARAMS = SamplingParams(
    max_tokens=combo.OUTPUTS,
    temperature=0,
    min_loops=4,
    max_loops=4,
    exit_threshold=1,
    ignore_eos=True,
)


def prompt_pair(tokenizer, case):
    source = tokenizer.encode(
        " Review the service manual. Record the inspection date and condition of each component.",
        add_special_tokens=False,
    )
    replacement = tokenizer.encode(
        " Summarize the invoice. List the quantities, delivery dates and total amount due.",
        add_special_tokens=False,
    )
    length = case.prefix + case.suffix
    if not source or not replacement:
        raise ValueError("tokenizer produced an empty replay passage")
    cold = (source * ceil(length / len(source)))[:length]
    # Pin the first changed token even if the tokenizer shares leading whitespace.
    offset = next((i for i, token in enumerate(replacement) if token != cold[case.prefix]), None)
    if offset is None:
        raise ValueError("tokenizer could not construct a distinct branch")
    tail = replacement[offset:] + replacement[:offset]
    branch = cold[: case.prefix] + (tail * ceil(case.suffix / len(tail)))[: case.suffix]
    return {"cold": cold, "repeat": cold, "branch": branch}


def run_request(engine, name, prompt, expected_hit, budget, *, audit):
    cache = engine.cache_manager
    expected_spans = [
        [i, min(i + budget, len(prompt))] for i in range(expected_hit, len(prompt), budget)
    ]
    hits, queries = cache.prefix_hits, cache.prefix_queries
    spans, audits, finishes = [], [], []
    engine.add_request(name, prompt, PARAMS)
    # One request at a time. Each ordinary PREFILL step completes all four loops.
    # Bound the replay so a scheduler that keeps resubmitting a span cannot hang the suite.
    max_steps = len(prompt) + (4 + 2) * (PARAMS.max_tokens + 1)
    for step in range(max_steps):
        if not engine.has_unfinished_requests():
            break
        if audit and step < len(expected_spans):
            start, end = expected_spans[step]
            observer = PrefillAudit(cache, {name: start}, end)
            try:
                with observer.observe():
                    outputs = engine.step()
                audits.append(observer.result())
            except AssertionError as error:
                raise AssertionError(f"{name} prefill [{start},{end}): {error}") from error
        else:
            outputs = engine.step()
        batch = engine.last_schedule
        if batch is not None and batch.phase == RequestPhase.PREFILL:
            require(batch.loop_count == 4, f"{name} prefill ran {batch.loop_count} loops")
            for item in batch.requests:
                require(item.request.request_id == name, f"unexpected request in {name} replay")
                spans.append([item.token_start, item.token_start + item.token_count])
        finishes.extend(out for out in outputs if out.finished)
    require(not engine.has_unfinished_requests(), f"{name} did not drain after {max_steps} steps")
    require(
        len(finishes) == 1 and finishes[0].request_id == name, f"{name} missing or duplicate finish"
    )
    out = finishes[0]
    controls = {
        "hit_tokens": (expected_hit, cache.prefix_hits - hits),
        "admissions": (1, cache.prefix_queries - queries),
        "prefill_spans": (expected_spans, spans),
        "live_blocks_after": (0, cache.num_used_blocks),
        "owns_allocation_after": (False, cache.has_allocation(name)),
        "finish_reason": ("length", out.finish_reason),
        "output_length": (PARAMS.max_tokens, len(out.token_ids)),
        "exit_depths": ([4] * PARAMS.max_tokens, out.exit_depths),
    }
    return dict(
        prompt_sha256=ids_digest(prompt),
        expected_hit_tokens=expected_hit,
        expected_prefill_spans=expected_spans,
        **{key: actual for key, (_, actual) in controls.items()},
        token_ids=list(out.token_ids),
        controls_failed=[key for key, (expected, actual) in controls.items() if expected != actual],
        attention_steps=audits,
    )


def compare_tokens(left, ref, right, actual):
    check = dict(reference=left, actual=right, status="NOT_RUN")
    if "token_ids" not in ref or "token_ids" not in actual:
        return check
    require(
        ref["prompt_sha256"] == actual["prompt_sha256"], "different prompts in token comparison"
    )
    for pos, (x, y) in enumerate(zip_longest(ref["token_ids"], actual["token_ids"]), 1):
        if x != y:
            return check | dict(status="FAIL", first_diff_1based=pos, expected_id=x, actual_id=y)
    return check | dict(status="PASS")


@torch.inference_mode()
def replay(model, tokenizer, report, names, *, linear_work=None):
    cases = {}
    failed = []
    for name in names:
        case, budget = CASES[name]
        prompts = prompt_pair(tokenizer, case)
        length = len(prompts["cold"])
        # Only complete pages can hit. Even an exact repeat needs the final prompt hidden state.
        expected_on = {
            "cold": 0,
            "repeat": (length - 1) // combo.BLOCK * combo.BLOCK,
            "branch": case.prefix // combo.BLOCK * combo.BLOCK,
        }
        runs = {"off": {}, "on": {}}
        for mode, rows in runs.items():
            report.location = dict(case=name, mode=mode, phase="engine")
            engine = combo.open_engine(
                model, case, "off" if mode == "off" else "reuse", token_budget=budget
            )
            try:
                require(
                    engine.execution_config.packed_prefill
                    and engine.cache_manager.backend == "ascend_eager"
                    and engine.cache_manager.key_cache.dtype
                    == engine.cache_manager.value_cache.dtype
                    == next(model.parameters()).dtype,
                    "expected packed native-dtype ascend_eager",
                )
                for request, prompt in prompts.items():
                    report.location = dict(case=name, mode=mode, request=request)
                    expected = expected_on[request] if mode == "on" else 0
                    hits_before = engine.cache_manager.prefix_hits
                    try:
                        row = run_request(
                            engine,
                            request,
                            prompt,
                            expected,
                            budget,
                            audit=name == "short_split" and request == "branch",
                        )
                    except AssertionError as error:
                        row = dict(
                            controls_failed=[str(error)],
                            prompt_sha256=ids_digest(prompt),
                            expected_hit_tokens=expected,
                            hit_tokens=engine.cache_manager.prefix_hits - hits_before,
                        )
                    rows[request] = row
                    report.record("REPLAY_REQUEST", **report.location, **row)
                    if "token_ids" not in row:
                        break  # Subsequent requests depend on a completed donor in this engine.
            finally:
                engine.close()
        comparisons = [
            compare_tokens(
                f"off.{rid}", runs["off"].get(rid, {}), f"on.{rid}", runs["on"].get(rid, {})
            )
            for rid in prompts
        ]
        comparisons.extend(
            compare_tokens(
                f"{mode}.cold", rows.get("cold", {}), f"{mode}.repeat", rows.get("repeat", {})
            )
            for mode, rows in runs.items()
        )
        controls_ok = all(
            len(rows) == len(prompts) and not any(row["controls_failed"] for row in rows.values())
            for rows in runs.values()
        )
        cases[name] = dict(
            prompt_tokens=length,
            branch_at=case.prefix,
            token_budget=budget,
            runs=runs,
            controls_ok=controls_ok,
            token_comparisons=comparisons,
        )
        if not controls_ok or any(check["status"] != "PASS" for check in comparisons):
            failed.append(name)
    if linear_work is not None:
        usage = linear_usage(linear_work)
        report.record("LINEAR_USAGE", **usage)
        check_linear_usage(usage)
    # Different budgets may change BF16 arithmetic even on an engine with caching off.
    # Report these comparisons alongside the same-budget acceptance checks.
    budget_comparisons = []
    if "short_full" in cases and "short_split" in cases:
        for mode in ("off", "on"):
            for rid in ("cold", "repeat", "branch"):
                budget_comparisons.append(
                    compare_tokens(
                        f"short_full.{mode}.{rid}",
                        cases["short_full"]["runs"][mode].get(rid, {}),
                        f"short_split.{mode}.{rid}",
                        cases["short_split"]["runs"][mode].get(rid, {}),
                    )
                )
    return report.review(
        "FAIL" if failed else "PASS",
        cases=cases,
        failed=failed,
        budget_comparisons=budget_comparisons,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    add_precision_arg(parser)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error("provide a local Ouro model directory")
    report = Report("REPLAY", args.out_dir, linear_fp32_chunk=args.linear_fp32_chunk)

    def run(args, device, report):
        model = OuroForCausalLM.from_pretrained(
            args.model, device=device, dtype=benchmark_dtype()
        ).eval()
        require(
            all(p.dtype == benchmark_dtype() for p in model.parameters()),
            "unexpected checkpoint weight dtype",
        )
        tokenizer = load_tokenizer(str(args.model.resolve()))
        with padded_fp32_linears(model, args.linear_fp32_chunk) as work:
            return replay(model, tokenizer, report, CASES, linear_work=work)

    return report.execute(args, run)


if __name__ == "__main__":
    raise SystemExit(main())
