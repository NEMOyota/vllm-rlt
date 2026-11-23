"""Start the stock HTTP server with benchmark-side counters in its spawned stage worker."""

import json
import os
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

from torch import nn

from benchmarks.ascend_main_prefix_grouping import (
    GroupedAscendAttention,
    attention_mode,
    group_counts,
)
from benchmarks.ascend_main_prefix_linear import (
    linear_usage,
    padded_fp32_linears,
    precision_settings,
)
from benchmarks.ascend_main_prefix_results import ids_digest
from benchmarks.ascend_prefix_opportunities import observe_prefill
from vllm_rlt.engine import worker
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.entrypoints import serve
from vllm_rlt.request import RequestOutput


def observed_worker(mode, audit_path, *args, linear_fp32_chunk=None, **kwargs):
    # This function runs inside the ordinary spawned worker. Grouping is selected
    # before its model/backend is built. No HTTP or engine protocol is changed.
    records = []
    state = {}
    work = None
    steps = []
    from_request = RequestOutput.from_request
    close = StageEngine.close

    def capture(cls, request):
        output = from_request(request)
        if output.finished:
            records.append(
                dict(
                    prompt_sha256=ids_digest(output.prompt_token_ids),
                    tokens=output.token_ids,
                    cached_tokens=request.prefill_window_origin,
                    exit_depths=output.exit_depths,
                    finish_reason=output.finish_reason,
                )
            )
        return output

    def close_observed(engine):
        cache = engine.cache_manager
        state.update(
            **precision_settings(linear_fp32_chunk),
            backend=cache.backend,
            packed=engine.execution_config.packed_prefill,
            dtype=str(cache.dtype),
            linear_weight_dtypes=sorted(
                {
                    str(module.weight.dtype)
                    for module in engine.model.modules()
                    if isinstance(module, nn.Linear)
                }
            ),
            async_scheduling=engine.execution_config.async_scheduling,
            multi_stream=engine.execution_config.multi_stream,
            prefix_enabled=cache.enable_prefix_caching,
            prefix_hits=cache.prefix_hits,
            prefix_queries=cache.prefix_queries,
            group_calls=group_counts(cache.attention.kernel),
            grouping_enabled=isinstance(cache.attention.kernel, GroupedAscendAttention),
            live_blocks=cache.num_used_blocks,
        )
        if work is not None:
            state["linear_work"] = linear_usage(work)
        close(engine)

    try:
        with ExitStack() as stack:
            if os.environ.get("RLT_BENCH_PREFILL_STEPS") == "1":
                stack.enter_context(observe_prefill(steps))
            if linear_fp32_chunk is not None:

                def build_engine(model, *engine_args, **engine_kwargs):
                    nonlocal work
                    work = stack.enter_context(padded_fp32_linears(model, linear_fp32_chunk))
                    return StageEngine(model, *engine_args, **engine_kwargs)

                stack.enter_context(patch.object(worker, "StageEngine", build_engine))
            stack.enter_context(attention_mode(mode))
            stack.enter_context(patch.object(RequestOutput, "from_request", classmethod(capture)))
            stack.enter_context(patch.object(StageEngine, "close", close_observed))
            worker.worker_main(*args, **kwargs)
    finally:
        Path(audit_path).write_text(
            json.dumps(dict(state=state, requests=records, prefill_steps=steps)) + "\n"
        )


def main():
    rows = os.environ.get("RLT_BENCH_LINEAR_FP32_CHUNK")
    observed = partial(
        observed_worker,
        os.environ["RLT_BENCH_ATTENTION"],
        os.environ["RLT_BENCH_AUDIT"],
        linear_fp32_chunk=int(rows) if rows else None,
    )
    # start_stage_worker resolves this target in the parent. The child imports
    # this module normally and calls the original worker_main.
    with patch.object(worker, "worker_main", observed):
        serve.main()


if __name__ == "__main__":
    main()
