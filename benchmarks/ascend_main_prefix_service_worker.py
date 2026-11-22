"""Start the stock HTTP server with benchmark-side counters in its spawned stage worker."""

import json
import os
from functools import partial
from pathlib import Path
from unittest.mock import patch

from benchmarks.ascend_main_prefix_grouping import (
    GroupedAscendAttention,
    attention_mode,
    group_counts,
)
from benchmarks.ascend_main_prefix_results import ids_digest
from vllm_rlt.engine import worker
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.entrypoints import serve
from vllm_rlt.request import RequestOutput


def observed_worker(mode, audit_path, *args, **kwargs):
    # This function runs inside the ordinary spawned worker. Grouping is selected
    # before its model/backend is built. No HTTP or engine protocol is changed.
    records = []
    state = {}
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
            backend=cache.backend,
            packed=engine.execution_config.packed_prefill,
            dtype=str(cache.dtype),
            prefix_enabled=cache.enable_prefix_caching,
            prefix_hits=cache.prefix_hits,
            prefix_queries=cache.prefix_queries,
            group_calls=group_counts(cache.attention.kernel),
            grouping_enabled=isinstance(cache.attention.kernel, GroupedAscendAttention),
            live_blocks=cache.num_used_blocks,
        )
        close(engine)

    try:
        with (
            attention_mode(mode),
            patch.object(RequestOutput, "from_request", classmethod(capture)),
            patch.object(StageEngine, "close", close_observed),
        ):
            worker.worker_main(*args, **kwargs)
    finally:
        Path(audit_path).write_text(json.dumps(dict(state=state, requests=records)) + "\n")


def main():
    observed = partial(
        observed_worker, os.environ["RLT_BENCH_ATTENTION"], os.environ["RLT_BENCH_AUDIT"]
    )
    # start_stage_worker resolves this target in the parent. The child imports
    # this module normally and calls the original worker_main.
    with patch.object(worker, "worker_main", observed):
        serve.main()


if __name__ == "__main__":
    main()
