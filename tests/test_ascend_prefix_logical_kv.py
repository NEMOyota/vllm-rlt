import pytest
import torch

from benchmarks.ascend_prefix_logical_kv import LogicalKVProbe, compare_logical_kv
from vllm_rlt.core.kv_cache_manager import KVCacheManager


def test_logical_kv_ignores_physical_page_order_and_detects_bf16_signed_zero():
    cache = KVCacheManager(2, 1, 4, 12, 2, 2, dtype=torch.bfloat16)
    assert cache.allocate("padding", 4)
    assert cache.allocate("reference", 4, initial_tokens=2)
    cache.free("padding")
    assert cache.ensure_capacity("reference", 4)
    assert cache.allocate("same", 4)
    assert cache.allocate("modified", 4)
    reference_pages, *_ = cache.plane_block_tables("reference")
    assert reference_pages[0] > reference_pages[1]
    ids = ["reference", "same", "modified"]
    with LogicalKVProbe(cache, 4, ids) as probe:
        for rid in ids:
            for depth, table in enumerate(cache.plane_block_tables(rid)):
                for page, block in enumerate(table):
                    for layer in range(cache.num_layers):
                        n = depth * 100 + page * 10 + layer
                        cache.key_cache[block, layer].fill_(n + 1)
                        cache.value_cache[block, layer].fill_(n + 2)
                    if depth == 1 and page == 1:
                        cache.value_cache[block, 1, 0, 0, 0] = -0.0 if rid == "modified" else 0.0
            cache.mark_imported_prefix(rid, 4)
            cache.free(rid)

    data = probe.report()
    assert compare_logical_kv(data["reference"], data["same"])["equal"]
    changed = compare_logical_kv(data["reference"], data["modified"])
    assert changed["mismatched_components"] == 1 and changed["compared_components"] == 8
    first = changed["first_diff"]
    assert (first["depth"], first["layer"], first["k_changed"], first["v_changed"]) == (
        1,
        1,
        False,
        True,
    )


def test_logical_kv_rejects_unwritten_pages_and_still_releases_request():
    cache = KVCacheManager(1, 1, 4, 8, 2, 2, dtype=torch.bfloat16)
    assert cache.allocate("target-0", 4)
    with pytest.raises(ValueError, match="incomplete logical prefix"):
        with LogicalKVProbe(cache, 4, "target-0"):
            cache.free("target-0")
    assert not cache.has_allocation("target-0")
    assert "free" not in vars(cache)
