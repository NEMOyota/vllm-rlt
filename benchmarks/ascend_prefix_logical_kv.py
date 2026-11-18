"""Compare logical prefix KV at the free boundary of synchronous BF16 replays."""

import hashlib
from unittest.mock import patch

import torch


def _hash_layers(pool, blocks):
    hashes = [hashlib.sha256() for _ in range(pool.shape[1])]
    for physical in blocks:
        # One host transfer contains every layer of this page. Keep the original
        # BF16 bits and hash in token/head/dimension order within each layer.
        page = pool[physical].detach().cpu().contiguous().view(torch.uint8).numpy()
        for digest, layer_bytes in zip(hashes, page, strict=True):
            digest.update(layer_bytes)
    return [digest.hexdigest() for digest in hashes]


def _snapshot(cache, rid, prefix_tokens):
    pages = prefix_tokens // cache.block_size
    fingerprints = []
    for depth, table in enumerate(cache.plane_block_tables(rid)):
        if len(table) < pages or not all(
            cache.token_written(rid, position, depth) for position in range(prefix_tokens)
        ):
            raise ValueError(f"incomplete logical prefix at depth {depth} for {rid}")
        # Each depth has its own physical block IDs. Both hashes concatenate
        # only the requested logical pages, in ascending logical token order.
        blocks = table[:pages]
        keys = _hash_layers(cache.key_cache, blocks)
        values = _hash_layers(cache.value_cache, blocks)
        fingerprints.append([{"k": k, "v": v} for k, v in zip(keys, values, strict=True)])
    return {
        "tap": "before_free",
        "dtype": "bfloat16",
        "prefix_tokens": prefix_tokens,
        "block_size": cache.block_size,
        "num_kv_heads": cache.num_kv_heads,
        "head_dim": cache.head_dim,
        "sha256": fingerprints,
    }


class LogicalKVProbe:
    """Fingerprint selected requests immediately before their allocation is freed.

    Use one context per synchronous replay with complete, page-aligned prefix
    tokens. These snapshots describe bytes at release time only. Equal hashes
    do not establish that earlier intermediate states matched.
    """

    def __init__(self, cache, prefix_tokens, request_ids):
        if cache.layout != "last_exited" or cache.dtype != torch.bfloat16:
            raise ValueError("logical fingerprint requires BF16 last_exited KV")
        if prefix_tokens <= 0 or prefix_tokens % cache.block_size:
            raise ValueError("prefix_tokens must be a positive multiple of the page size")
        self._cache = cache
        self._prefix_tokens = prefix_tokens
        self._ids = {request_ids} if isinstance(request_ids, str) else set(request_ids)
        if not self._ids:
            raise ValueError("select at least one request ID to fingerprint")
        self._snapshots = {}
        self._hook = None
        self._complete = False

    def __enter__(self):
        if self._hook is not None or self._cache is None:
            raise RuntimeError("use a fresh logical KV probe for each replay")
        cache = self._cache
        free = cache.free

        def observe_free(rid):
            try:
                if rid in self._ids and cache.has_allocation(rid):
                    if rid in self._snapshots:
                        raise RuntimeError(f"duplicate release of a watched request ID: {rid}")
                    self._snapshots[rid] = _snapshot(cache, rid, self._prefix_tokens)
            finally:
                free(rid)

        self._hook = patch.object(cache, "free", observe_free)
        self._hook.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._hook.stop()
        self._hook = self._cache = None
        if exc_type is None:
            missing = self._ids.difference(self._snapshots)
            if missing:
                raise RuntimeError(f"missing logical KV snapshots for {sorted(missing)}")
            self._complete = True

    def report(self):
        if not self._complete:
            raise RuntimeError("close a successful logical KV probe before requesting its report")
        return self._snapshots


def compare_logical_kv(reference, actual):
    """Report the first differing zero-based (depth, layer), ordered by recurrence."""
    for field in ("tap", "dtype", "prefix_tokens", "block_size", "num_kv_heads", "head_dim"):
        if reference[field] != actual[field]:
            raise ValueError(f"incompatible logical KV snapshots: {field}")
    first = None
    mismatches = compared = 0
    for depth, (ref_layers, test_layers) in enumerate(
        zip(reference["sha256"], actual["sha256"], strict=True)
    ):
        for layer, (ref, test) in enumerate(zip(ref_layers, test_layers, strict=True)):
            k_changed = ref["k"] != test["k"]
            v_changed = ref["v"] != test["v"]
            compared += 2
            mismatches += k_changed + v_changed
            if first is None and (k_changed or v_changed):
                first = {
                    "depth": depth,
                    "layer": layer,
                    "k_changed": k_changed,
                    "v_changed": v_changed,
                    "reference_sha256": ref,
                    "actual_sha256": test,
                }
    return {
        "equal": first is None,
        "first_diff": first,
        "compared_components": compared,
        "mismatched_components": mismatches,
    }
