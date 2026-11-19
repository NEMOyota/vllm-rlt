"""Diagnostic replacement of published donor KV with an actual cold target's logical prefix."""

import hashlib
from unittest.mock import patch

import torch


def _prefix_tables(cache, rid, tokens):
    """Validate the whole allocation before selecting the complete logical prefix pages."""
    tables = cache.plane_block_tables(rid)
    pages = tokens // cache.block_size
    if len(tables) != cache.storage_depths or any(len(t) < pages for t in tables):
        raise ValueError(f"incomplete block tables for {rid}")
    blocks = [b for plane in tables for b in plane]
    if len(set(blocks)) != len(blocks) or any(not 0 <= b < cache.num_blocks for b in blocks):
        raise ValueError(f"invalid or aliased physical pages within {rid}")
    for depth in range(len(tables)):
        if not all(cache.token_written(rid, pos, depth) for pos in range(tokens)):
            raise ValueError(f"incomplete KV at depth {depth} in {rid}")
    return tuple(plane[:pages] for plane in tables)


def _read_page(cache, block):
    # On CPU copy explicitly so that a freed/reused allocation cannot mutate the snapshot.
    return tuple(
        pool[block].detach().to(device="cpu", copy=True).contiguous()
        for pool in (cache.key_cache, cache.value_cache)
    )


def _same_page(left, right):
    # Compare BF16 storage bits, including signed zero and NaN payloads.
    return all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(left, right, strict=True)
    )


class _AtRelease:
    """Watch one live request release; leave release itself to the next observer or native free."""

    def __init__(self, cache, prefix_tokens, rid):
        if cache.layout != "last_exited" or cache.dtype != torch.bfloat16:
            raise ValueError("the page control requires a BF16 last_exited cache")
        if prefix_tokens <= 0 or prefix_tokens % cache.block_size:
            raise ValueError("prefix_tokens must be positive and page-aligned")
        shape = (
            cache.num_blocks,
            cache.num_layers,
            cache.block_size,
            cache.num_kv_heads,
            cache.head_dim,
        )
        if any(
            p.shape != shape or p.dtype != cache.dtype for p in (cache.key_cache, cache.value_cache)
        ):
            raise ValueError("unexpected physical KV pool geometry")
        self._cache = cache
        self._tokens = prefix_tokens
        self._rid = rid
        self._hook = None
        self._started = self._observed = self._complete = False

    def __enter__(self):
        if self._cache is None or self._hook is not None:
            raise RuntimeError("use a fresh page control for each replay")
        cache = self._cache
        free = cache.free

        def before_free(rid):
            try:
                if rid == self._rid and cache.has_allocation(rid):
                    if self._started:
                        raise RuntimeError(f"duplicate live release for watched request {rid}")
                    self._started = True
                    self._visit(cache, _prefix_tables(cache, rid, self._tokens))
                    self._observed = True
            finally:
                free(rid)

        self._hook = patch.object(cache, "free", before_free)
        self._hook.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._hook.stop()
        self._cache = self._hook = None
        if exc_type is None:
            if not self._observed:
                raise RuntimeError(f"missing successful release observation for {self._rid}")
            self._complete = True


class ColdPages(_AtRelease):
    """Copy one target's actual full-layer K/V pages to host memory in [depth][logical] order."""

    def __init__(self, cache, prefix_tokens, focus_id):
        super().__init__(cache, prefix_tokens, focus_id)
        self._pages = None

    def _visit(self, cache, tables):
        self._pages = tuple(tuple(_read_page(cache, b) for b in plane) for plane in tables)

    @property
    def pages(self):
        if not self._complete:
            raise RuntimeError("finish the cold replay before using its prefix")
        return self._pages


class DonorPageControl(_AtRelease):
    """Round-trip or replace only the single donor's published prefix, before target admission."""

    def __init__(self, cache, prefix_tokens, replacement=None):
        super().__init__(cache, prefix_tokens, "donor")
        if not cache.enable_prefix_caching:
            raise ValueError("donor page control requires prefix caching")
        self._replacement = replacement
        self._counts = None
        self.before_sha256 = None

    def _validate_source(self, cache, tables):
        if set(cache._allocations) != {"donor"}:
            raise ValueError("donor must be the only live request before the intervention")
        # Reading the index directly avoids an additional lookup that would modify LRU.
        published = set(cache._prefixes.values())
        if any(group not in published for group in zip(*tables, strict=True)):
            raise ValueError("donor physical prefix has not been fully published")
        source = self._replacement
        if source is None:
            return
        shape = cache.key_cache.shape[1:]
        for table, plane in zip(tables, source, strict=True):
            if len(plane) != len(table):
                raise ValueError("replacement prefix has a different logical length")
            for pair in plane:
                if len(pair) != 2 or any(
                    t.shape != shape
                    or t.device.type != "cpu"
                    or t.dtype != cache.dtype
                    or not t.is_contiguous()
                    for t in pair
                ):
                    raise ValueError("expected CPU BF16 K/V pages matching the cache geometry")

    @torch.no_grad()
    def _visit(self, cache, tables):
        # Validate all pages before the first write. A device copy/readback failure
        # aborts this isolated engine replay before any target is admitted.
        self._validate_source(cache, tables)
        checked = changed = 0
        original = hashlib.sha256()
        for depth, plane in enumerate(tables):
            for logical, block in enumerate(plane):
                before = _read_page(cache, block)
                # Digest order is depth, logical page, then K and V.
                for part in before:
                    original.update(part.view(torch.uint8).numpy())
                source = before if self._replacement is None else self._replacement[depth][logical]
                differs = not _same_page(before, source)
                for pool, host in zip((cache.key_cache, cache.value_cache), source, strict=True):
                    pool[block].copy_(host)
                if not _same_page(_read_page(cache, block), source):
                    raise RuntimeError(
                        f"KV page readback mismatch at depth {depth}, logical page {logical}"
                    )
                changed += int(differs)
                checked += 1
        self._counts = {"checked_pages": checked, "changed_pages": changed}
        self.before_sha256 = original.hexdigest()

    def report(self):
        if not self._complete:
            raise RuntimeError("finish the controlled donor replay before reading its report")
        return self._counts
