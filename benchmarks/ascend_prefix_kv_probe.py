"""Observe physical KV ownership and donor bytes during one synchronous diagnostic replay."""

import hashlib
from contextlib import ExitStack
from unittest.mock import patch

import torch


def _page_digest(pool, block):
    # Copy one logical page, preserving BF16 bit patterns (including signed zero).
    # Each page contains every model layer at this recurrent depth.
    host = pool[block].detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(host.numpy()).hexdigest()


class KVProbe:
    """Watch one donor followed by a target burst. Copies are outside benchmark timing."""

    def __init__(self, cache, expected_hits):
        if not expected_hits or any(n < 0 or n % cache.block_size for n in expected_hits):
            raise ValueError("expected_hits must give one page-aligned token count per target")
        self.cache = cache
        self.expected = {f"target-{i}": n for i, n in enumerate(expected_hits)}
        self.shared_pages = max(expected_hits) // cache.block_size
        self.donor_prefix_pages = []
        self.targets = {}
        self.issues = []
        self._before = {}
        self._live_private = {}
        self._stack = ExitStack()
        self._result = None

    def __enter__(self):
        allocate, free = self.cache.allocate, self.cache.free

        def watch_allocate(request_id, max_tokens, *, initial_tokens=None, prefix=()):
            success = allocate(request_id, max_tokens, initial_tokens=initial_tokens, prefix=prefix)
            if success and request_id in self.expected:
                self._inspect_target(request_id, len(prefix))
            return success

        def watch_free(request_id):
            if request_id == "donor" and self.shared_pages:
                # Read actual donor ownership, before the engine drops its reference.
                # Calling lookup_prefix here would change the LRU order.
                tables = self.cache.plane_block_tables(request_id)
                self.donor_prefix_pages = [list(t[: self.shared_pages]) for t in tables]
                owned = [p for plane in self.donor_prefix_pages for p in plane]
                if any(len(plane) != self.shared_pages for plane in self.donor_prefix_pages):
                    self.issues.append({"kind": "short_donor_table"})
                if len(owned) != len(set(owned)):
                    self.issues.append({"kind": "duplicate_donor_page"})
                self._before = self._digests()
            free(request_id)
            self._live_private.pop(request_id, None)

        self._stack.enter_context(patch.object(self.cache, "allocate", watch_allocate))
        self._stack.enter_context(patch.object(self.cache, "free", watch_free))
        return self

    def _digests(self):
        return {
            (depth, logical, block, kind): _page_digest(pool, block)
            for depth, plane in enumerate(self.donor_prefix_pages)
            for logical, block in enumerate(plane)
            for kind, pool in (("key", self.cache.key_cache), ("value", self.cache.value_cache))
        }

    def _inspect_target(self, request_id, claimed_pages):
        tables = [list(t) for t in self.cache.plane_block_tables(request_id)]
        expected = self.expected[request_id]
        wanted = expected // self.cache.block_size
        admitted = claimed_pages * self.cache.block_size
        if admitted != expected:
            self.issues.append(
                dict(kind="hit_length", target=request_id, expected=expected, admitted=admitted)
            )
        self.targets[request_id] = dict(
            admitted_hit_tokens=admitted, expected_hit_tokens=expected, block_tables=tables
        )
        if wanted:
            if len(tables) != len(self.donor_prefix_pages):
                self.issues.append(dict(kind="donor_depth_count", target=request_id))
            for depth, (plane, donor) in enumerate(zip(tables, self.donor_prefix_pages)):
                if plane[:wanted] != donor[:wanted]:
                    self.issues.append(
                        dict(
                            kind="prefix_table",
                            target=request_id,
                            depth=depth,
                            expected=donor[:wanted],
                            actual=plane[:wanted],
                        )
                    )

        # Fully reserved target allocations are live together in the diagnostic burst.
        # A reused physical page may occur in many shared slots, but no private slot
        # may overlap a retained donor prefix or another concurrently owned tail.
        private = [block for plane in tables for block in plane[wanted:]]
        occupied = {p for plane in self.donor_prefix_pages for p in plane}
        for blocks in self._live_private.values():
            occupied.update(blocks)
        duplicates = set()
        for block in private:
            if block in occupied:
                duplicates.add(block)
            occupied.add(block)
        if duplicates:
            self.issues.append(
                dict(kind="private_page_overlap", target=request_id, pages=sorted(duplicates))
            )
        self._live_private[request_id] = private

    def __exit__(self, exc_type, exc, tb):
        self._stack.close()
        if exc_type is None:
            if self.shared_pages and not self.donor_prefix_pages:
                self.issues.append({"kind": "missing_donor"})
            missing = sorted(self.expected.keys() - self.targets.keys())
            if missing:
                self.issues.append(dict(kind="missing_target_allocations", targets=missing))
            changed = []
            for location, digest in self._digests().items():
                if self._before[location] != digest:
                    depth, logical, block, kind = location
                    changed.append(
                        dict(
                            depth=depth,
                            logical_page=logical,
                            physical_page=block,
                            kind=kind,
                            before_sha256=self._before[location],
                            after_sha256=digest,
                        )
                    )
            if changed:
                self.issues.append(
                    dict(kind="shared_kv_changed", changed_page_components=len(changed))
                )
            self._result = dict(
                ok=not self.issues,
                issues=self.issues,
                donor_prefix_pages=self.donor_prefix_pages,
                targets=self.targets,
                checked_kv_pages=sum(map(len, self.donor_prefix_pages)),
                changed_pages=changed,
            )
        # Retaining a completed probe to read its report must not retain the NPU pool.
        self.cache = None

    def report(self):
        if self._result is None:
            raise RuntimeError("close a successful KV probe context before requesting its report")
        return self._result
