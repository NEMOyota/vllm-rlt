"""Identify the single logical NPU used by the prefix cache validation workers."""

import json
import os
import re
import sys
from pathlib import Path


def identify(npu, selector, expected=None):
    npu.set_device(0)
    logical = npu.current_device()
    name = npu.get_device_name(logical)
    count = npu.device_count()
    # A family name such as "Ascend910B" does not establish a particular SKU.
    match = re.search(r"(?<![0-9])((?:910|310)[A-Z][0-9]+[A-Z]*)(?![A-Z0-9])", name.upper())
    sku = match[1] if match else None
    verification = "observed"
    detail = None
    if expected:
        if count != 1 or logical != 0:
            verification = "mismatch"
            detail = f"expected a single visible NPU at logical index 0, got {count=} {logical=}"
        elif sku is None:
            verification = "unverified"
            detail = f"runtime name {name!r} does not identify the requested {expected} SKU"
        elif sku != expected:
            verification = "mismatch"
            detail = f"requested {expected}, runtime reports {name!r}"
        else:
            verification = "matched"
    try:
        _, total = npu.mem_get_info(logical)
    except (AttributeError, OSError, RuntimeError) as exc:
        print(f"Could not query device HBM: {exc}", file=sys.stderr)
        total = None
    return dict(
        expected_chip=expected,
        runtime_device_name=name,
        detected_sku=sku,
        visible_selector=selector,
        logical_device=logical,
        visible_device_count=count,
        device_memory_bytes=total,
        os_physical_memory_bytes=os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
        verification=verification,
        detail=detail,
    )


if __name__ == "__main__":
    import torch_npu

    evidence = identify(
        torch_npu.npu,
        os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        sys.argv[2] if len(sys.argv) > 2 else None,
    )
    Path(sys.argv[1]).write_text(json.dumps(evidence, indent=2) + "\n")
