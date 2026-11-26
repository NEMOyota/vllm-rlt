import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from capture import load_inputs
from common import CASES
from probes import Probes
from replay_op import replay
from report import differences

from benchmarks import ascend_main_prefix_matrix as matrix
from benchmarks import ascend_pr1_acceptance as acc
from benchmarks.ascend_main_prefix_linear import padded_fp32_linears
from tests.helpers import tiny_ouro_config
from vllm_rlt.attention import AttentionBackend, BackendCapabilities
from vllm_rlt.kernels.ascend_attention import AscendAttention
from vllm_rlt.models import OuroForCausalLM

torch.set_num_threads(1)
torch.manual_seed(13)
model = (
    OuroForCausalLM(tiny_ouro_config(max_position_embeddings=1024, vocab_size=600))
    .to(torch.bfloat16)
    .eval()
)
root = Path(tempfile.mkdtemp(prefix="op-diagnostic-smoke-"))
parser = argparse.ArgumentParser(
    description="CPU harness integration check using tiny random Ouro weights."
)
parser.add_argument("--history", type=Path)
history = parser.parse_args().history
if history:
    assert load_inputs(matrix, acc, "total512_p256", 49152, history)[1]["validation"] == "MATCH"


def backend(*a, **kw):
    return AttentionBackend(
        AscendAttention(),
        BackendCapabilities(packed_prefill=True),
        {"selected_backend": "ascend_eager"},
    )


with (
    patch("vllm_rlt.engine.stage_engine.create_backend", backend),
    patch("vllm_rlt.engine.stage_engine.packed_prefill_eligible", return_value=True),
):
    for case, spec in CASES.items():
        config = matrix.FULL[case]
        inputs = matrix.combo.workload(config, 600, seed=spec["seed"])
        for precision in [None, 256] if case == "total512_p256" else [None]:
            with padded_fp32_linears(model, precision) as work:
                records = {}
                plains = {}
                blobs = []
                for arm in ("off", "miss", "reuse"):
                    folder = root / f"{case}-{precision}-{arm}"
                    folder.mkdir()
                    plains[arm] = matrix.trial(model, config, "main", arm, inputs, linear_work=work)
                    scan = Probes(model, spec, inputs, "scan", folder)
                    with scan.installed():
                        got = matrix.trial(model, config, "main", arm, inputs, linear_work=work)
                    assert got["hit_tokens"] == config.starts(arm)
                    assert all(
                        v["tokens"] == plains[arm]["requests"][r]["tokens"]
                        for r, v in got["requests"].items()
                    )
                    assert scan.head["step"] == spec["step"] and scan.sdpa and scan.layouts
                    records[arm] = scan.events
                diffs, extra, missing = differences(records["off"], records["reuse"])
                opts = [e for e in records["reuse"] if e["kind"] == "linear_output"]
                # Exercise full-shape replay even when a tiny model has no token flip.
                selection = next((d for d in diffs if d["kind"] == "linear_output"), opts[0])
                for arm in ("off", "reuse"):
                    folder = root / f"{case}-{precision}-dump" / arm
                    folder.mkdir(parents=True)
                    rows = [
                        e
                        for e in records[arm]
                        if e["call"] == selection["call"] and e["pos"] == selection["pos"]
                    ]
                    selection.setdefault("sides", {})[arm] = {
                        k: next(e["digest"] for e in rows if e["kind"] == "linear_" + k)
                        for k in ("input", "output")
                    }
                    obs = Probes(model, spec, inputs, "dump", folder, selection)
                    with obs.installed():
                        matrix.trial(model, config, "main", arm, inputs, linear_work=work)
                    assert obs.saved
                    blobs.append(torch.load(folder / "operator.pt", weights_only=True))
                result = replay(*blobs, "cpu")
                assert (
                    result["off"]["full_output_bitwise"] and result["reuse"]["full_output_bitwise"]
                )
                print(case, precision, len(records["off"]), result["call"], flush=True)
print(root)
