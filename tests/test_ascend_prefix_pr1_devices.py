"""CPU regressions for device attribution and cross-chip service comparisons."""

import json
from types import SimpleNamespace

import pytest

from benchmarks import ascend_prefix_pr1_compare as compare
from benchmarks import ascend_prefix_pr1_suite as suite
from benchmarks.ascend_prefix_pr1_hardware import identify


@pytest.mark.parametrize(
    "name,expected,count,verdict,sku",
    [
        ("Ascend910B3", "910B3", 1, "matched", "910B3"),
        ("Ascend 910B4", "910B4", 1, "matched", "910B4"),
        ("Ascend910B4", "910B3", 1, "mismatch", "910B4"),
        ("Ascend910B30", "910B3", 1, "mismatch", "910B30"),
        ("Ascend910B3", "910B3", 8, "mismatch", "910B3"),
        ("Ascend910B", "910B3", 1, "unverified", None),
        ("Ascend910B3", None, 1, "observed", "910B3"),
    ],
)
def test_runtime_device_evidence(name, expected, count, verdict, sku):
    selected = []
    npu = SimpleNamespace(
        set_device=selected.append,
        current_device=lambda: selected[-1],
        get_device_name=lambda i: {0: name}[i],
        device_count=lambda: count,
        mem_get_info=lambda i: {0: (1024, 4096)}[i],
    )
    device = identify(npu, "7", expected)
    assert selected == [0]
    assert device["visible_selector"] == "7" and device["logical_device"] == 0
    assert device["runtime_device_name"] == name and device["detected_sku"] == sku
    assert device["visible_device_count"] == count
    assert device["device_memory_bytes"] == 4096 and device["os_physical_memory_bytes"] > 0
    assert device["verification"] == verdict


@pytest.mark.parametrize("scope", ["all", "pr2"])
def test_both_skus_schedule_identical_checks(scope, tmp_path, capsys):
    out = tmp_path / "result"
    for chip in ("910B3", "910B4"):
        assert suite.main([scope, "--chip", chip, "--out", str(out), "--list"]) == 0
        listing = capsys.readouterr().out
        if chip == "910B3":
            first = listing
        else:
            assert listing == first
    assert not out.exists()


def test_manifest_layouts_match_by_arrival_mode(tmp_path):
    for mode in ("burst", "paced"):
        legacy = tmp_path / mode / "traffic.json"
        legacy.parent.mkdir()
        legacy.touch()
        assert suite.reference_manifest(tmp_path, mode) == legacy
        current = tmp_path / ("bf16-" + mode) / "traffic.json"
        current.parent.mkdir()
        current.touch()
        assert suite.reference_manifest(tmp_path, mode) == current


def test_explicit_sku_requires_one_card(tmp_path):
    result = suite.check_hardware(
        tmp_path, {"ASCEND_RT_VISIBLE_DEVICES": "0,7"}, tmp_path, "unused", tmp_path, "910B3"
    )
    assert result["verification"] == "unavailable"
    assert "one numeric" in result["detail"]


@pytest.mark.parametrize(
    "verification,expected_stages",
    [
        ("mismatch", ["config"]),
        ("unverified", ["config", "bf16-burst"]),
    ],
)
def test_unconfirmed_target_never_qualifies_run(
    tmp_path, monkeypatch, verification, expected_stages
):
    stages = []
    monkeypatch.setattr(
        suite, "check_hardware", lambda *_: dict(verification=verification, detail="test device")
    )
    monkeypatch.setattr(suite, "install_checks", lambda *_: None)

    def run(stage, *_):
        stages.append(stage.name)
        return {"status": "PASS"}

    monkeypatch.setattr(suite, "run_stage", run)
    out = tmp_path / "results"
    code = suite.main(
        ["--chip", "910B3", "--only", "config", "--only", "bf16-burst", "--out", str(out)]
    )
    review = compare.read(out / "review.json")
    assert stages == expected_stages
    assert code == 2 and review["status"] == "INCOMPLETE" and not review["full_pr1_pass"]
    assert verification in (out / "PR_RESULTS.md").read_text()


@pytest.fixture
def two_runs(tmp_path):
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    roots = [tmp_path / "first", tmp_path / "second"]
    for root, chip in zip(roots, ("910B3", "910B4"), strict=True):
        hardware = dict(
            runtime_device_name="Ascend" + chip,
            detected_sku=chip,
            verification="matched",
            visible_device_count=1,
            logical_device=0,
        )
        stages = {"environment": {"status": "PASS"}}
        for name in compare.TRAFFIC:
            write(root / name / "traffic.json", {"requests": [name]})
            service = dict(
                precision="native_bf16",
                linear_fp32_chunk=None,
                modes=["off", "main"],
                diagnostic_replay=False,
                control_issues=0,
                output_diffs=0,
                rounds=2,
                requests_per_mode=32,
                paired_ttft={"off_to_main": dict(median_saved_ms=-8, faster_pairs=0, pairs=2)},
            )
            stages[name] = {"status": "PASS", "evidence": service}
        write(root / "environment/checkpoint.json", {"model.safetensors": {"sha256": "same"}})
        write(
            root / "environment/environment.json",
            dict(
                torch="torch",
                torch_npu="torch-npu",
                chip=hardware["runtime_device_name"],
                visible="7",
                commit="candidate",
            ),
        )
        write(
            root / "review.json",
            dict(
                candidate="candidate",
                harness="tools",
                hardware=hardware,
                unpacked_policy="support",
                assigned_device="7",
                status="PASS",
                full_pr1_pass=False,
                stages=stages,
            ),
        )
    return roots


def test_compare_keeps_failures_and_negative_savings(two_runs):
    path = two_runs[0] / "review.json"
    run = compare.read(path)
    run["status"] = "FAIL"
    stage = run["stages"]["bf16-burst"]
    stage["status"] = "FAIL"
    stage["evidence"].update(output_diffs=1, paired_ttft={})
    path.write_text(json.dumps(run))
    result = compare.comparison(two_runs)
    assert "910B3 | bf16-burst | FAIL | 0 | 1 | — | —" in result
    assert "910B4 | bf16-burst | PASS | 0 | 0 | -8 | 0 / 2" in result


@pytest.mark.parametrize(
    "file,key,value,reason",
    [
        ("review.json", "candidate", "other", "candidate differs"),
        ("review.json", "hardware", {}, "single-device evidence"),
        ("bf16-paced/traffic.json", "requests", ["burst"], "bf16-paced/traffic.json"),
        (
            "environment/checkpoint.json",
            "model.safetensors",
            {"sha256": "other"},
            "checkpoint.json",
        ),
        ("environment/environment.json", "chip", "Ascend910B4", "environment stage"),
    ],
)
def test_compare_rejects_incompatible_runs(two_runs, file, key, value, reason):
    path = two_runs[0] / file
    data = compare.read(path)
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=reason):
        compare.comparison(two_runs)
