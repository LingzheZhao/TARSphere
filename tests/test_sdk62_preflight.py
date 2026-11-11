"""Installed SDK identity must be checked without launching the simulator."""

import pytest
from importlib import metadata
from tarsphere.cli import check_environment as preflight


@pytest.mark.parametrize(
    "version,passed",
    [("6.2.0", True), ("6.2.1+local", True), ("5.1.0", False), ("6.20.0", False)],
)
def test_sdk_identity(monkeypatch, version, passed):
    monkeypatch.setattr(
        preflight.metadata,
        "version",
        lambda name: version if name == "isaacsim" else pytest.fail(name),
    )
    check = preflight.installed_sdk62_verification()
    assert check.passed is passed
    assert check.required
    assert check.observed == version


def test_missing_sdk_metadata_fails(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(preflight.metadata, "version", missing)
    check = preflight.installed_sdk62_verification()
    assert not check.passed
    assert check.required


def test_external_grasp_resources(monkeypatch, tmp_path):
    root = tmp_path / "models"
    assets = tmp_path / "grippers"
    assets.mkdir()
    for kind in ("gen", "dis"):
        directory = root / kind
        directory.mkdir(parents=True)
        (directory / "config.yaml").write_text("model: test\n")
        (directory / "epoch_1.pth").write_bytes(b"test fixture")
    monkeypatch.setenv("TARSPHERE_GRASPGENX_CHECKPOINT_ROOT", str(root))
    monkeypatch.setenv("TARSPHERE_GRASPGENX_GRIPPER_ASSETS", str(assets))
    assert preflight.graspgenx_asset_verification().passed
    (root / "dis" / "epoch_1.pth").unlink()
    assert not preflight.graspgenx_asset_verification().passed


def test_default_grasp_paths(monkeypatch, tmp_path):
    monkeypatch.delenv("TARSPHERE_GRASPGENX_CHECKPOINT_ROOT", raising=False)
    monkeypatch.delenv("TARSPHERE_GRASPGENX_GRIPPER_ASSETS", raising=False)
    monkeypatch.setattr(preflight, "REPOSITORY_ROOT", tmp_path)
    check = preflight.graspgenx_asset_verification()
    assert not check.passed
    assert (
        str(tmp_path / "third_party/GraspGenX/ext/graspgenx_checkpoints/release")
        in check.observed
    )
    assert (
        str(
            tmp_path
            / "third_party/GraspGenX/ext/gripper_descriptions/gripper_descriptions/assets"
        )
        in check.observed
    )


@pytest.mark.parametrize("override", [None, "5557"])
def test_service_dry_run_matches_grasp_client(monkeypatch, tmp_path, override):
    import os
    import subprocess
    import sys
    from pathlib import Path
    from tarsphere.infra.manipulation.graspgenx import GraspGenXSettings

    env = os.environ.copy()
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "TARSPHERE_GRASPGENX_PORT",
        "TARSPHERE_SAM3_PORT",
        "TARSPHERE_GRASPGENX_HOST",
        "TARSPHERE_SAM3_HOST",
    ):
        env.pop(key, None)
    checkpoint = tmp_path / "sam.pt"
    checkpoint.touch()
    env.update(
        TARSPHERE_PYTHON=sys.executable,
        TARSPHERE_GRASPGENX_PYTHON=sys.executable,
        TARSPHERE_SAM3_CHECKPOINT=str(checkpoint),
        TARSPHERE_MODEL_SERVICE_STATE_DIR=str(tmp_path / "state"),
    )
    if override:
        env["TARSPHERE_GRASPGENX_PORT"] = override
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "tarsphere/infra/services/start.sh"), "--dry-run"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    expected = int(override) if override else GraspGenXSettings().port
    assert f"GraspGenX 127.0.0.1:{expected}" in result.stdout
    assert "SAM3 127.0.0.1:5558" in result.stdout

def test_xnavdp_client_root_required(monkeypatch):
    monkeypatch.delenv('TARSPHERE_XNAVDP_ROOT', raising=False)
    check = preflight.xnavdp_client_verification()
    assert not check.passed
    assert check.required


def test_xnavdp_valid_client_root(monkeypatch, tmp_path):
    client = tmp_path / 'eval/src/client_utils.py'
    client.parent.mkdir(parents=True)
    client.touch()
    monkeypatch.setenv('TARSPHERE_XNAVDP_ROOT', str(tmp_path))
    monkeypatch.delenv('TARSPHERE_XNAVDP_PORT', raising=False)
    assert preflight.xnavdp_client_verification().passed
    client.unlink()
    assert not preflight.xnavdp_client_verification().passed
