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
