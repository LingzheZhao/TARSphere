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
