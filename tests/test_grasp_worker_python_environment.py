"""The isolated worker must execute the supplied virtualenv entry point."""
from pathlib import Path
import sys
from tarsphere.infra.manipulation.g1_grasp import G1GraspPlannerClient


def test_venv_python_symlink_is_preserved(tmp_path):
    executable = tmp_path / 'venv/bin/python'
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    client = G1GraspPlannerClient(python_executable=executable)
    assert client.python_executable == executable
    assert client.python_executable != executable.resolve()


def test_explicit_python_overrides_configured_isolated_environment(monkeypatch, tmp_path):
    configured = tmp_path / 'configured/bin/python'
    configured.parent.mkdir(parents=True)
    configured.symlink_to(sys.executable)
    monkeypatch.setenv('TARSPHERE_CUROBO_PYTHON', str(configured))
    assert G1GraspPlannerClient().python_executable == configured
    explicit = tmp_path / 'explicit/bin/python'
    explicit.parent.mkdir(parents=True)
    explicit.symlink_to(sys.executable)
    assert G1GraspPlannerClient(python_executable=explicit).python_executable == explicit


def test_clean_environment_keeps_explicit_toolkit_and_allows_discovery(monkeypatch):
    monkeypatch.delenv('TARSPHERE_DEMO_GPU_UUID', raising=False)
    monkeypatch.setenv('CUDA_HOME', '/provisioned/cuda13')
    monkeypatch.setenv('CUDA_PATH', '/provisioned/cuda13')
    environment = G1GraspPlannerClient()._clean_environment()
    assert environment['CUDA_HOME'] == '/provisioned/cuda13'
    assert environment['CUDA_PATH'] == '/provisioned/cuda13'
    monkeypatch.delenv('CUDA_HOME')
    monkeypatch.delenv('CUDA_PATH')
    environment = G1GraspPlannerClient()._clean_environment()
    assert 'CUDA_HOME' not in environment and 'CUDA_PATH' not in environment
