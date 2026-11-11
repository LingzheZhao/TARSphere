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
