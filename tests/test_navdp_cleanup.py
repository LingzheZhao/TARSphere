"""Real child processes exercise termination and unreaped zombie cleanup."""
import subprocess
import sys
import time
import pytest
from tarsphere.cli import evaluate

@pytest.mark.parametrize('already_exited', [False, True])
def test_cleanup_owned_process(monkeypatch, tmp_path, already_exited):
    monkeypatch.setenv('TARSPHERE_XNAVDP_STATE_DIR', str(tmp_path))
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', 'eval.src.policy_server', '--port', '19000'])
    path = evaluate.navdp_pid_file(19000)
    path.write_text(str(child.pid))
    try:
        if already_exited:
            child.terminate()
            for _ in range(200):
                if evaluate._process_identity(child.pid)[0] == 'Z':
                    break
                time.sleep(.01)
            assert evaluate._process_identity(child.pid)[0] == 'Z'
        evaluate.stop_navdp(19000)
        assert not path.exists()
    finally:
        child.wait(timeout=5)


def test_live_unrelated_process_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setenv('TARSPHERE_XNAVDP_STATE_DIR', str(tmp_path))
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    evaluate.navdp_pid_file(19000).write_text(str(child.pid))
    try:
        with pytest.raises(RuntimeError, match='Refusing'):
            evaluate.stop_navdp(19000)
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=5)
