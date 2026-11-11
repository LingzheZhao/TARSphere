"""The diagnostic must not submit work to an occupied or wrong SceneServer."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location('grasp_diagnostic',
    Path(__file__).parents[1] / 'examples/grasp_system_diagnostic.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.mark.parametrize('active,scene', [('other-run', 'expected'), (None, 'wrong')])
def test_identity_failure_preserves_evidence_without_submission(tmp_path, monkeypatch, active, scene):
    monkeypatch.setattr(cli, 'resolve_scene', lambda *a, **kw: SimpleNamespace(scene_id='expected'))
    identity = SimpleNamespace(active_request_id=active, scene=scene,
        model_dump_json=lambda **kw: json.dumps({'active_request_id': active, 'scene': scene}))
    monkeypatch.setattr(cli, 'SceneServerClient', lambda **kw: SimpleNamespace(identity=lambda: identity))
    def forbidden(*a, **kw):
        pytest.fail('must not prepare or submit a trial after identity mismatch')
    monkeypatch.setattr(cli, 'resolve_grasp_system_trial', forbidden)
    output = tmp_path / 'attempt'
    assert cli.main(['--assets-root', '/unused', '--scene-id', 'expected',
        '--request-id', 'attempt-a1', '--output-root', str(output)]) == 2
    assert json.loads((output/'failure.json').read_text())['type'] == 'RuntimeError'
    assert (output/'server_identity.json').exists()


@pytest.mark.parametrize('option,value', [('--max-steps', '0'), ('--timeout-s', 'inf')])
def test_budget_rejected_before_creating_output(tmp_path, option, value):
    output = tmp_path/'attempt'
    with pytest.raises(SystemExit):
        cli.main(['--assets-root', '/unused', '--scene-id', 'expected',
            '--request-id', 'a1', '--output-root', str(output), option, value])
    assert not output.exists()
