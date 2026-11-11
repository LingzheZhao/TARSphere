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
    monkeypatch.setattr('tarsphere.bench.grounding.resolve.resolve_scene', lambda *a, **kw: SimpleNamespace(scene_id='expected'))
    identity = SimpleNamespace(active_request_id=active, scene=scene,
        model_dump_json=lambda **kw: json.dumps({'active_request_id': active, 'scene': scene}))
    monkeypatch.setattr(cli, 'SceneServerClient', lambda **kw: SimpleNamespace(identity=lambda: identity))
    def forbidden(*a, **kw):
        pytest.fail('must not prepare or submit a trial after identity mismatch')
    monkeypatch.setattr(cli, 'resolve_grasp_system_trial', forbidden)
    output = tmp_path / 'attempt'
    assert cli.main(['--http', '--assets-root', '/unused', '--scene-id', 'expected',
        '--request-id', 'attempt-a1', '--output-root', str(output)]) == 2
    assert json.loads((output/'failure.json').read_text())['type'] == 'RuntimeError'
    assert (output/'server_identity.json').exists()


@pytest.mark.parametrize('option,value', [('--max-steps', '0'), ('--timeout-s', 'inf')])
def test_budget_rejected_before_creating_output(tmp_path, option, value):
    output = tmp_path/'attempt'
    with pytest.raises(SystemExit):
        cli.main(['--http', '--assets-root', '/unused', '--scene-id', 'expected',
            '--request-id', 'a1', '--output-root', str(output), option, value])
    assert not output.exists()


def test_inprocess_adapter_forwards_original_certification_request():
    calls = []
    server = SimpleNamespace(
        run_episode=lambda episode, **kw: calls.append((episode, kw)) or {'raw': True},
        abort_active=lambda request: calls.append(request))
    adapter = cli.InProcessCertificationClient(server)
    assert adapter.run_certification_episode('episode', certification_stage='A',
        request_id='a1', record_video=True, max_steps=60) == {'raw': True}
    assert calls == [('episode', dict(certification_stage='A', request_id='a1',
        record_video=True, max_steps=60))]
    adapter.abort('a1')
    assert calls[-1] == 'a1'


def test_inprocess_requires_cpu_prepared_scene_before_output(tmp_path):
    output = tmp_path/'attempt'
    with pytest.raises(SystemExit):
        cli.main(['--assets-root', '/unused', '--scene-id', 'expected',
            '--request-id', 'a1', '--output-root', str(output)])
    assert not output.exists()


def test_cached_scene_identity_mismatch_never_imports_scene_server(tmp_path, monkeypatch):
    import sys
    scene_file = tmp_path/'grounded.json'
    scene_file.write_text('{}')
    monkeypatch.setattr(cli.GroundedScene, 'model_validate_json',
        lambda text: SimpleNamespace(scene_id='wrong'))
    monkeypatch.setitem(sys.modules, 'tarsphere.infra.simulation.scene_server', None)
    output = tmp_path/'attempt'
    assert cli.main(['--assets-root', '/unused', '--scene-id', 'expected',
        '--grounded-scene', str(scene_file), '--request-id', 'a1',
        '--output-root', str(output)]) == 2
    failure = json.loads((output/'failure.json').read_text())
    assert failure['type'] == 'ValueError'
    assert 'Grounded scene identity' in failure['message']
