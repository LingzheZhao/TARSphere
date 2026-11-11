"""The table entrypoint uses normal public skill execution without Golden DTOs."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('table_grasp_diagnostic',
    Path(__file__).parents[1] / 'examples/table_grasp_diagnostic.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_normal_run_contract_preserves_raw_failure_and_closes(tmp_path, monkeypatch):
    model = SimpleNamespace(model_dump_json=lambda **kw: '{}')
    monkeypatch.setattr(cli, 'prepare_episode', lambda *a: (model, model))
    calls = []
    raw = {'failure': {'reason': 'retention lost'}, 'video': '/raw/video.mp4',
        'trace': [{'grasp_skill': {'retention': 'finite_coulomb_friction', 'transport_ready': False}}]}
    class Server:
        def __init__(self, **kw):
            assert kw['agent_mode'] == 'grasp-certification'
        def bootstrap(self, episode): calls.append('bootstrap')
        def run_episode(self, episode, **kw):
            assert kw == {'request_id': 'table-a1', 'record_video': True, 'max_steps': 12}
            calls.append('run')
            return raw
        def close(self): calls.append('close')
    monkeypatch.setitem(sys.modules, 'tarsphere.infra.simulation.scene_server', SimpleNamespace(SceneServer=Server))
    monkeypatch.setitem(sys.modules, 'tarsphere.infra.simulation.gpu_bootstrap', SimpleNamespace(configure_cuda_allocator=lambda: None))
    monkeypatch.setitem(sys.modules, 'tarsphere.infra.simulation.scene_server_preflight', SimpleNamespace(preload_support_surface_geometry=lambda: {'version': 'test'}))
    output = tmp_path / 'attempt'
    assert cli.main(['--assets-root', str(tmp_path), '--grounded-scene', 'prepared.json',
        '--request-id', 'table-a1', '--output-root', str(output)]) == 1
    assert calls == ['bootstrap', 'run', 'close']
    assert json.loads((output/'result.json').read_text()) == raw
    samples = json.loads((output/'reported_measurements.json').read_text())
    assert samples['holding_samples'] == []
    assert samples['transport_ready_samples'] == [False]
    assert samples['retention_samples'] == ['finite_coulomb_friction']
    assert 'certification_status' not in samples
