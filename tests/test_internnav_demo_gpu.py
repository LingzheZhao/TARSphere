"""InternNav must preserve allocation identity when it initializes its env."""
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest


def load_configure(monkeypatch):
    config = ModuleType('internnav.configs.evaluator')
    config.EnvCfg = config.TaskCfg = object
    base = SimpleNamespace(Env=type('Env', (), {'register': staticmethod(lambda name: lambda cls: cls)}))
    env = ModuleType('internnav.env')
    env.base = base
    loader = ModuleType('internnav.env.utils.episode_loader')
    loader.ResumablePathKeyEpisodeloader = loader.generate_vln_episode = None
    for name, value in [('internnav.configs.evaluator', config), ('internnav.env', env),
                        ('internnav.env.utils.episode_loader', loader)]:
        monkeypatch.setitem(sys.modules, name, value)
    return runpy.run_path('third_party/InternNav/internnav/env/tarsphere_env.py')['_configure_visible_device']


def test_demo_preserves_verified_uuid_and_rejects_ordinal(monkeypatch):
    from tarsphere.infra.simulation import demo_gpu
    uuid = 'GPU-592eec6a-cc01-65a5-b880-fc4a40cd58c5'
    monkeypatch.setenv('TARSPHERE_DEMO_GPU_UUID', uuid)
    monkeypatch.setenv('TARSPHERE_KIT_ACTIVE_GPU', '2')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', uuid)
    monkeypatch.setattr(demo_gpu, 'lookup_physical_gpu_uuid', lambda index: uuid)
    configure = load_configure(monkeypatch)
    configure(0)
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES'] == uuid
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(RuntimeError, match='exact TARSPHERE_DEMO_GPU_UUID'):
        configure(0)


def test_legacy_ordinal_behavior(monkeypatch):
    monkeypatch.delenv('TARSPHERE_DEMO_GPU_UUID', raising=False)
    load_configure(monkeypatch)(3)
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
