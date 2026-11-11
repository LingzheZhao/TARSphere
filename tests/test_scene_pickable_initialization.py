"""Scene loading must normalize pickables before simulator body registration."""
import sys
from types import SimpleNamespace as NS
import pytest
from tarsphere.simulator.core.scene.isaacsim.scene import IsaacsimScene, prepare_scene_pickables


def test_load_normalizes_before_static_scan_and_base_task_returns(monkeypatch):
    calls = []
    root_path = '/World/env_0/scene'
    class Prim:
        def __init__(self, path, valid=True, proxy=False):
            self.path, self.valid, self.proxy = path, valid, proxy
        def GetPath(self): return self.path
        def GetStage(self): return stage
        def IsValid(self): return self.valid
        def IsInstanceProxy(self): return self.proxy
    root = Prim(root_path)
    bottle = Prim(root_path+'/Bottle')
    stage = NS(GetPrimAtPath=lambda p: bottle if p == bottle.path else Prim(p, valid=False))
    monkeypatch.setattr('tarsphere.infra.manipulation.pickable_physics.configure_dynamic_pickable',
        lambda prim, **kw: calls.append(('normalize', prim.path, kw)))
    monkeypatch.setitem(sys.modules, 'isaacsim.core.utils.prims', NS(create_prim=lambda *a, **kw: root))
    monkeypatch.setitem(sys.modules, 'pxr', NS(Usd=NS(PrimRange=NS(AllPrims=lambda p: []))))
    monkeypatch.setattr('tarsphere.simulator.core.scene.isaacsim.scene.validate_scene_file', lambda *a: ('scene.usda', root_path))
    monkeypatch.setattr('tarsphere.simulator.core.scene.isaacsim.scene.prepare_static_scene_bodies',
        lambda *a: calls.append(('static',)))
    scene = IsaacsimScene.__new__(IsaacsimScene)
    scene.load(NS(scene_asset_path='scene.usda',scene_position=[0,0,0],scene_scale=[1,1,1],
        scene_static=True,scene_dynamic_body_paths=[bottle.path,bottle.path,root_path+'/LaterInjected','/World/elsewhere']),0,[0,0,0])
    calls.append(('base_task_can_wrap',))
    assert calls == [('normalize', bottle.path, {'kinematic': True}), ('static',), ('base_task_can_wrap',)]


def test_instance_proxy_root_fails_before_registration(monkeypatch):
    prim = NS(IsValid=lambda:True,IsInstanceProxy=lambda:True)
    root=NS(GetPath=lambda:'/World/env_0/scene',GetStage=lambda:NS(GetPrimAtPath=lambda p:prim))
    with pytest.raises(RuntimeError,match='instance proxy'):
        prepare_scene_pickables(root,['/World/env_0/scene/Bottle'])
