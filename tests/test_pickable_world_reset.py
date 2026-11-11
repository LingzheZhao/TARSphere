"""Episode reset must use the runner's existing view and preserve true reset."""
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
from tarsphere.infra.manipulation.pickable_physics import reset_rigid_body_to_authored_pose


@pytest.mark.parametrize('kinematic', [False, True])
@pytest.mark.parametrize('failure', [None, 'missing_view', 'bad_readback'])
def test_reset_reuses_world_view_and_reads_back_pose(monkeypatch, kinematic, failure):
    pose = np.array([[1, 2, 3, 0, 0, 0, 1]], dtype=np.float32)
    calls = []
    body = NS(count=1, max_shapes=2,
        set_transforms=lambda p, i: calls.append(('pose', p.copy())),
        set_kinematic_targets=lambda p, i: calls.append(('target', p.copy())),
        set_velocities=lambda v, i: calls.append(('velocity', v.copy())),
        get_transforms=lambda: pose.copy() + (1 if failure == 'bad_readback' else 0))
    view = NS(create_rigid_body_view=lambda path: body)
    convert_calls = []
    def convert(data, *, device, dtype):
        convert_calls.append((device, dtype))
        # Isaac 6.2 numpy.tensor.as_type supports only these names.
        if dtype in ("float32", "bool", "int32", "int64", "uint8"):
            return np.asarray(data).astype(dtype)
        return None
    def resolve_indices(indices, count, device=None):
        # Copied logic from Isaac 6.2 core.utils.numpy.tensor.resolve_indices.
        result = indices
        if isinstance(indices, list):
            result = np.array(indices)
        if indices is None:
            result = np.arange(count)
        return result
    def record(name, value, indices):
        assert indices is not None
        np.testing.assert_array_equal(indices, [0])
        assert indices.dtype.kind == "i"
        calls.append((name, value.copy()))
    body.set_transforms = lambda p, i: record("pose", p, i)
    body.set_kinematic_targets = lambda p, i: record("target", p, i)
    body.set_velocities = lambda v, i: record("velocity", v, i)
    world = NS(physics_sim_view=view, device='cpu',
        backend_utils=NS(convert=convert, resolve_indices=resolve_indices, to_numpy=np.asarray))
    prim = NS(IsValid=lambda: True, HasAPI=lambda api: True)
    quaternion = NS(GetImaginary=lambda: (0, 0, 0), GetReal=lambda: 1)
    transform = NS(GetTranslation=lambda: (1, 2, 3),
        GetRotation=lambda: NS(GetQuat=lambda: quaternion))
    physics = NS(RigidBodyAPI=lambda p: NS(GetKinematicEnabledAttr=lambda: NS(Get=lambda: kinematic)))
    monkeypatch.setitem(sys.modules, 'pxr', NS(Gf=NS(Transform=lambda m: transform),
        Usd=NS(TimeCode=NS(Default=lambda: None)),
        UsdGeom=NS(Xformable=lambda p: NS(ComputeLocalToWorldTransform=lambda t: None)),
        UsdPhysics=physics))
    monkeypatch.setitem(sys.modules, 'omni', NS())
    monkeypatch.setitem(sys.modules, 'omni.physx', NS(
        get_physx_simulation_interface=lambda: NS(flush_changes=lambda: None)))
    if failure == 'missing_view':
        world.physics_sim_view = None
    if failure is not None:
        with pytest.raises(RuntimeError, match='initialized World' if failure == 'missing_view' else 'disagrees with USD'):
            reset_rigid_body_to_authored_pose(NS(GetPrimAtPath=lambda path: prim), '/bottle', world=world)
        if failure == 'missing_view':
            assert calls == []
        return
    reset_rigid_body_to_authored_pose(NS(GetPrimAtPath=lambda path: prim), '/bottle', world=world)
    assert [name for name, value in calls] == ['pose', 'target' if kinematic else 'velocity']
    np.testing.assert_array_equal(calls[0][1], pose)
    if not kinematic:
        np.testing.assert_array_equal(calls[1][1], np.zeros((1, 6)))
    assert all(dtype != 'uint32' for device, dtype in convert_calls)
    assert ('cpu', 'float32') in convert_calls
