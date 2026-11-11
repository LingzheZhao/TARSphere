import numpy as np
import pytest
from studies.fabric_joint_measurement import relative_live_pose


def test_common_root_and_world_motion_cancel_but_limb_motion_remains():
    paths = ['Root', 'Root/Pelvis', 'Root/Pelvis/Foot']
    matrices = [np.eye(4) for _ in paths]
    matrices[2][3, 0] = 0.4
    selected = {'Pelvis': paths[1], 'Foot': paths[2]}
    baseline = relative_live_pose(paths, paths, matrices, np.eye(4), selected)
    matrices[0][3, :3] = [20, -7, 3]
    world = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [15, 3, 0, 1]])
    moved = relative_live_pose(paths, paths, matrices, world, selected)
    assert np.allclose(moved['Foot'], baseline['Foot'])
    matrices[2][3, 2] = 0.2
    changed = relative_live_pose(paths, paths, matrices, world, selected)
    assert not np.allclose(changed['Foot'], baseline['Foot'])


def test_missing_dynamic_parent_is_rejected_even_if_selected_joint_present():
    paths = ['Root', 'Root/Pelvis', 'Root/Pelvis/Foot']
    with pytest.raises(LookupError, match='Missing live animation coverage for Root'):
        relative_live_pose(paths, paths[1:], [np.eye(4), np.eye(4)], np.eye(4),
                           {'Pelvis': paths[1], 'Foot': paths[2]})


def test_body_scale_preserves_physical_limb_displacement():
    paths = ['Root', 'Root/Pelvis', 'Root/Pelvis/Foot']
    selected = {'Pelvis': paths[1], 'Foot': paths[2]}
    matrices = [np.eye(4) for _ in paths]
    matrices[2][3, 0] = 0.4
    world = np.eye(4)
    baseline = relative_live_pose(paths, paths, matrices, world, selected)
    world[:3, :3] *= 2
    world[3, :3] = [7, -9, 2]
    doubled = relative_live_pose(paths, paths, matrices, world, selected)
    assert np.allclose(np.array(doubled['Foot'])[3, :3],
                       2 * np.array(baseline['Foot'])[3, :3])
    matrices[2][3, 2] = 0.2
    lifted_double = relative_live_pose(paths, paths, matrices, world, selected)
    lifted_single = relative_live_pose(paths, paths, matrices, np.eye(4), selected)
    double_delta = np.array(lifted_double['Foot'])[3, :3] - np.array(doubled['Foot'])[3, :3]
    single_delta = np.array(lifted_single['Foot'])[3, :3] - np.array(baseline['Foot'])[3, :3]
    assert np.allclose(double_delta, 2 * single_delta)
