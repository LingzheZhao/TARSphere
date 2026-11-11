from types import SimpleNamespace
import math
import pytest
from tarsphere.infra.control.behavior_human_executor import BehaviorHumanTasks


class Agent:
    def __init__(self):
        self.tasks = {}
        self.calls = []
    def is_task_running(self, task):
        return self.tasks[task] == 'running'
    def cancel_task(self, task):
        self.calls.append(('cancel', task))
        self.tasks[task] = 'failed'
    def move_to(self, **kwargs):
        self.calls.append(('move_to', kwargs))
        task = len(self.calls)
        self.tasks[task] = 'running'
        return task
    def idle(self, **kwargs):
        self.calls.append(('idle', kwargs))
        task = len(self.calls)
        self.tasks[task] = 'running'
        return task
    def get_task_status(self, task):
        return self.tasks[task]
    def get_world_translation(self):
        return (100., 200., 0.)


API = SimpleNamespace(BEHAVIOR_TASK_ID_INVALID=0,
    BehaviorTaskStatus=SimpleNamespace(RUNNING='running', SUCCEEDED='succeeded', FAILED='failed'))


def test_centimetre_scene_keeps_metric_goal_and_native_completion():
    agent = Agent()
    tasks = BehaviorHumanTasks(agent, API, meters_per_unit=.01)
    tasks.goto([1, 2, 0])
    command, args = agent.calls[-1]
    assert command == 'move_to'
    assert args['target'] == (100., 200., 0.)
    assert tasks.position_m() == [1., 2., 0.]
    # Being at the goal does not mean the native final turn has completed.
    assert not tasks.succeeded()
    agent.tasks[tasks.task_id] = 'succeeded'
    assert tasks.succeeded()


def test_idle_preempts_walk_without_old_task_completing_successor():
    agent = Agent()
    tasks = BehaviorHumanTasks(agent, API, meters_per_unit=1)
    old = tasks.goto([1, 0, 0])
    new = tasks.idle()
    assert agent.calls[-2] == ('cancel', old)
    agent.tasks[old] = 'succeeded'
    assert tasks.task_id == new
    assert not tasks.succeeded()
    agent.tasks[new] = 'failed'
    assert tasks.failed()


def test_invalid_goal_does_not_cancel_current_action():
    agent = Agent()
    tasks = BehaviorHumanTasks(agent, API, meters_per_unit=1)
    current = tasks.goto([1, 0, 0])
    with pytest.raises(ValueError):
        tasks.goto([math.nan, 0, 0])
    assert tasks.task_id == current
    assert agent.tasks[current] == 'running'


def test_idle_heading_uses_documented_world_direction():
    agent = Agent()
    tasks = BehaviorHumanTasks(agent, API, meters_per_unit=.01)
    tasks.idle(heading_deg=90)
    assert agent.calls[-1][1]['facing'] == pytest.approx((0, 1, 0))


def test_relative_limb_measurement_rejects_root_only_motion_as_articulation():
    import numpy as np
    from studies.human_behavior_diagnostic import relative_joint_matrices
    pelvis = np.eye(4)
    foot = np.eye(4)
    foot[3, :3] = [.2, .1, -.8]
    rigid = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[4,3,0,1.]])
    original = relative_joint_matrices(pelvis, {'foot':foot})['foot']
    root_only = relative_joint_matrices(pelvis @ rigid, {'foot':foot @ rigid})['foot']
    np.testing.assert_allclose(original, root_only)
    foot[3, 0] += .1
    articulated = relative_joint_matrices(pelvis @ rigid, {'foot':foot @ rigid})['foot']
    assert not np.allclose(original, articulated)


def test_canceled_task_is_not_left_ongoing():
    agent = Agent()
    tasks = BehaviorHumanTasks(agent, API, meters_per_unit=1)
    task = tasks.goto([1, 0, 0])
    agent.tasks[task] = 'canceled'
    assert tasks.failed()
    assert not tasks.succeeded()


def test_behavior_joint_readback_cancels_rotated_root_and_keeps_articulation():
    import numpy as np
    from tarsphere.infra.control.behavior_human_executor import behavior_limb_pose
    names = ['Pelvis', 'L_Thigh', 'R_Thigh', 'L_Foot', 'R_Foot']
    class Joints:
        angle = 0.
        foot_offset = 0.
        def get_joint_index(self, tag):
            return names.index(tag)
        def get_joint_world_transform(self, index, translation, rotation):
            local = np.array([.1 * index, .2, -.2 * index]) if index else np.zeros(3)
            if index == 3:
                local[0] += self.foot_offset
            c, s = math.cos(self.angle), math.sin(self.angle)
            world = np.array([[c,-s,0],[s,c,0],[0,0,1]]) @ local + [3,4,1]
            translation[:] = world.tolist()
            rotation[:] = [0,0,math.sin(self.angle/2),math.cos(self.angle/2)]
            return True
    agent = Joints()
    read = lambda: behavior_limb_pose(agent, dict(zip(names,names)), meters_per_unit=1,
        vector_factory=lambda size:[float('nan')]*size)['joints_relative_pelvis']
    before = read()
    agent.angle = math.pi / 2
    turned = read()
    for name in before:
        np.testing.assert_allclose(before[name], turned[name], atol=1e-12)
    agent.foot_offset = .15
    articulated = read()
    assert not np.allclose(articulated['L_Foot'], turned['L_Foot'])


def test_behavior_joint_failure_cannot_become_static_zero_pose():
    from tarsphere.infra.control.behavior_human_executor import behavior_limb_pose
    tags = {name:name for name in ['Pelvis','L_Thigh','R_Thigh','L_Foot','R_Foot']}
    agent = SimpleNamespace(get_joint_index=lambda tag:0,
        get_joint_world_transform=lambda index, translation, rotation:False)
    with pytest.raises(RuntimeError, match='world transform failed'):
        behavior_limb_pose(agent,tags,meters_per_unit=1,vector_factory=lambda size:[0]*size)
    agent.get_joint_world_transform = lambda index, translation, rotation:True
    with pytest.raises(ValueError, match='Invalid'):
        behavior_limb_pose(agent,tags,meters_per_unit=1,vector_factory=lambda size:[float('nan')]*size)


def test_control_rig_mapping_uses_sdk_conversion_and_keeps_raw_failure(monkeypatch):
    import sys
    from tarsphere.infra.control.behavior_human_executor import skeleton_joint_tags
    attr = lambda value: SimpleNamespace(Get=lambda: value)
    tags = ['pelvis-tag', 'foot-tag']
    monkeypatch.setitem(sys.modules, 'RetargetingSchema', SimpleNamespace(
        ControlRigAPI=lambda prim: SimpleNamespace(GetJointTagsAttr=lambda: attr(tags))))
    calls = []
    def convert(joints):
        calls.append(joints)
        return ['Pelvis', 'L_Foot']
    monkeypatch.setitem(sys.modules, 'omni.anim.retarget.core.scripts.utils', SimpleNamespace(convert_to_simple_joints=convert))
    skeleton = SimpleNamespace(GetPrim=lambda: SimpleNamespace(GetPath=lambda: '/Skeleton'),
        GetJointsAttr=lambda: attr(['Root/Pelvis', 'Root/Pelvis/L_Foot']))
    assert skeleton_joint_tags(skeleton) == {'Pelvis': 'pelvis-tag', 'L_Foot': 'foot-tag'}
    assert calls == [['Root/Pelvis', 'Root/Pelvis/L_Foot']]
    tags.clear()
    with pytest.raises(ValueError, match="raw_tags=\\[\\]"):
        skeleton_joint_tags(skeleton)


def test_measured_joint_tags_refreshes_and_serializes(monkeypatch):
    import json
    import tarsphere.infra.control.behavior_human_executor as module
    body = object.__new__(module.BehaviorHumanExecutor)
    body._behavior_skeletons = {'Nora': object()}
    values = iter([{}, {'Pelvis': 'actual-tag'}])
    monkeypatch.setattr(module, 'skeleton_joint_tags', lambda skeleton: next(values))
    assert body.measured_joint_tags('Nora')['value'] == {}
    receipt = body.measured_joint_tags('Nora')
    assert json.loads(json.dumps(receipt))['value'] == {'Pelvis': 'actual-tag'}


def test_goto_final_heading_waits_for_native_facing_and_preserves_units():
    agent = Agent()
    tasks = BehaviorHumanTasks(agent, API, meters_per_unit=.01)
    moving = tasks.goto([1, 2, 0], end_heading_deg=90)
    assert agent.calls[-1][1]['target'] == (100, 200, 0)
    assert not tasks.succeeded()
    agent.tasks[moving] = 'succeeded'
    assert not tasks.succeeded()
    assert tasks.command == 'goto'
    assert agent.calls[-1][0] == 'idle'
    assert agent.calls[-1][1]['facing'] == pytest.approx((0, 1, 0))
    agent.tasks[tasks.task_id] = 'succeeded'
    assert tasks.succeeded()
