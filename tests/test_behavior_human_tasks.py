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


def test_installed_1104_signature_and_unsupported_heading_preserves_walk():
    class InstalledAgent(Agent):
        def move_to(self, target, auto_brake=True):
            return super().move_to(target=target, auto_brake=auto_brake)
    agent = InstalledAgent()
    tasks = BehaviorHumanTasks(agent, API, meters_per_unit=.01)
    task = tasks.goto([1, 2, 0])
    calls = list(agent.calls)
    with pytest.raises(NotImplementedError, match='no facing parameter'):
        tasks.goto([2, 2, 0], end_heading_deg=90)
    assert tasks.task_id == task
    assert agent.calls == calls


def test_original_custom_action_preserves_native_task_and_completion():
    agent = Agent()
    agent.get_custom_action_names = lambda: ['TypeKeyboard', 'PushButton']
    def custom_action(name, *, duration):
        agent.calls.append(('custom_action', name, duration))
        agent.tasks[71] = 'running'
        return 71
    agent.custom_action = custom_action
    tasks = BehaviorHumanTasks(agent, API, meters_per_unit=.01)
    previous = tasks.idle()
    assert tasks.custom_action('TypeKeyboard', duration=2.5) == 71
    assert agent.calls[-2:] == [('cancel', previous), ('custom_action', 'TypeKeyboard', 2.5)]
    assert not tasks.succeeded()
    agent.tasks[previous] = 'succeeded'
    assert not tasks.succeeded()
    agent.tasks[71] = 'succeeded'
    assert tasks.succeeded()


@pytest.mark.parametrize('duration', [-1, math.nan, math.inf])
def test_invalid_custom_duration_preserves_current_task(duration):
    agent = Agent()
    tasks = BehaviorHumanTasks(agent, API, meters_per_unit=1)
    current = tasks.idle()
    with pytest.raises(ValueError):
        tasks.custom_action('TypeKeyboard', duration=duration)
    assert tasks.task_id == current
    assert agent.tasks[current] == 'running'


def test_unregistered_custom_action_preserves_current_task():
    agent = Agent()
    agent.get_custom_action_names = lambda: []
    tasks = BehaviorHumanTasks(agent, API, meters_per_unit=1)
    current = tasks.idle()
    with pytest.raises(RuntimeError, match='not registered'):
        tasks.custom_action('TypeKeyboard', duration=0)
    assert tasks.task_id == current
    assert agent.tasks[current] == 'running'


def test_custom_executor_waits_for_real_sdk_success():
    from tarsphere.infra.control.behavior_human_executor import BehaviorHumanExecutor
    agent = Agent()
    agent.tasks[7] = 'running'
    tasks = BehaviorHumanTasks(agent, API, meters_per_unit=1)
    tasks.task_id = 7
    body = object.__new__(BehaviorHumanExecutor)
    body._scene_bound_ids = set()
    body._behavior_tasks = {'Nora': tasks}
    body._pending_command_ids = {'Nora': {'7'}}
    state = SimpleNamespace(active_action='type_keyboard', action_status='ongoing')
    body._poll_command_completion('Nora', state)
    assert state.action_status == 'ongoing'
    assert body._pending_command_ids
    agent.tasks[7] = 'succeeded'
    body._poll_command_completion('Nora', state)
    assert state.action_status == 'succeed'
    assert not body._pending_command_ids


def test_author_clip_import_uses_sdk_kwargs_and_private_storage(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from tarsphere.infra.control.behavior_human_executor import BehaviorHumanExecutor
    # Mock only the SDK boundary; command argument names match shipped 110.4 tests.
    modules = {}
    names = ['carb', 'carb.settings', 'omni', 'omni.usd', 'omni.kit',
             'omni.kit.commands', 'omni.anim.behavior.core.scripts.user_storage', 'pxr']
    for name in names:
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    modules['carb'].settings = modules['carb.settings']
    modules['omni'].usd = modules['omni.usd']
    modules['omni'].kit = modules['omni.kit']
    modules['omni.kit'].commands = modules['omni.kit.commands']
    key = '/persistent/exts/omni.anim.behavior.ui/userCustomActionFolder'
    modules[names[-2]].SETTING_USER_CUSTOM_ACTION_FOLDER = key
    values = {key: 'previous-user-storage'}
    modules['carb.settings'].get_settings = lambda: SimpleNamespace(
        get=values.get, set=lambda k,v: values.__setitem__(k,v))
    modules['pxr'].Sdf = SimpleNamespace(Path=lambda p: p)
    prim = SimpleNamespace(IsValid=lambda: True,
        GetAttribute=lambda key: SimpleNamespace(Get=lambda: current_name[0]))
    current_name = ['']
    stage = SimpleNamespace(GetPrimAtPath=lambda p: prim, Load=lambda p: None)
    modules['omni.usd'].get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    calls = []
    def execute(command, **kwargs):
        calls.append((command, kwargs, values[key]))
        current_name[0] = kwargs['action_name']
        return True, ['/World/Characters/HumanMotionLibrary/CustomActions/' + current_name[0]]
    modules['omni.kit.commands'].execute = execute
    root = tmp_path / 'Characters'
    skeleton = root / 'biped_demo' / 'biped_demo_meters.usd'
    skeleton.parent.mkdir(parents=True)
    skeleton.touch()
    animations = tmp_path / 'Animations'
    animations.mkdir()
    for clip in ['type_keyboard', 'push_button']:
        (animations / f'{clip}.skelanim.usd').touch()
    monkeypatch.setenv('TARSPHERE_LOCAL_SCRATCH', str(tmp_path))
    body = object.__new__(BehaviorHumanExecutor)
    body._character_assets_path = str(root)
    body._behavior_custom_storage = None
    body._behavior_library_path = 'staged-library.usd'
    body._import_author_custom_actions()
    assert values[key] == 'previous-user-storage'
    assert len(calls) == 2
    for (command, args, storage), (name, clip) in zip(calls,
            [('TypeKeyboard', 'type_keyboard'), ('PushButton', 'push_button')]):
        assert command == 'ImportCustomActionCommand'
        assert args == dict(motion_library_prim_path='/World/Characters/HumanMotionLibrary',
            rig_name='Human', skeleton_asset=str(skeleton), action_name=name,
            animation_asset_urls=[str(animations / f'{clip}.skelanim.usd')], action_type='Custom')
        assert storage == body._behavior_custom_storage
        assert storage.startswith(str(tmp_path / 'tarsphere-custom-actions-'))


@pytest.mark.parametrize('terminal, expected', [('succeeded', 'succeed'), ('failed', 'fail')])
def test_custom_body_step_elapsed_time_cannot_complete_running_sdk_task(terminal, expected):
    from tarsphere.infra.control.behavior_human_executor import BehaviorHumanExecutor
    agent = Agent()
    agent.tasks[7] = 'running'
    tasks = BehaviorHumanTasks(agent, API, meters_per_unit=1)
    tasks.task_id = 7
    body = object.__new__(BehaviorHumanExecutor)
    body._scene_bound_ids = set()
    body._behavior_tasks = {'Nora': tasks}
    body._pending_command_ids = {'Nora': {'7'}}
    state = SimpleNamespace(active_action='push_button', action_status='ongoing')
    body._states = {'Nora': state}
    body._poll_character_position = lambda *args: None
    body._sync = lambda *args: None
    # Step far beyond a requested two-second hold. Only the SDK owns completion.
    for _ in range(3):
        body.step(100.0)
        assert state.action_status == 'ongoing'
        assert body._pending_command_ids == {'Nora': {'7'}}
    agent.tasks[7] = terminal
    body.step(100.0)
    assert state.action_status == expected
    assert not body._pending_command_ids
    if expected == 'fail':
        assert 'failed' in state.last_error
