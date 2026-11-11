"""Isaac 6.x behavior-agent task bridge for skeletal Human motion.

The task interface follows the shipped behavior.core tests and IRA runtime
states. It does not write character transforms or synthesize joint motion.
"""
from __future__ import annotations

import math
from typing import Any


def uses_behavior_animation_sdk() -> bool:
    """Select the installed 6.x API when the legacy people extension is absent."""
    import importlib.util
    import os
    from pathlib import Path
    spec = importlib.util.find_spec('isaacsim')
    roots = [Path(p) / 'extscache' for p in (spec.submodule_search_locations or [])] if spec else []
    extra = os.environ.get('TARSPHERE_EXTRA_EXTENSION_ROOT')
    if extra:
        roots.append(Path(extra))
    legacy = any(any(root.glob('omni.anim.people*')) for root in roots)
    behavior = any(any(root.glob('omni.anim.behavior.core*')) for root in roots)
    return behavior and not legacy


class BehaviorHumanTasks:
    """Translate author GoTo/Idle/custom-action semantics to an existing behavior agent.

    Agent creation, animation-library loading and clock advancement are owned
    by the environment. Completion is read from the actual task, including its
    final facing phase; proximity alone never completes a task.
    """

    def __init__(self, agent: Any, api: Any, *, meters_per_unit: float):
        scale = float(meters_per_unit)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('meters_per_unit must be finite and positive')
        self.agent = agent
        self.api = api
        self.meters_per_unit = scale
        self.task_id = None
        self.command = None

    def _replace(self, task_id, command):
        if task_id == self.api.BEHAVIOR_TASK_ID_INVALID:
            raise RuntimeError(f'Isaac behavior agent rejected {command}')
        self.task_id = task_id
        self.command = command
        return task_id

    def cancel(self):
        if self.task_id is not None and self.agent.is_task_running(self.task_id):
            self.agent.cancel_task(self.task_id)
        self.task_id = None
        self.command = None

    def goto(self, position_m, *, end_heading_deg=None):
        target = tuple(float(v) for v in position_m)
        if len(target) != 3 or not all(math.isfinite(v) for v in target):
            raise ValueError('GoTo requires a finite XYZ target in metres')
        kwargs = {'target': tuple(v / self.meters_per_unit for v in target)}
        if end_heading_deg is not None:
            heading = math.radians(float(end_heading_deg))
            if not math.isfinite(heading):
                raise ValueError('GoTo heading must be finite')
            raise NotImplementedError('Installed Isaac 110.4 move_to has no facing parameter')
        self.cancel()
        return self._replace(self.agent.move_to(**kwargs), 'goto')

    def idle(self, *, heading_deg=None):
        kwargs = {}
        if heading_deg is not None:
            heading = math.radians(float(heading_deg))
            if not math.isfinite(heading):
                raise ValueError('Idle heading must be finite')
            kwargs['facing'] = (math.cos(heading), math.sin(heading), 0.0)
        self.cancel()
        return self._replace(self.agent.idle(**kwargs), 'idle')

    def custom_action(self, name, *, duration):
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('Custom action duration must be finite and nonnegative')
        if name not in self.agent.get_custom_action_names():
            raise RuntimeError(f'Isaac behavior custom action is not registered: {name}')
        self.cancel()
        return self._replace(self.agent.custom_action(name, duration=duration), name)

    def status(self):
        if self.task_id is None:
            return None
        return self.agent.get_task_status(self.task_id)

    def succeeded(self):
        return self.task_id is not None and self.status() == self.api.BehaviorTaskStatus.SUCCEEDED

    def failed(self):
        return self.task_id is not None and self.status() not in (
            self.api.BehaviorTaskStatus.RUNNING, self.api.BehaviorTaskStatus.SUCCEEDED)

    def position_m(self):
        return [float(v) * self.meters_per_unit for v in self.agent.get_world_translation()]


def behavior_limb_pose(agent, joint_tags, *, meters_per_unit, vector_factory=None):
    """Read actual behavior-agent joints using its output-argument binding.

    joint_tags maps real skeleton leaf names to ControlRigAPI joint tags.
    Unknown tags, failed queries, untouched outputs and invalid quaternions
    remain errors instead of becoming a static zero skeleton.
    """
    import numpy as np
    if vector_factory is None:
        import carb
        vector_factory = lambda size: (carb.Float3 if size == 3 else carb.Float4)(*([float('nan')] * size))
    pelvis = next((name for name in ('Pelvis', 'Hip') if name in joint_tags), None)
    if pelvis is None:
        raise LookupError(f'No Pelvis or Hip joint tag; actual mapping: {joint_tags}')
    names = [pelvis, 'L_Thigh', 'R_Thigh', 'L_Foot', 'R_Foot']
    transforms = {}
    raw = {}
    for name in names:
        if name not in joint_tags:
            raise LookupError(f'Missing tag for actual skeleton joint {name}; mapping: {joint_tags}')
        tag = joint_tags[name]
        index = agent.get_joint_index(tag)
        if index < 0:
            raise LookupError(f'Isaac behavior agent has no joint tag {tag!r} for {name}')
        translation, rotation = vector_factory(3), vector_factory(4)
        if not agent.get_joint_world_transform(index, translation, rotation):
            raise RuntimeError(f'Isaac behavior joint world transform failed for {name} ({tag})')
        position = np.array([float(translation[i]) for i in range(3)]) * meters_per_unit
        quaternion = np.array([float(rotation[i]) for i in range(4)])
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) <= 0:
            raise ValueError(f'Invalid Isaac behavior transform for {name}')
        x, y, z, w = quaternion / np.linalg.norm(quaternion)
        # USD row-vector matrix, quaternion is carb.Float4 XYZW.
        rotation_column = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        matrix = np.eye(4)
        matrix[:3,:3] = rotation_column.T
        matrix[3,:3] = position
        transforms[name] = matrix
        raw[name] = dict(tag=tag, joint_index=index, position_world_m=position.tolist(), rotation_world_xyzw=quaternion.tolist())
    inverse_pelvis = np.linalg.inv(transforms[pelvis])
    return dict(source='isaac_behavior_agent_joint_world_transform', pelvis_joint=pelvis,
        joints=raw, joints_relative_pelvis={name: (transforms[name] @ inverse_pelvis).tolist()
            for name in names if name != pelvis})


def skeleton_joint_tags(skeleton):
    """Read the SDK ControlRigAPI mapping without guessing joint aliases."""
    import RetargetingSchema
    from omni.anim.retarget.core.scripts.utils import convert_to_simple_joints
    rig = RetargetingSchema.ControlRigAPI(skeleton.GetPrim())
    tags = list(rig.GetJointTagsAttr().Get() or [])
    joints = list(skeleton.GetJointsAttr().Get() or [])
    simple_joints = list(convert_to_simple_joints(joints))
    if len(tags) != len(simple_joints):
        raise ValueError(f'ControlRig mapping unavailable: skeleton={skeleton.GetPrim().GetPath()}, '
                         f'raw_joints={list(map(str, joints))!r}, simple_joints={list(map(str, simple_joints))!r}, '
                         f'raw_tags={list(map(str, tags))!r}')
    return {str(joint): str(tag) for joint, tag in zip(simple_joints, tags, strict=True) if str(tag)}



# Keep the author's action admission, ordered routes, physical-state contracts,
# and feedback objects. Only the Isaac animation execution layer changes.
from tarsphere.infra.control.anim_people_executor import AnimPeopleBodyExecutor
from tarsphere.infra.control.human_body import HumanBodyState


class BehaviorHumanExecutor(AnimPeopleBodyExecutor):
    """Isaac 6.x skeletal animation; GoTo, Idle, Listen, speech and imported author custom clips.

    Other author commands fail explicitly until their behavior-library actions
    and physical semantics have been exercised. The legacy executor remains
    available on SDKs that ship omni.anim.people.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._behavior_tasks = {}
        self._behavior_library_path = None
        self._behavior_skeletons = {}
        self._behavior_custom_storage = None
        self._behavior_custom_actions_enabled = False

    def author_character_animation_graph(self):
        # The 6.x motion library and BehaviorAgentAPI replace Biped_Setup.
        import os
        from pathlib import Path
        path = Path(os.environ.get('TARSPHERE_BEHAVIOR_MOTION_LIBRARY', '')).expanduser()
        if not path.is_file():
            raise FileNotFoundError('TARSPHERE_BEHAVIOR_MOTION_LIBRARY must name the staged Isaac HumanMotionLibrary.usd')
        self._behavior_library_path = str(path.resolve())
        self._behavior_custom_actions_enabled = os.environ.get('TARSPHERE_BEHAVIOR_CUSTOM_ACTIONS') == '1'
        if self._behavior_custom_actions_enabled:
            self._import_author_custom_actions()

    def _import_author_custom_actions(self):
        # SDK 110.4 ImportCustomActionCommand retargets copies into user storage;
        # neither the original clips nor the staged library are modified.
        import os
        import tempfile
        from pathlib import Path
        import carb.settings
        import omni.usd
        import omni.kit.commands
        from omni.anim.behavior.core.scripts.user_storage import SETTING_USER_CUSTOM_ACTION_FOLDER
        from pxr import Sdf
        root = Path(self._character_assets_path).expanduser()
        skeleton = root / 'biped_demo' / 'biped_demo_meters.usd'
        animations = root.parent / 'Animations'
        clips = {name: animations / f'{clip}.skelanim.usd'
                 for name, clip in [('TypeKeyboard', 'type_keyboard'), ('PushButton', 'push_button')]}
        missing = [str(p) for p in [skeleton, *clips.values()] if not p.is_file()]
        if missing:
            raise FileNotFoundError('Original custom-action resources missing: ' + ', '.join(missing))
        if self._behavior_custom_storage is None:
            scratch = os.environ.get('TARSPHERE_LOCAL_SCRATCH')
            self._behavior_custom_storage = tempfile.mkdtemp(prefix='tarsphere-custom-actions-', dir=scratch)
        settings = carb.settings.get_settings()
        previous = settings.get(SETTING_USER_CUSTOM_ACTION_FOLDER)
        try:
            settings.set(SETTING_USER_CUSTOM_ACTION_FOLDER, self._behavior_custom_storage)
            stage = omni.usd.get_context().get_stage()
            library_path = Sdf.Path('/World/Characters/HumanMotionLibrary')
            library = stage.GetPrimAtPath(library_path)
            if not library:
                library = stage.DefinePrim(library_path)
                if not library.GetPayloads().AddPayload(self._behavior_library_path):
                    raise RuntimeError('Cannot compose the staged behavior motion library')
            stage.Load(library_path)
            for name, clip in clips.items():
                ok, _ = omni.kit.commands.execute('ImportCustomActionCommand',
                    motion_library_prim_path=library_path, rig_name='Human',
                    skeleton_asset=str(skeleton), action_name=name,
                    animation_asset_urls=[str(clip)], action_type='Custom')
                # The command can log an error and return None despite successful
                # dispatch. Registration is checked on each real agent below.
                if not ok:
                    raise RuntimeError(f'Isaac custom action import failed: {name}')
        finally:
            settings.set(SETTING_USER_CUSTOM_ACTION_FOLDER, previous or '')


    def despawn_all(self):
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        for tasks in self._behavior_tasks.values():
            tasks.cancel()
        for path in self._character_prim_paths.values():
            stage.RemovePrim(path)
        self._behavior_tasks.clear()
        self._character_prim_paths.clear()
        self._character_skelroot_paths.clear()

    def reset(self, humans):
        import omni.usd
        import omni.kit.commands
        import omni.anim.behavior.core as bh
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel
        self.require_character_runtime_prepared()
        self.despawn_all()
        self._pump_app_updates(count=2)
        self._states.clear()
        self._scene_bound_ids.clear()
        self._pending_command_ids.clear()
        self._replay_log.clear()
        self._last_injected_goto.clear()
        self._goto_waypoint_queues = {}
        self._physics_step = None
        self._last_native_motion_trace_tick.clear()
        self._spawn_epoch += 1
        stage = omni.usd.get_context().get_stage()
        scale = UsdGeom.GetStageMetersPerUnit(stage)
        for human in humans:
            human_id = str(human.id)
            position = [float(v) for v in human.start_position]
            self._states[human_id] = HumanBodyState(human_id=human_id, position=position)
            if getattr(human, 'scene_person_prim_path', None):
                self._bind_scene_person(human, human.scene_person_prim_path)
                continue
            position = [float(v) for v in human.start_position]
            path = f'/World/Characters/{human_id}/behavior_{self._spawn_epoch}'
            asset = self._resolve_character_usd(self._resolve_character_name(getattr(human, 'character_name', None)))
            # Match the author's named actor parent. Compose and load the exact
            # catalog payload before the SDK applies BehaviorAgentAPI; its
            # convenience command traverses immediately after payload creation.
            UsdGeom.Xform.Define(stage, f'/World/Characters/{human_id}')
            character = UsdGeom.Xform.Define(stage, path).GetPrim()
            if not character.GetPayloads().AddPayload(str(asset)):
                raise RuntimeError(f'Cannot compose original character payload for {human_id}: {asset}')
            stage.Load(path)
            if not UsdGeom.XformCommonAPI(character).SetTranslate(Gf.Vec3d(*(v/scale for v in position))):
                raise RuntimeError(f'Cannot author initial character placement for {human_id}')
            roots = [p for p in Usd.PrimRange(character) if p.IsA(UsdSkel.Root)]
            if len(roots) != 1:
                raise RuntimeError(f'Expected one loaded catalog SkelRoot for {human_id}, got {len(roots)}')
            skeletons = [p for p in Usd.PrimRange(roots[0]) if p.IsA(UsdSkel.Skeleton)]
            if len(skeletons) != 1 or not UsdSkel.Skeleton(skeletons[0]).GetJointsAttr().Get():
                raise RuntimeError(f'Expected one populated catalog Skeleton for {human_id}')
            library_path = Sdf.Path('/World/Characters/HumanMotionLibrary')
            library = stage.GetPrimAtPath(library_path)
            if not library:
                library = stage.DefinePrim(library_path)
                if not library.GetPayloads().AddPayload(self._behavior_library_path):
                    raise RuntimeError('Cannot compose the staged behavior motion library')
            stage.Load(library_path)
            import omni.anim.behavior.schema as behavior_schema
            if not library.IsA(behavior_schema.BEHAVIOR_MOTION_LIBRARY):
                raise RuntimeError('Loaded motion-library payload is not an Isaac BehaviorMotionLibrary')
            # This is the same public SDK command used inside
            # CreateBehaviorAgentCommand after its payload traversal succeeds.
            ok, _ = omni.kit.commands.execute('ApplyBehaviorAgentAPICommand',
                skelroot_prim_paths=[roots[0].GetPath()],
                motion_library_prim_path=library_path,
                motion_library_skeleton_rig='Human')
            if not ok or not roots[0].HasAPI(behavior_schema.BEHAVIOR_AGENT_API):
                raise RuntimeError(f'Isaac ApplyBehaviorAgentAPICommand failed for {human_id}')
            if list(roots[0].GetRelationship('behavior:motionLibrary').GetTargets()) != [library_path]:
                raise RuntimeError(f'Isaac behavior motion-library binding is missing for {human_id}')
            skelroot = str(roots[0].GetPath())
            self._character_prim_paths[human_id] = path
            self._character_skelroot_paths[human_id] = skelroot
            skeletons = [p for p in Usd.PrimRange(roots[0]) if p.IsA(UsdSkel.Skeleton)]
            if len(skeletons) != 1:
                raise RuntimeError(f'Expected one behavior skeleton for {human_id}')
            from tarsphere.infra.control.human_native_motion_trace import measured
            self._behavior_skeletons[human_id] = UsdSkel.Skeleton(skeletons[0])
            agent = None
            for _ in range(120):
                agent = bh.acquire_interface().get_agent(skelroot)
                if agent is not None:
                    break
                self._pump_app_updates(count=1)
            if agent is None:
                raise RuntimeError(f'Isaac behavior agent did not register {human_id} after 120 setup updates')
            missing_actions = {'TypeKeyboard', 'PushButton'} - set(agent.get_custom_action_names())
            if self._behavior_custom_actions_enabled and missing_actions:
                raise RuntimeError(f'Isaac agent lacks imported original custom actions: {sorted(missing_actions)}')
            tasks = BehaviorHumanTasks(agent, bh, meters_per_unit=scale)
            self._behavior_tasks[human_id] = tasks
            # Episode initialization only: the SDK's test_agent_reset uses
            # reset(target=..., facing=carb.Float3(...)) for authored placement.
            # Runtime movement and the diagnostic heading phase still animate.
            import carb
            desired_yaw = float(getattr(human, 'start_yaw', 0))
            if not math.isfinite(desired_yaw):
                raise ValueError('Initial Human heading must be finite')
            if not agent.reset(target=carb.Float3(*(v / scale for v in position)),
                    facing=carb.Float3(math.cos(desired_yaw), math.sin(desired_yaw), 0.0)):
                raise RuntimeError(f'Isaac behavior initial pose reset failed for {human_id}')
            self._pump_app_updates(count=1)
            direction = agent.get_facing_direction()
            yaw = math.atan2(float(direction[1]), float(direction[0]))
            error = abs((yaw - desired_yaw + math.pi) % (2 * math.pi) - math.pi)
            if not math.isfinite(error) or error > math.radians(0.5):
                raise RuntimeError(f'Initial Isaac behavior heading mismatch for {human_id}; error_rad={error}')
            from tarsphere.infra.control.physical_contracts import HUMAN_INITIAL_POSE_TOLERANCE_M
            if math.dist(tasks.position_m()[:2], position[:2]) > HUMAN_INITIAL_POSE_TOLERANCE_M:
                raise RuntimeError(f'Initial Isaac behavior pose reset left {human_id} outside its authored spawn tolerance')
            tasks.idle()
            state = self._states[human_id]
            state.body_status = 'behavior_agent_ready'
            state.metadata['animation_backend'] = 'isaac_behavior_agent'
            self._poll_character_position(human_id, state)
            self._sync(human_id)

    def apply_action(self, human_id, action):
        command = str(action.get('command') or '')
        if command not in {'goto', 'idle', 'listen', *self.SPEECH_COMMANDS, *self.CUSTOM_ANIMATED_COMMANDS}:
            return self._failed(human_id, command,
                f'Isaac behavior animation action {command!r} is not yet verified in the loaded motion library')
        if command in self.CUSTOM_ANIMATED_COMMANDS and not self._behavior_custom_actions_enabled:
            return self._failed(human_id, command,
                'Original custom clips require TARSPHERE_BEHAVIOR_CUSTOM_ACTIONS=1 during setup')
        if command == 'goto' and action.get('end_rot') not in (None, '_'):
            return self._failed(human_id, command, 'Installed Isaac 110.4 move_to has no facing parameter')
        return super().apply_action(human_id, action)

    def _inject_command(self, human_id, command):
        parts = str(command).split()
        tasks = self._behavior_tasks[human_id]
        if len(parts) >= 5 and parts[1] == 'GoTo':
            task = tasks.goto([float(v) for v in parts[2:5]],
                end_heading_deg=float(parts[5]) if len(parts) > 5 and parts[5] != '_' else None)
        elif len(parts) >= 2 and parts[1] == 'Idle':
            task = tasks.idle()
        elif len(parts) == 3 and parts[1] in self.CUSTOM_ANIMATED_COMMANDS.values():
            task = tasks.custom_action(parts[1], duration=float(parts[2]))
        else:
            raise RuntimeError(f'Unsupported Isaac behavior command: {command}')
        self._pending_command_ids[human_id] = {str(task)}

    def _poll_character_runtime_pose(self, human_id):
        if human_id in self._scene_bound_ids:
            return super()._poll_character_runtime_pose(human_id)
        tasks = self._behavior_tasks[human_id]
        direction = tasks.agent.get_facing_direction()
        return tasks.position_m(), math.atan2(float(direction[1]), float(direction[0])), {
            'pose_source': 'isaac_behavior_agent', 'behavior_task_id': tasks.task_id,
            'behavior_task_status': str(tasks.status())}

    def _poll_command_completion(self, human_id, state):
        if human_id in self._scene_bound_ids:
            return super()._poll_command_completion(human_id, state)
        tasks = self._behavior_tasks[human_id]
        if tasks.succeeded():
            self._pending_command_ids.pop(human_id, None)
            if state.active_action in self.CUSTOM_ANIMATED_COMMANDS:
                state.action_status = 'succeed'
        elif tasks.failed():
            self._pending_command_ids.pop(human_id, None)
            state.action_status = 'fail'
            state.last_error = f'Isaac behavior agent task ended without success: {tasks.status()}'

    def _start_ambient_look_around(self, human_id, state):
        # No claim that Idle reproduces the author's LookAround action.
        self._behavior_tasks[human_id].idle()
        state.animation_state = 'idle'

    def measured_joint_tags(self, human_id):
        from tarsphere.infra.control.human_native_motion_trace import measured
        return measured(lambda: skeleton_joint_tags(self._behavior_skeletons[human_id]))

    def poll_native_motion_trace(self, *, observation_boundary, force=False):
        if not self._capture_native_motion_trace or self._physics_step is None:
            return
        from tarsphere.infra.control.human_native_motion_trace import measured
        for human_id, tasks in self._behavior_tasks.items():
            tick = self._physics_step
            state = self._states[human_id]
            period = 8 if state.action_status == 'ongoing' else 240
            previous = self._last_native_motion_trace_tick.get(human_id)
            if not force and previous is not None and tick - previous < period:
                continue
            tag_mapping = self.measured_joint_tags(human_id)
            self._replay_log.append(dict(event='human_behavior_motion_snapshot', physics_step=tick,
                human_id=human_id, observation_boundary=observation_boundary,
                sampling_period_ticks=period, body_state=state.to_observation(),
                behavior_readback=measured(lambda: dict(position_m=tasks.position_m(),
                    facing_direction=tasks.agent.get_facing_direction(), task_id=tasks.task_id,
                    task_status=str(tasks.status()))),
                joint_readback=measured(lambda: behavior_limb_pose(tasks.agent,
                    tag_mapping.get('value', {}),
                    meters_per_unit=tasks.meters_per_unit)),
                skeleton_joint_tag_mapping=tag_mapping))
            self._last_native_motion_trace_tick[human_id] = tick

    def poll_animation_graph(self, *, observation_boundary):
        # Graph-state readers for Biped_Setup do not apply to BehaviorAgentAPI.
        self.poll_native_motion_trace(observation_boundary=observation_boundary)
