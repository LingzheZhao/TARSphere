"""Finite Isaac 6.x skeletal-animation control; no agent or task evaluator.

Requires the pinned Isaac environment and TARSPHERE_BEHAVIOR_MOTION_LIBRARY.
python -m studies.human_behavior_diagnostic --config studies/human_behavior_diagnostic.yaml --output NEW_ROOT
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path


def relative_joint_matrices(pelvis_world, joints_world):
    """USD row-vector transforms: cancel the complete common rigid motion."""
    import numpy as np
    inverse = np.linalg.inv(np.asarray(pelvis_world, dtype=float))
    return {name: (np.asarray(matrix, dtype=float) @ inverse).tolist()
            for name, matrix in joints_world.items()}


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    phases = config['phases']
    total = sum(float(p['seconds']) for p in phases)
    if not phases or not math.isfinite(total) or not 0 < total <= 30 or any(not 0 < float(p['seconds']) <= 12 for p in phases):
        raise ValueError('Finite phase budgets must total at most 30 seconds, each at most 12')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'resolved.yaml').write_text(yaml.safe_dump(config))
    from tarsphere.infra.simulation.gpu_bootstrap import configure_cuda_allocator
    configure_cuda_allocator()
    import numpy as np
    from PIL import Image
    from tarsphere.bench.case_loader import BenchmarkCaseLoader
    from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv, RenderMode
    from tarsphere.infra.control.behavior_human_executor import BehaviorHumanExecutor, behavior_limb_pose
    from tarsphere.infra.control.robot_body_executor import RobotBodyExecutor
    from tarsphere.infra.control.human_native_motion_trace import json_value, measured
    from tarsphere.cli.teleop import _step_simulation
    episode = BenchmarkCaseLoader().load_file(config['episode'])
    env = IsaacAutonomousEnv(episode, human_motion_backend='animation',
        render_mode=RenderMode.DEMO, headless=True, enable_ego_sensors=True,
        enable_streaming=False, enable_human_ego_cameras=False,
        image_dir=str(args.output / 'images'))
    rows = []
    try:
        if not isinstance(env._human_body, BehaviorHumanExecutor):
            raise RuntimeError('This diagnostic requires the installed Isaac 6.x behavior animation API')
        env._human_body._capture_native_motion_trace = True
        env.reset(skip_navmesh=False)
        # Match worker startup: SimulationApp/Kit own USD before study queries.
        from pxr import Usd, UsdGeom, UsdSkel
        human_id = config['human_id']
        body = env._human_body
        stage = env._env.runner._world.stage
        dt = float(env._env.runner._world.get_physics_dt())
        def skeleton():
            root = stage.GetPrimAtPath(body._character_skelroot_paths[human_id])
            cache = UsdSkel.Cache()
            cache.Populate(UsdSkel.Root(root), Usd.PrimDefaultPredicate)
            skeletons = [p for p in Usd.PrimRange(root) if p.IsA(UsdSkel.Skeleton)]
            if len(skeletons) != 1:
                raise RuntimeError('Expected exactly one behavior skeleton')
            query = cache.GetSkelQuery(UsdSkel.Skeleton(skeletons[0]))
            time = Usd.TimeCode.Default()
            matrices = query.ComputeJointWorldTransforms(UsdGeom.XformCache(time))
            joints = {str(name).rsplit('/',1)[-1]: np.array(matrix).tolist()
                      for name, matrix in zip(query.GetJointOrder(), matrices, strict=True)}
            pelvis = next((name for name in ('Pelvis','Hip') if name in joints), None)
            if pelvis is None:
                raise RuntimeError('Skeleton has neither measured Pelvis nor Hip joint')
            selected = {name: joints[name] for name in ('L_Thigh','R_Thigh','L_Foot','R_Foot') if name in joints}
            if len(selected) != 4:
                raise RuntimeError(f'Required limb joints absent; actual joints: {list(joints)}')
            return dict(source='USD evaluated skeleton; Fabric synchronization must be checked against video',
                pelvis_joint=pelvis, pelvis_world=joints[pelvis], joints_world=selected,
                joints_relative_pelvis=relative_joint_matrices(joints[pelvis], selected))
        for phase in phases:
            if phase['command'] == 'idle_facing':
                tasks = body._behavior_tasks[human_id]
                tasks.idle(heading_deg=float(phase['heading_deg']))
                state = body.get_state(human_id)
                state.active_action = ''
                state.action_status = 'idle'
                state.animation_state = 'idle_facing'
                state.target_position = None
                body._goto_waypoint_queues.pop(human_id, None)
                body._pending_command_ids[human_id] = {str(tasks.task_id)}
                admission = dict(command='idle_facing', heading_deg=float(phase['heading_deg']), task_id=tasks.task_id)
            else:
                action = {'command':phase['command']}
                if 'target' in phase:
                    action['target'] = phase['target']
                admission = asdict(env.apply_human_action(human_id, action))
            remaining = int(round(float(phase['seconds']) / dt))
            while remaining > 0:
                before = env._physics_steps
                state = body.get_state(human_id)
                tag_mapping = body.measured_joint_tags(human_id)
                row = dict(phase=phase['name'], admission=admission,
                    physics_step=before, state=asdict(state),
                    behavior_pose=measured(lambda: body._poll_character_runtime_pose(human_id)),
                    skeleton=measured(skeleton),
                    behavior_joint_tag_mapping=tag_mapping,
                    behavior_joints=measured(lambda: behavior_limb_pose(
                        body._behavior_tasks[human_id].agent,
                        tag_mapping.get('value', {}),
                        meters_per_unit=body._behavior_tasks[human_id].meters_per_unit)))
                proxy_path = env._human_collision_proxy_paths.get(human_id)
                row['collision_proxy'] = measured(lambda: dict(path=proxy_path,
                    world_transform=np.array(UsdGeom.Xformable(stage.GetPrimAtPath(proxy_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).tolist()))
                obs = env.observe()
                image = f'{len(rows):04d}.png'
                Image.fromarray(np.asarray(obs['rgb'])[...,:3].astype(np.uint8)).save(args.output / image)
                row['rgb'] = image
                if env._physics_steps != before:
                    raise RuntimeError('Observation capture advanced physics')
                rows.append(row)
                (args.output / 'trace.json').write_text(json.dumps(rows, default=json_value, indent=2))
                requested = min(60, remaining)
                terminal = _step_simulation(env, env_action=RobotBodyExecutor.zero_velocity_action(), physics_ticks_per_sample=requested)
                advanced = env._physics_steps - before
                if advanced <= 0:
                    raise RuntimeError('Physics did not advance')
                remaining -= advanced
                if terminal:
                    raise RuntimeError('Robot terminated during independent animation control')
        (args.output / 'motion_trace.json').write_text(json.dumps(body.get_replay_log(), default=json_value, indent=2))
        (args.output / 'complete.json').write_text(json.dumps(dict(kind='skeletal_animation_control', task_success=None, physics_steps=env._physics_steps)))
    except BaseException as exc:
        # Isaac teardown can terminate the process; persist failure beforehand.
        import traceback
        (args.output / 'failure.json').write_text(json.dumps(dict(
            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
            physics_steps=env._physics_steps), indent=2))
        raise
    finally:
        env.close(skip_kit_cleanup=True)


if __name__ == '__main__':
    main()
