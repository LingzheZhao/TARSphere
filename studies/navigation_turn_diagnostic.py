"""Four reset-separated physical turn probes; no benchmark score or agent.

Run in the pinned Isaac environment: python -m studies.navigation_turn_diagnostic
--config studies/navigation_turn_diagnostic.yaml --output NEW_ROOT.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
import traceback
import sys
from pathlib import Path
import yaml


def validate(config):
    if not math.isfinite(config['settle_seconds']) or not 0 < config['settle_seconds'] <= 3:
        raise ValueError('settle_seconds must be in (0, 3]')
    if config['turn_degrees'] != [-60, 60] or len(config['locations']) != 2:
        raise ValueError('Exactly two locations and matched +/-60 degree turns required')
    for pose in config['locations'].values():
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            raise ValueError('Locations require finite x, y, yaw')


def execute_decision(env, decision, trace):
    """Use the episode's public move_body route, preserving every native hook."""
    original = env._on_physics_tick
    start_tick = env._physics_steps

    def capture_tick(timing, step_index):
        original(timing, step_index)
        if timing == 'after':
            position, orientation = env._runtime_g1_robot().get_robot_base().get_pose()
            from dataclasses import asdict
            from tarsphere.infra.control.robot_body_executor import RobotBodyExecutor
            trace.append(dict(physics_tick=start_tick + step_index + 1,
                position=position,
                yaw_rad=RobotBodyExecutor.observation_yaw({'orientation': orientation}),
                dynamics=asdict(env._g1_body_dynamics()),
                contacts=env._robot_contact_monitor.snapshot()
                    if env._robot_contact_monitor else None))
    env._on_physics_tick = capture_tick
    from tarsphere.infra.episode.errors import RobotFallError
    try:
        try:
            return env.step(decision)
        except RobotFallError as exc:
            env._obs = exc.observation
            return exc.observation, True, {**exc.details, 'physical_fall': True, 'exception': str(exc),
                              'exception_type': type(exc).__name__}
    finally:
        env._on_physics_tick = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('studies/navigation_turn_diagnostic.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate(config)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'resolved.yaml').write_text(yaml.safe_dump(config))
    from tarsphere.infra.simulation.gpu_bootstrap import configure_cuda_allocator
    configure_cuda_allocator()
    import numpy as np
    from PIL import Image
    from tarsphere.bench.case_loader import BenchmarkCaseLoader
    from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv, RenderMode
    from tarsphere.infra.control.robot_body_executor import RobotBodyExecutor
    from tarsphere.harness.actions.router import locomotion_control_profile, duration_physics_steps
    from tarsphere.infra.control.human_native_motion_trace import json_value
    episode = BenchmarkCaseLoader().load_file(config['episode'])
    env = IsaacAutonomousEnv(episode, human_motion_backend='kinematic', render_mode=RenderMode.DEMO,
        headless=True, enable_ego_sensors=True, enable_streaming=False,
        enable_human_ego_cameras=False, image_dir=str(args.output / 'images'))
    from pxr import Usd, UsdGeom, UsdPhysics
    def state():
        robot = env._live_robots()[0]
        art = robot.articulation.unwrap()
        return dict(physics_tick=env._physics_steps, pose=env._robot_pose(),
            dynamics=asdict(env._g1_body_dynamics()), joint_names=list(art.dof_names),
            joint_positions=art.get_joint_positions(), joint_velocities=art.get_joint_velocities(),
            contacts=env._robot_contact_monitor.snapshot() if env._robot_contact_monitor else None)
    def capture(label):
        tick = env._physics_steps
        obs = env.observe()
        Image.fromarray(np.asarray(obs['rgb'])[..., :3].astype(np.uint8)).save(args.output / f'{label}.png')
        if tick != env._physics_steps:
            raise RuntimeError('Capture advanced physics')
    completed = []
    try:
        env.reset(scene_prepare_only=True, skip_navmesh=True)
        for label, (x, y, yaw) in config['locations'].items():
            for angle in config['turn_degrees']:
                name = f'{label}_{angle:+d}'
                current = episode.model_copy(deep=True)
                current.robot.start_position = [x, y, episode.robot.start_position[2]]
                current.robot.start_orientation = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
                env.load_episode(current)
                env.reset(skip_navmesh=True)
                row = dict(case=name, benchmark_score=None, requested_pose=[x,y,yaw], reset=state())
                row['settle_trace'] = []
                before_tick = env._physics_steps
                _, terminal, details = execute_decision(env,
                    {'action': 'wait', 'seconds': config['settle_seconds']}, row['settle_trace'])
                row['settle_ticks'] = env._physics_steps - before_tick
                row['settle_execution_details'] = details
                row['settled'] = state()
                capture(name + '_before')
                # Conservative collider AABB distances are reported, not labelled verified free space.
                stage = env._env.runner._world.stage
                import carb
                settings = carb.settings.get_settings()
                row['usd_sync_settings'] = {key: settings.get(key) for key in (
                    '/physics/fabricEnabled', '/physics/updateToUsd')}
                row['collider_aabb_measurement'] = 'USD authored transforms; native Fabric synchronization unverified'
                cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render','proxy'])
                boxes = []
                settled_xy = np.asarray(row['settled']['pose']['position'][:2], dtype=float)
                robot_root = env._robot_root_prim_path()
                if not robot_root:
                    raise RuntimeError('Robot collider root unavailable')
                for prim in stage.Traverse():
                    if str(prim.GetPath()) == robot_root or str(prim.GetPath()).startswith(robot_root + '/'):
                        continue
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    if bound.IsEmpty():
                        continue
                    lo, hi = np.array(bound.GetMin()), np.array(bound.GetMax())
                    if hi[2] < .15 or lo[2] > 1.6:
                        continue
                    distance = float(np.linalg.norm(np.maximum(np.maximum(lo[:2]-settled_xy, settled_xy-hi[:2]), 0)))
                    boxes.append(dict(path=str(prim.GetPath()), bbox_min=lo, bbox_max=hi, xy_distance_m=distance))
                row['collider_aabbs'] = sorted(boxes, key=lambda b:b['xy_distance_m'])
                profile = locomotion_control_profile('move_body', params={'delta':[0,0,angle]})
                row['profile'] = asdict(profile)
                command = env._robot_body.command_for_decision(
                    {'action': 'move_body', 'delta': [0, 0, angle]})
                row['command_physics_budget'] = command.physics_step_budget
                if command.physics_step_budget > duration_physics_steps(8):
                    raise ValueError('Turn correction budget exceeds eight seconds')
                if not terminal:
                    row['turn_trace'] = []
                    before_tick = env._physics_steps
                    _, terminal, details = execute_decision(env, {'action': 'move_body', 'delta': [0, 0, angle]}, row['turn_trace'])
                    row['turn_ticks'] = env._physics_steps - before_tick
                    row['execution_details'] = details
                row['after'] = state()
                row['terminated_or_truncated'] = terminal
                row['physical_fall'] = bool(row['settle_execution_details'].get('physical_fall') or row.get('execution_details', {}).get('physical_fall'))
                start, end = row['settled']['pose'], row['after']['pose']
                row['actual_yaw_delta_rad'] = math.atan2(math.sin(end['yaw_rad']-start['yaw_rad']), math.cos(end['yaw_rad']-start['yaw_rad']))
                row['actual_world_translation_m'] = [b-a for a,b in zip(start['position'], end['position'])]
                row['fall_diagnostic'] = dict(
                    minimum_pelvis_height_m=min(t['dynamics']['pelvis_height_m'] for t in row['settle_trace'] + row.get('turn_trace', [])),
                    maximum_gravity_tilt_deg=max(t['dynamics']['gravity_tilt_deg'] for t in row['settle_trace'] + row.get('turn_trace', [])),
                    terminated_or_truncated=bool(terminal))
                row['clearance_classification'] = 'unverified_candidate'
                capture(name + '_after')
                (args.output / f'{name}.json').write_text(json.dumps(row, default=json_value, indent=2))
                completed.append(name)
        (args.output / 'complete.json').write_text(json.dumps({'cases': completed, 'case_count': len(completed)}))
    except BaseException as exc:
        failure = dict(exception_type=type(exc).__name__, exception=str(exc),
                       traceback=traceback.format_exc(), completed_cases=completed)
        (args.output / 'failure.json').write_text(json.dumps(failure, indent=2))
        print(failure['traceback'], file=sys.stderr, flush=True)
        raise
    finally:
        env.close(skip_kit_cleanup=True)


if __name__ == '__main__':
    main()
