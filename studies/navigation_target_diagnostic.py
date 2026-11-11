"""Offline target visibility diagnostic; resets are not navigation or task success.

Run under the provisioned Isaac 6.2 environment with --output NEW_ROOT.
No agent, dialogue policy, or benchmark evaluator is installed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('studies/navigation_target_diagnostic.yaml'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from tarsphere.infra.simulation.gpu_bootstrap import configure_cuda_allocator
    configure_cuda_allocator()
    from tarsphere.bench.case_loader import BenchmarkCaseLoader
    from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv, RenderMode
    from tarsphere.infra.perception.teleop_capture import _NativeCapture
    from tarsphere.infra.control.human_native_motion_trace import json_value
    import numpy as np
    from PIL import Image

    import yaml
    config = yaml.safe_load(args.config.read_text())
    (args.output / 'resolved.yaml').write_text(yaml.safe_dump(config))
    episode = BenchmarkCaseLoader().load_file(config['episode'])
    target_record = next(item for item in episode.world_knowledge['objects']
                         if item['object_id'] == config['target_object_id'])
    target = target_record['object_id']
    poses = [(item['label'], item.get('xy_yaw')) for item in config['poses']]
    env = IsaacAutonomousEnv(episode, human_motion_backend='kinematic',
        render_mode=RenderMode.DEMO, headless=True, enable_ego_sensors=True,
        enable_streaming=False, enable_human_ego_cameras=False,
        image_dir=str(args.output / 'images'))
    from pxr import UsdGeom, Usd, UsdPhysics
    records = []
    try:
        env.reset(scene_prepare_only=True, skip_navmesh=True)
        for label, pose in poses:
            current = episode.model_copy(deep=True)
            if pose:
                x, y, yaw = pose
                current.robot.start_position = [x, y, episode.robot.start_position[2]]
                current.robot.start_orientation = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
            env.load_episode(current)
            env.reset(skip_navmesh=True)
            stage = env._env.runner._world.stage
            import carb
            from isaacsim.core.simulation_manager import SimulationManager
            sync = {'fabric_enabled': bool(SimulationManager.is_fabric_enabled()),
                    'update_to_usd': bool(carb.settings.get_settings().get('/physics/updateToUsd'))}
            if sync['fabric_enabled'] or not sync['update_to_usd']:
                raise RuntimeError('Diagnostic requires Fabric disabled and USD synchronization enabled')
            prim = stage.GetPrimAtPath(target_record['runtime_prim'])
            matches = [prim] if prim.IsValid() else []
            row = {'label': label, 'diagnostic_reset_only': True,
                   'requested_start': current.robot.model_dump(),
                   'actual_pose': env._robot_pose(), 'target': target,
                   'sync': sync, 'target_catalog': target_record,
                   'projection_limit': 'Corner-in-frame is geometry only; neither visibility nor evaluator success',
                   'target_matches': [str(p.GetPath()) for p in matches]}
            before = env._physics_steps
            observation = env.observe()
            Image.fromarray(np.asarray(observation['rgb'])[..., :3].astype(np.uint8)).save(args.output / f'{label}_rgb.png')
            np.save(args.output / f'{label}_depth.npy', np.array(observation['depth'], copy=True))
            row['evaluator_visibility'] = env.evaluate_visual_object_visibility(object_ids=(target,))
            row['camera_params'] = observation['camera_params']
            row['depth_status'] = observation.get('depth_status')
            (args.output / f'{label}_measurement.json').write_text(json.dumps(row, default=json_value, indent=2))
            runtime = _NativeCapture(env)
            try:
                row['camera'] = runtime.prepare((640, 480), True)
                rgb, (ids, labels) = runtime.render()
                np.save(args.output / f'{label}_instance_ids.npy', ids)
                row['instance_labels'] = labels
                row['target_visible_pixels'] = sum(int((ids == int(k)).sum()) for k, v in labels.items() if target in str(v))
                Image.fromarray(rgb).save(args.output / f'{label}_segmentation_rgb.png')
                (args.output / f'{label}_measurement.json').write_text(json.dumps(row, default=json_value, indent=2))
            finally:
                runtime.close()
            row['target_geometry'] = []
            camera = UsdGeom.Camera(stage.GetPrimAtPath(env._agent_camera_path())).GetCamera(Usd.TimeCode.Default())
            view_projection = camera.frustum.ComputeViewMatrix() * camera.frustum.ComputeProjectionMatrix()
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
            for prim in matches:
                box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                from itertools import product
                from pxr import Gf
                corners = [Gf.Vec3d(*v) for v in product(*zip(box.GetMin(), box.GetMax()))]
                projected = [view_projection.Transform(p) for p in corners]
                row['target_geometry'].append({'path': str(prim.GetPath()),
                    'world_bbox_min': list(box.GetMin()), 'world_bbox_max': list(box.GetMax()),
                    'world_transform': str(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())),
                    'projected_ndc_corners': [list(p) for p in projected],
                    'projected_in_frame_corner': any(abs(p[0]) <= 1 and abs(p[1]) <= 1 and -1 <= p[2] <= 1 for p in projected),
                    'physics_prims': [{'path': str(p.GetPath()), 'rigid_body': p.HasAPI(UsdPhysics.RigidBodyAPI),
                        'collision': p.HasAPI(UsdPhysics.CollisionAPI),
                        'attributes': {a.GetName(): {'type': str(a.GetTypeName()), 'value': str(a.Get())} for a in p.GetAttributes() if a.GetName().startswith('physics:')}}
                        for p in Usd.PrimRange(prim)]})
            row['physics_counter_before_capture'] = before
            row['physics_counter_after_capture'] = env._physics_steps
            if before != env._physics_steps:
                raise RuntimeError('Capture advanced physics counter')
            row['task_success'] = None
            records.append(row)
            (args.output / 'observations.json').write_text(json.dumps(records, default=json_value, indent=2))
        (args.output / 'completion.json').write_text(json.dumps({'completed_poses': len(records), 'expected_poses': len(poses)}))
    except BaseException as exc:
        import traceback
        import sys
        (args.output / 'failure.json').write_text(json.dumps({'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}))
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        env.close(skip_kit_cleanup=True)


if __name__ == '__main__':
    main()
