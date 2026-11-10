"""Scripted component check, never a TU benchmark run.

Invoke with --config YAML --output NEW_DIRECTORY. Configuration keys: episode,
max_simulation_seconds (<=120), settle_seconds, question, facts (author policy
fact_id/text records), outbound_target, return_target, arrival_tolerance_m,
capture_interval_seconds. Support endpoint uses the author's explicit environment
configuration. No robot agent, milestone evaluator, or task-score writer is installed.
Human routes use the author's occupancy planner.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text())
    budget = float(config['max_simulation_seconds'])
    settle = float(config['settle_seconds'])
    interval = float(config['capture_interval_seconds'])
    tolerance = float(config['arrival_tolerance_m'])
    if not (math.isfinite(budget) and 0 < budget <= 120 and 0 <= settle < budget
            and math.isfinite(interval) and interval > 0
            and math.isfinite(tolerance) and tolerance > 0):
        parser.error('Invalid finite time budget, interval, or arrival tolerance')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'resolved.yaml').write_text(yaml.safe_dump(config))
    from tarsphere.infra.simulation.gpu_bootstrap import configure_cuda_allocator
    configure_cuda_allocator()
    from tarsphere.bench.case_loader import BenchmarkCaseLoader
    from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv, RenderMode
    from tarsphere.cli.teleop import _step_simulation
    from tarsphere.humans.llm_policy import HumanDialoguePolicy
    from tarsphere.infra.perception.teleop_capture import TeleopCapture
    from tarsphere.infra.control.robot_body_executor import RobotBodyExecutor
    from tarsphere.infra.control.human_native_motion_trace import json_value

    episode = BenchmarkCaseLoader().load_file(config['episode'])
    env = IsaacAutonomousEnv(episode, human_motion_backend='kinematic',
        render_mode=RenderMode.DEMO, headless=True, enable_ego_sensors=True, enable_streaming=False,
        enable_human_ego_cameras=False,
        image_dir=str(args.output / 'images'))
    failed = False
    capture = None
    summary = {'kind': 'scripted_component_check', 'benchmark_score': None,
               'backend': 'kinematic root motion with static idle skeleton',
               'contacts': 'available in captured execution_safety when the author monitor is active',
               'terminal': 'initialization', 'route_completions': []}
    with (args.output / 'trace.jsonl').open('x') as trace:
        def record(kind, **data):
            trace.write(json.dumps({'kind': kind, **data}, default=json_value) + '\n')
            trace.flush()
        try:
            env.reset(scene_prepare_only=True, skip_navmesh=True)
            env.load_episode(episode)
            env.reset(skip_navmesh=True)
            world = env._env.runner._world
            dt = float(world.get_physics_dt())
            start = env._physics_steps
            start_world_time = float(world.current_time)
            max_ticks = int(budget / dt)
            capture = TeleopCapture(env, args.output / 'captures', actor_layers=True,
                resolution=tuple(config.get('capture_resolution', [640, 480])))
            import carb
            from isaacsim.core.simulation_manager import SimulationManager
            sync = {'fabric_enabled': bool(SimulationManager.is_fabric_enabled()),
                    'update_to_usd': bool(carb.settings.get_settings().get('/physics/updateToUsd'))}
            record('runtime_sync', **sync)
            if sync['fabric_enabled'] or not sync['update_to_usd']:
                raise RuntimeError('Diagnostic requires Fabric disabled and USD synchronization enabled')
            def elapsed():
                return (env._physics_steps - start) * dt
            def world_elapsed():
                return float(world.current_time) - start_world_time
            def snapshot(label):
                import numpy as np
                from PIL import Image
                before = env._physics_steps
                observation = env.observe()
                rgb = np.array(observation['rgb'], copy=True)
                depth = np.array(observation['depth'], copy=True)
                Image.fromarray(rgb[..., :3].astype(np.uint8)).save(args.output / (label + '_rgb.png'))
                np.save(args.output / (label + '_depth.npy'), depth)
                metadata = capture.capture(label)
                record('capture', label=label, counter_time_seconds=elapsed(),
                       world_elapsed_seconds=world_elapsed(),
                       rgb_file=label + '_rgb.png', depth_file=label + '_depth.npy',
                       pose=observation['pose'], camera_params=observation['camera_params'],
                       depth_status=observation.get('depth_status'), capture=metadata)
                if env._physics_steps != before:
                    raise RuntimeError('Observation/capture advanced the author step counter')
            snapshot('initial')
            phase = 'settle'
            target = None
            next_capture = interval
            while env._physics_steps - start < max_ticks and world_elapsed() < budget:
                remaining_ticks = min(
                    max_ticks - (env._physics_steps - start),
                    int(max(0, budget - world_elapsed()) / dt),
                )
                if remaining_ticks == 0:
                    summary['terminal'] = 'world_time_horizon'
                    break
                if phase == 'settle' and elapsed() >= settle:
                    answer = HumanDialoguePolicy(npc_context={'name': 'Nora'}).answer(
                        question=config['question'], facts=config['facts'], history=[])
                    record('support_answer_selection', answer=answer.model_dump())
                    distance = math.dist(env._robot_position()[:2],
                                         env._human_body.position('Nora')[:2])
                    if answer.message and distance <= 2.5:
                        result = env.apply_human_action('Nora',
                            {'command': 'speak', 'message': answer.message})
                        record('speech_admission', result=asdict(result), distance_m=distance,
                               author_last_utterance=env.last_utterance,
                               robot_heard_event='unavailable: no episode runner inbox installed')
                    else:
                        record('speech_not_delivered', distance_m=distance,
                               reason='proximity_failure' if distance > 2.5 else 'empty_answer')
                    phase = 'outbound'
                    target = config['outbound_target']
                    record('route_admission', phase=phase, result=asdict(
                        env.apply_human_action('Nora', {'command': 'goto', 'target': target})))
                state = env._human_body.get_state('Nora')
                if (target is not None and state.action_status == 'succeed'
                        and math.dist(state.position, target) <= tolerance):
                    record('route_completion', phase=phase, counter_time_seconds=elapsed(), state=asdict(state))
                    summary['route_completions'].append(phase)
                    if phase == 'outbound':
                        phase = 'return'
                        target = config['return_target']
                        record('route_admission', phase=phase, result=asdict(
                            env.apply_human_action('Nora', {'command': 'goto', 'target': target})))
                    else:
                        target = None
                        summary['terminal'] = 'both_routes_completed'
                        break
                record('state', counter_time_seconds=elapsed(), physics_step=env._physics_steps,
                       world_elapsed_seconds=world_elapsed(),
                       robot_pose=env._robot_pose(), nora=asdict(state),
                       author_last_utterance=env.last_utterance,
                       execution_safety=(env._robot_contact_monitor.snapshot()
                           if env._robot_contact_monitor is not None else None))
                terminal = _step_simulation(env, env_action=RobotBodyExecutor.zero_velocity_action(),
                    physics_ticks_per_sample=min(60, remaining_ticks))
                if terminal:
                    summary['terminal'] = 'author_terminated_or_truncated'
                    break
                if elapsed() >= next_capture:
                    snapshot('periodic_%06d' % env._physics_steps)
                    next_capture += interval
            else:
                summary['terminal'] = 'physical_time_horizon'
            snapshot('final')
            summary['counter_time_seconds'] = elapsed()
            summary['world_elapsed_seconds'] = world_elapsed()
        except Exception as exc:
            summary['terminal'] = type(exc).__name__
            record('failure', exception_type=type(exc).__name__, message=str(exc))
            failed = True
        finally:
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
            if capture is not None:
                capture.close()
            if failed:
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)
            env.close(skip_kit_cleanup=True)


if __name__ == '__main__':
    main()
