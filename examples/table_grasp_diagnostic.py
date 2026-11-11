"""Run the author's existing table grasp skill without an MLLM.

Requires CPU-prepared GroundedScene JSON and its reviewed occupancy under the
pinned assets root. Provisioning must enforce a finite wall-clock budget.
This exercises the public grasp stack; full Fetch remains the main validation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

from tarsphere.bench.grounding.schema import GroundedScene
from tarsphere.bench.certification.grasp_motion import resolve_grasp_motion_trial
from tarsphere.bench.certification.grasp_motion_episode import compile_table_grasp_skill_episode


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def prepare_episode(grounded_scene, assets_root):
    """Read prepared facts, never resolve USD before Kit starts."""
    os.environ['TARSPHERE_ASSETS_PATH'] = str(assets_root)
    scene = GroundedScene.model_validate_json(Path(grounded_scene).read_text())
    occupancy = (Path(assets_root) / scene.map.occupancy).resolve()
    # Do not substitute an unrelated cached NPY for the declared input.
    if not occupancy.is_relative_to(Path(assets_root).resolve()):
        raise ValueError('Reviewed occupancy must reside under --assets-root')
    if not occupancy.is_file():
        raise FileNotFoundError(occupancy)
    trial = resolve_grasp_motion_trial(scene)
    episode = compile_table_grasp_skill_episode(
        'table_grasp_diagnostic', trial=trial, navigation_scene=scene)
    return scene, episode


def reported_measurements(result):
    """Preserve reported physical samples; absence is unknown, never success."""
    rows = list(result.get('trace') or [])
    artifact = result.get('rollout_artifact_path')
    heartbeats = []
    if artifact and Path(artifact).is_file():
        heartbeats = json.loads(Path(artifact).read_text()).get('heartbeats') or []
    return {
        'holding_samples': [row['held'] for row in heartbeats if 'held' in row],
        'retention_samples': [row['grasp_skill']['retention'] for row in rows
            if isinstance(row.get('grasp_skill'), dict) and 'retention' in row['grasp_skill']],
        'physical_retention_samples': [row['grasp_skill']['physical_retention'] for row in rows
            if isinstance(row.get('grasp_skill'), dict) and 'physical_retention' in row['grasp_skill']],
        'transport_ready_samples': [row['grasp_skill']['transport_ready'] for row in rows
            if isinstance(row.get('grasp_skill'), dict) and 'transport_ready' in row['grasp_skill']],
        'grasp_skill_samples': [row['grasp_skill'] for row in rows if row.get('grasp_skill')],
        'grasp_failures': [row['grasp_failure'] for row in rows if row.get('grasp_failure')],
        'termination_reason': result.get('termination_reason'),
        'failure': result.get('failure'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--grounded-scene', type=Path, required=True)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--max-steps', type=positive_int, default=12)
    args = parser.parse_args(argv)
    if not args.request_id.strip():
        parser.error('--request-id must not be blank')
    args.assets_root = args.assets_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output_root / name).write_text(json.dumps(value, indent=2))
    save('arguments.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    server = None
    try:
        scene, episode = prepare_episode(args.grounded_scene, args.assets_root)
        (args.output_root / 'grounded_scene.json').write_text(scene.model_dump_json(indent=2))
        (args.output_root / 'episode.json').write_text(episode.model_dump_json(indent=2))
        from tarsphere.infra.simulation.scene_server_preflight import preload_support_surface_geometry
        save('support_geometry_environment.json', preload_support_surface_geometry())
        from tarsphere.infra.simulation.gpu_bootstrap import configure_cuda_allocator
        configure_cuda_allocator()
        from tarsphere.infra.simulation.scene_server import SceneServer
        server = SceneServer(output_dir=str(args.output_root / 'server'), headless=True,
            agent_mode='grasp-certification', evaluate_after_rollout=False)
        server.bootstrap(episode)
        result = server.run_episode(episode, request_id=args.request_id,
            record_video=True, max_steps=args.max_steps)
        save('result.json', result)
        save('reported_measurements.json', reported_measurements(result))
    except Exception as exc:
        save('failure.json', {'type': type(exc).__name__, 'message': str(exc),
            'traceback': traceback.format_exc()})
        print(f'Table diagnostic failed: {exc}')
        return 2
    finally:
        if server is not None:
            server.close()
    print(f"Table diagnostic ended: {result.get('termination_reason')}; inspect raw physical evidence")
    return 1 if result.get('failure') else 0


if __name__ == '__main__':
    raise SystemExit(main())
