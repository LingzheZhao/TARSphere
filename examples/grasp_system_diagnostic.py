"""Run one author known-positive grasp diagnostic, never an evaluated MLLM task.

Starts the original SceneServer in-process in golden-oracle mode by default.
Use --http only with an independently provided compatible HTTP transport.
In-process execution requires --grounded-scene prepared by the original
resolve_scene API in a separate CPU process; this avoids importing USD before Kit.
The provisioning job must enforce a finite wall-clock budget in-process. Frozen target geometry is used only by the certification policy.
A single pass is a diagnostic, not the author's consecutive-pass certificate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import traceback

from tarsphere.bench.grounding.schema import GroundedScene
from tarsphere.bench.certification.scene_client import SceneServerClient
from tarsphere.bench.certification.grasp_system import resolve_grasp_system_trial
from tarsphere.bench.certification.grasp_system_episode import compile_grasp_system_episode
from tarsphere.bench.certification.physical_rollout import run_grasp_system_trial


class InProcessCertificationClient:
    """Adapt the original server methods to the original strict result reader."""
    def __init__(self, server):
        self.server = server

    def identity(self):
        from tarsphere.bench.certification.scene_client import SceneServerIdentity
        return SceneServerIdentity(
            scene_server_id=self.server.scene_server_id,
            scene_server_git_sha=self.server.scene_server_git_sha,
            scene=self.server.scene_asset_path,
            active_request_id=self.server.active_request_id)

    def run_certification_episode(self, episode, **kwargs):
        return self.server.run_episode(episode, **kwargs)

    def abort(self, request_id):
        return self.server.abort_active(request_id)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', required=True)
    parser.add_argument('--scene-id', required=True)
    parser.add_argument('--grounded-scene', type=Path, help='Validated GroundedScene JSON from CPU preparation; required in-process')
    parser.add_argument('--http', action='store_true', help='Use external HTTP transport instead of owning Isaac')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=positive_int, default=8765)
    parser.add_argument('--timeout-s', type=positive_seconds, default=1800)
    parser.add_argument('--max-steps', type=positive_int, default=60)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    args.assets_root = str(Path(args.assets_root).expanduser().resolve())
    if not args.http and args.grounded_scene is None:
        parser.error('--grounded-scene is required for in-process execution')
    if args.grounded_scene is not None:
        args.grounded_scene = args.grounded_scene.expanduser().resolve()
    if not args.request_id.strip():
        parser.error('--request-id must not be blank')
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / 'arguments.json').write_text(json.dumps(
        {key: str(value) if isinstance(value, Path) else value
         for key, value in vars(args).items()}, indent=2))
    server = None
    try:
        if args.grounded_scene is not None:
            scene = GroundedScene.model_validate_json(args.grounded_scene.read_text())
            if scene.scene_id != args.scene_id:
                raise ValueError('Grounded scene identity differs from --scene-id')
            (args.output_root / 'grounded_scene.json').write_text(scene.model_dump_json(indent=2))
        else:
            # Only the client-only CPU process may resolve USD metadata here.
            from tarsphere.bench.grounding.resolve import resolve_scene
            scene = resolve_scene(args.scene_id, assets_root=args.assets_root)
        if args.http:
            client = SceneServerClient(host=args.host, port=args.port, timeout_s=args.timeout_s)
            identity = client.identity()
            (args.output_root / 'server_identity.json').write_text(identity.model_dump_json(indent=2))
            if identity.active_request_id is not None or scene.scene_id not in identity.scene:
                raise RuntimeError('Expected an idle SceneServer loaded with the grounded scene')
        else:
            os.environ['TARSPHERE_ASSETS_PATH'] = args.assets_root
            from tarsphere.infra.simulation.gpu_bootstrap import configure_cuda_allocator
            configure_cuda_allocator()
            from tarsphere.infra.simulation.scene_server import SceneServer
            server = SceneServer(output_dir=str(args.output_root/'server'),
                headless=True, agent_mode='golden-oracle', evaluate_after_rollout=False)
            # Trial resolution needs the real pinned server identity before
            # bootstrap; compilation supplies the episode needed to load Isaac.
            from types import SimpleNamespace
            identity = SimpleNamespace(scene_server_id=server.scene_server_id,
                scene_server_git_sha=server.scene_server_git_sha)
        trial = resolve_grasp_system_trial(scene, scene_server_id=identity.scene_server_id,
            scene_server_git_sha=identity.scene_server_git_sha, assets_root=args.assets_root)
        episode = compile_grasp_system_episode('single_grasp_component_diagnostic', trial=trial, scene=scene)
        (args.output_root / 'episode.json').write_text(episode.model_dump_json(indent=2))
        if server is not None:
            server.bootstrap(episode)
            client = InProcessCertificationClient(server)
            identity = client.identity()
        (args.output_root / 'server_identity.json').write_text(identity.model_dump_json(indent=2))
        result = run_grasp_system_trial(client, episode, request_id=args.request_id,
            record_video=True, max_steps=args.max_steps)
        (args.output_root / 'result.json').write_text(result.model_dump_json(indent=2))
    except Exception as exc:
        (args.output_root / 'failure.json').write_text(json.dumps({
            'type': type(exc).__name__, 'message': str(exc),
            'traceback': traceback.format_exc()}, indent=2))
        print(f'Component diagnostic failed: {exc}')
        return 2
    finally:
        if server is not None:
            server.close()
    print(f'Component diagnostic passed: {result.passed}')
    return 0 if result.passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
