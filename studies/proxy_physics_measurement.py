"""Read one proxy actor from the runner-owned physics view, without mutation."""
import numpy as np


def measure_proxy_physics(world, prim_path, meters_per_unit):
    receipt = dict(status='unavailable', source='world_physics_sim_view', path=prim_path,
                   count=None, backend=getattr(world, 'backend', None), pose=None)
    try:
        if not prim_path or any(token in prim_path for token in '*?[]'):
            raise ValueError('An exact proxy prim path is required')
        simulation = getattr(world, 'physics_sim_view', None)
        if simulation is None:
            raise RuntimeError('Existing World.physics_sim_view is unavailable')
        body = simulation.create_rigid_body_view(prim_path)
        receipt['count'] = int(body.count)
        if body.count != 1:
            raise LookupError(f'Expected one registered proxy actor; got {body.count}')
        pose = np.asarray(world.backend_utils.to_numpy(body.get_transforms()), dtype=float)
        if pose.shape != (1, 7) or not np.isfinite(pose).all():
            raise ValueError(f'Invalid PhysX pose shape or values: {pose.tolist()}')
        scale = float(meters_per_unit)
        if not np.isfinite(scale) or scale <= 0 or np.linalg.norm(pose[0, 3:]) <= 0:
            raise ValueError('Invalid units or PhysX quaternion')
        receipt.update(status='available', pose=dict(position_stage_units=pose[0, :3].tolist(),
                       position_m=(pose[0, :3] * scale).tolist(), quaternion_xyzw=pose[0, 3:].tolist()))
    except Exception as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc))
    return receipt
