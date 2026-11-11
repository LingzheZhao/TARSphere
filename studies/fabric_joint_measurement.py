"""Read live Fabric skeletal animation, following the SDK ragdoll helper convention."""
import numpy as np


def relative_live_pose(joints, animated_joints, local_matrices, world_matrix, selected):
    """Require animation coverage along every selected joint's complete parent chain."""
    indices = {str(path): i for i, path in enumerate(animated_joints)}
    skeleton = set(map(str, joints))
    world_matrix = np.asarray(world_matrix, dtype=float)
    if world_matrix.shape != (4, 4) or not np.isfinite(world_matrix).all():
        raise ValueError('Invalid Fabric world matrix')
    transforms = {}
    for name, path in selected.items():
        chain = []
        cursor = path
        while cursor:
            if cursor not in skeleton or cursor not in indices:
                raise LookupError(f'Missing live animation coverage for {cursor}')
            chain.append(cursor)
            cursor = cursor.rsplit('/', 1)[0] if '/' in cursor else ''
        transform = np.eye(4)
        for joint in reversed(chain):
            local = np.asarray(local_matrices[indices[joint]], dtype=float)
            if local.shape != (4, 4) or not np.isfinite(local).all():
                raise ValueError(f'Invalid live transform for {joint}')
            transform = local @ transform
        transforms[name] = transform @ np.asarray(world_matrix, dtype=float)
    # Remove only the pelvis rigid frame, retaining world body scale.
    pelvis = transforms['Pelvis']
    left, singular, right = np.linalg.svd(pelvis[:3, :3])
    if np.min(singular) <= 0:
        raise ValueError('Singular pelvis world transform')
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    rigid = np.eye(4)
    rigid[:3, :3] = rotation
    rigid[3, :3] = pelvis[3, :3]
    inverse = np.linalg.inv(rigid)
    return {name: (matrix @ inverse).tolist() for name, matrix in transforms.items() if name != 'Pelvis'}


def read_fabric_joints(stage, skeleton_path, meters_per_unit):
    import usdrt
    from usdrt import hierarchy
    from pxr import Gf, UsdUtils
    rt = usdrt.Usd.Stage.Attach(UsdUtils.StageCache.Get().GetId(stage).ToLongInt())
    prim = rt.GetPrimAtPath(usdrt.Sdf.Path(skeleton_path))
    skel = usdrt.UsdSkel.Skeleton(prim)
    joints = list(map(str, skel.GetJointsAttr().Get() or []))
    targets = usdrt.UsdSkel.BindingAPI(prim).GetAnimationSourceRel().GetTargets()
    if len(targets) != 1:
        raise LookupError(f'Expected one live animation binding at {skeleton_path}: {targets}')
    anim = usdrt.UsdSkel.Animation(rt.GetPrimAtPath(targets[0]))
    animated = list(map(str, anim.GetJointsAttr().Get() or []))
    arrays = [attr.Get() for attr in (anim.GetTranslationsAttr(), anim.GetRotationsAttr(), anim.GetScalesAttr())]
    if not animated or any(value is None or len(value) != len(animated) for value in arrays):
        raise ValueError(f'Incomplete live animation arrays at {targets[0]}')
    matrices = []
    for translation, rotation, scale in zip(*arrays, strict=True):
        values = [*translation, rotation.GetReal(), *rotation.GetImaginary(), *scale]
        if not np.isfinite(values).all() or np.linalg.norm(values[3:7]) <= 0:
            raise ValueError('Invalid live animation TRS')
        matrix = (Gf.Matrix4d().SetScale(Gf.Vec3d(*map(float, scale)))
                  * Gf.Matrix4d().SetRotate(Gf.Quatd(float(rotation.GetReal()), Gf.Vec3d(*map(float, rotation.GetImaginary()))))
                  * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, translation))))
        matrices.append(np.array(matrix))
    selected = {}
    for name in ('Pelvis', 'L_Thigh', 'R_Thigh', 'L_Foot', 'R_Foot'):
        matches = [path for path in joints if path.rsplit('/', 1)[-1] == name]
        if len(matches) != 1:
            raise LookupError(f'Expected one actual skeleton joint {name}: {matches}')
        selected[name] = matches[0]
    fabric = hierarchy.IFabricHierarchy().get_fabric_hierarchy(rt.GetFabricId(), rt.GetStageIdAsStageId())
    fabric.update_world_xforms()
    world = usdrt.Rt.Xformable(prim).GetFabricHierarchyWorldMatrixAttr().Get()
    if world is None:
        raise LookupError('Missing Fabric skeleton world matrix')
    world = np.array([[float(world[i][j]) for j in range(4)] for i in range(4)])
    relative = relative_live_pose(joints, animated, matrices, world, selected)
    for matrix in relative.values():
        for i in range(3):
            matrix[3][i] *= meters_per_unit
    return dict(source='fabric_live_skel_animation', skeleton_path=skeleton_path,
                animation_path=str(targets[0]), selected_joint_paths=selected,
                joints_relative_pelvis=relative)
