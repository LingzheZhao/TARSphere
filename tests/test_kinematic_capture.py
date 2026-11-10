"""Real USD skeletal capture, independent of Kit and animation backend mocks."""
import pytest

pytest.importorskip("pxr", reason="Kinematic capture requires OpenUSD")
from pxr import Gf, Usd, UsdGeom, UsdSkel, UsdSemantics

from tarsphere.infra.control.kinematic_human_executor import UsdKinematicCharacters
from tarsphere.infra.perception.teleop_capture import (
    _kinematic_character_snapshot,
    _same_state,
)


def character():
    stage = Usd.Stage.CreateInMemory()
    root = UsdSkel.Root.Define(stage, "/World/Characters/Nora")
    skeleton = UsdSkel.Skeleton.Define(stage, "/World/Characters/Nora/Skeleton")
    joints = ["Root", "Root/Arm"]
    skeleton.CreateJointsAttr(joints)
    skeleton.CreateRestTransformsAttr([Gf.Matrix4d(1), Gf.Matrix4d(1)])
    skeleton.CreateBindTransformsAttr([Gf.Matrix4d(1), Gf.Matrix4d(1)])
    animation = UsdSkel.Animation.Define(stage, "/World/Characters/Nora/IdlePose")
    animation.CreateJointsAttr(joints)
    animation.CreateTranslationsAttr([Gf.Vec3f(0), Gf.Vec3f(0, 0, 1)])
    animation.CreateRotationsAttr([Gf.Quatf(1), Gf.Quatf(1)])
    animation.CreateScalesAttr([Gf.Vec3h(1), Gf.Vec3h(1)])
    UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets(
        [animation.GetPath()]
    )
    return stage, root, animation


def test_capture_samples_bound_animation_and_detects_joint_and_root_changes():
    stage, root, animation = character()
    path = str(root.GetPath())
    before = _kinematic_character_snapshot(stage, path, scene_bound=False)
    assert _same_state(before, _kinematic_character_snapshot(stage, path, scene_bound=False))
    joint_path = path + "/Skeleton"
    assert before["skeletons"][joint_path]["joint_local_transforms"][1][3][2] == 1
    animation.GetTranslationsAttr().Set([Gf.Vec3f(0), Gf.Vec3f(0, 0, 2)])
    changed_joint = _kinematic_character_snapshot(stage, path, scene_bound=False)
    assert changed_joint["skeletons"][joint_path]["joint_local_transforms"][1][3][2] == 2
    assert not _same_state(before, changed_joint)
    UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(2, 0, 0))
    assert not _same_state(
        changed_joint, _kinematic_character_snapshot(stage, path, scene_bound=False)
    )


def test_scene_bound_capture_reads_pose_and_catalog_capture_requires_skeleton():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Person")
    root.AddTranslateOp().Set(Gf.Vec3d(2, 3, 0))
    state = _kinematic_character_snapshot(stage, str(root.GetPath()), scene_bound=True)
    assert state["world_transform"][3][:3] == [2, 3, 0]
    with pytest.raises(RuntimeError, match="no readable USD skeleton"):
        _kinematic_character_snapshot(stage, str(root.GetPath()), scene_bound=False)


def test_nora_class_label_overwrites_previous_and_inherits_to_mesh():
    stage, root, _ = character()
    UsdSemantics.LabelsAPI.Apply(root.GetPrim(), "class").CreateLabelsAttr().Set(["old"])
    UsdKinematicCharacters._apply_identity_label(root.GetPrim(), "Nora")
    mesh = UsdGeom.Mesh.Define(stage, str(root.GetPath()) + "/Body/Mesh")
    query = UsdSemantics.LabelsQuery("class", Usd.TimeCode.Default())
    assert list(query.ComputeUniqueDirectLabels(root.GetPrim())) == ["Nora"]
    assert list(query.ComputeUniqueDirectLabels(mesh.GetPrim())) == []
    assert list(query.ComputeUniqueInheritedLabels(mesh.GetPrim())) == ["Nora"]
