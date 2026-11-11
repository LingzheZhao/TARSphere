import os
from typing import List

from tarsphere.simulator.core.config import TaskCfg
from tarsphere.simulator.core.robot.rigid_body import IRigidBody
from tarsphere.simulator.core.scene import validate_scene_file
from tarsphere.simulator.core.scene.scene import IScene


class IsaacsimScene(IScene):
    """IsaacSim's implementation on `IScene` class."""

    def __init__(self):
        from isaacsim.core.api import World
        from isaacsim.core.api.scenes import Scene

        self._scene: Scene = World.instance().scene

    def load(self, task_config: TaskCfg, env_id: int, env_offset: List[float]):
        """See `IScene.load` for documentation."""
        usd_path = os.path.abspath(task_config.scene_asset_path)
        prim_path_root = f"World/env_{env_id}/scene"
        source, prim_path = validate_scene_file(usd_path, prim_path_root)

        from isaacsim.core.utils.prims import create_prim

        position = [
            env_offset[idx] + i for idx, i in enumerate(task_config.scene_position)
        ]
        scene_prim = create_prim(
            prim_path,
            usd_path=source,
            scale=task_config.scene_scale,
            translation=position,
        )
        # Prepare immutable geometry before BaseTask constructs dynamic-body
        # wrappers. Wrapping mirrored furniture first can decompose reflected
        # transforms and fail even though its static collision mesh is valid.
        if task_config.scene_static:
            from pxr import Usd

            # Hold declared pickables before BaseTask wraps bodies or reset
            # advances physics; the grasp executor later enables their motion.
            prepare_scene_pickables(scene_prim, task_config.scene_dynamic_body_paths)
            prepare_static_scene_bodies(
                Usd.PrimRange.AllPrims(scene_prim), task_config.scene_dynamic_body_paths
            )
        self.scene_prim = scene_prim

    def add(self, target: any):
        """See `IScene.add` for documentation."""
        if hasattr(target, "initialize") and hasattr(target, "unwrap"):
            # TODO: Implement initialize method on IArticulation._articulation to make
            # 'self._scene._scene_registry.add_articulated_system' -> 'self._scene.add'
            self._scene._scene_registry.add_articulated_system(
                name=target.name, articulated_system=target
            )
        elif hasattr(target, "unwrap"):
            self._scene.add(target.unwrap())
        else:
            # For instance of isaac-sim native classes
            self._scene.add(target)

    def remove(self, target: any, registry_only: bool = False):
        """See `IScene.remove` for documentation."""
        self._scene.remove_object(name=target, registry_only=registry_only)

    def object_exists(self, target: any) -> bool:
        """See `IScene.object_exists` for documentation."""
        return self._scene.object_exists(target)

    def get(self, target: any) -> IRigidBody:
        """See `IScene.get` for documentation."""
        object = self._scene.get_object(target)
        return IRigidBody.create(prim_path=object.prim_path, name=object.prim_path)

    def unwrap(self):
        """See `IScene.unwrap` for documentation."""
        return self._scene


def prepare_static_scene_bodies(prims, dynamic_body_paths):
    """Disable immutable body dynamics without changing transforms or collisions."""
    dynamic_paths = tuple(path.rstrip("/") for path in dynamic_body_paths)
    made_static = preserved_dynamic = 0
    for prim in prims:
        enabled = prim.GetAttribute("physics:rigidBodyEnabled")
        if not enabled or not enabled.IsValid() or not enabled.Get():
            continue
        path = str(prim.GetPath())
        if any(path == root or path.startswith(root + "/") for root in dynamic_paths):
            preserved_dynamic += 1
            continue
        if not enabled.Set(False):
            raise RuntimeError(f"Cannot disable immutable scene body {path}")
        made_static += 1
    return made_static, preserved_dynamic


def prepare_scene_pickables(scene_prim, declared_paths):
    """Hold existing scene pickables at their authored pose until grasp begins.

    Later episode-injected bodies are absent here and retain their own setup.
    The shared normalizer rejects uneditable instance-proxy rigid descendants.
    """
    from tarsphere.infra.manipulation.pickable_physics import configure_dynamic_pickable

    root = str(scene_prim.GetPath()).rstrip("/")
    stage = scene_prim.GetStage()
    for path in dict.fromkeys(str(value).rstrip("/") for value in declared_paths):
        if not path.startswith(root + "/"):
            continue
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        if prim.IsInstanceProxy():
            raise RuntimeError(f"Declared pickable root is an instance proxy: {path}")
        configure_dynamic_pickable(prim, kinematic=True)
