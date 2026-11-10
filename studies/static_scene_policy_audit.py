"""CPU USD audit: preserve geometry while preparing static scene bodies."""
import argparse
import ast
import json
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    # Load only the pure policy, without importing simulation backends.
    source = Path('tarsphere/simulator/core/scene/isaacsim/scene.py').read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'prepare_static_scene_bodies')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<static-policy>', 'exec'), namespace)
    prepare = namespace['prepare_static_scene_bodies']
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim('/World/env_0/scene', 'Xform')
    assert root.GetReferences().AddReference(args.asset)
    prims = list(Usd.PrimRange.AllPrims(root))

    def snapshot():
        return {
            str(prim.GetPath()): (
                prim.GetTypeName(), tuple(prim.GetAppliedSchemas()),
                tuple((attr.GetName(), str(attr.Get())) for attr in prim.GetAttributes()
                      if attr.GetName() != 'physics:rigidBodyEnabled'),
                tuple((rel.GetName(), tuple(map(str, rel.GetTargets())))
                      for rel in prim.GetRelationships()),
            ) for prim in prims
        }

    before = snapshot()
    enabled = [prim for prim in prims if prim.GetAttribute('physics:rigidBodyEnabled').Get()]
    assert enabled, 'Audit requires authored dynamic bodies'
    transform_cache = UsdGeom.XformCache()
    determinants = [transform_cache.GetLocalToWorldTransform(prim).GetDeterminant()
                    for prim in enabled]
    protected = str(enabled[0].GetPath())
    static, dynamic = prepare(prims, [protected])
    assert dynamic >= 1 and enabled[0].GetAttribute('physics:rigidBodyEnabled').Get()
    assert snapshot() == before, 'Policy changed geometry, schemas, collisions or relationships'
    static_remaining, dynamic_remaining = prepare(prims, [])
    assert not any(prim.GetAttribute('physics:rigidBodyEnabled').Get() for prim in prims)
    assert snapshot() == before
    report = dict(asset=args.asset, prim_count=len(prims), authored_enabled_bodies=len(enabled),
                  protected_body=protected, made_static=static, preserved_dynamic=dynamic,
                  reflected_enabled_bodies=sum(value < 0 for value in determinants),
                  singular_enabled_bodies=sum(value == 0 for value in determinants),
                  rigidbody_simulation_owner_api=hasattr(UsdPhysics.RigidBodyAPI, 'GetSimulationOwnerRel'),
                  navigation_made_static=static_remaining,
                  navigation_enabled_bodies=dynamic_remaining, geometry_attributes_unchanged=True,
                  physics_scenes=[str(p.GetPath()) for p in prims if p.GetTypeName() == 'PhysicsScene'])
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
