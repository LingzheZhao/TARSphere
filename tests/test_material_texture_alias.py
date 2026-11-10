"""Texture inputs resolve when USD references a differently nested symlink alias."""
import pytest

pytest.importorskip('pxr')
from pxr import Sdf, Usd

from tarsphere.infra.simulation.mdl_search_paths import (
    rewrite_relative_mdl_source_assets_in_layer,
)


def test_texture_rewrite_resolves_through_symlinked_model_alias(tmp_path):
    materials = tmp_path / 'collection' / 'Materials'
    texture = materials / 'Textures' / 'map.png'
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b'texture')
    canonical = tmp_path / 'assets' / 'deep' / 'nested' / 'model'
    canonical.mkdir(parents=True)
    source = Usd.Stage.CreateNew(str(canonical / 'instance.usda'))
    prim = source.DefinePrim('/Model')
    prim.CreateAttribute('inputs:diffuse_Tex', Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath('./Materials/Textures/map.png')
    )
    source.GetRootLayer().Save()
    scene = tmp_path / 'scene'
    scene.mkdir()
    (scene / 'models').symlink_to(canonical, target_is_directory=True)
    alias = scene / 'models' / 'instance.usda'
    layer = Sdf.Layer.FindOrOpen(str(alias))
    assert rewrite_relative_mdl_source_assets_in_layer(layer, materials_dir=materials) == 1
    layer.Save()
    stage = Usd.Stage.CreateNew(str(scene / 'stage.usda'))
    stage.DefinePrim('/Referenced').GetReferences().AddReference(str(alias), '/Model')
    value = stage.GetPrimAtPath('/Referenced').GetAttribute('inputs:diffuse_Tex').Get()
    assert value.path == str(texture.resolve())
    assert value.resolvedPath == str(texture.resolve())
    assert rewrite_relative_mdl_source_assets_in_layer(layer, materials_dir=materials) == 0
