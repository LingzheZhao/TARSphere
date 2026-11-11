import pytest

from tarsphere.infra.simulation import scene_materials as sm


@pytest.mark.parametrize('missing', [None, 'OmniUe4Function.mdl', 'OmniUe4Base.mdl', 'Textures/map.png'])
def test_relative_material_dependency_closure(tmp_path, monkeypatch, missing):
    scene = tmp_path / 'commercial_scenes'
    materials = scene / 'Materials'
    materials.mkdir(parents=True)
    donor = tmp_path / 'donor'
    (donor / 'Textures').mkdir(parents=True)
    files = {
        'Num0123456789abcdef01234567.mdl': 'mdl 1.7; using .::OmniUe4Function import *;',
        'OmniUe4Function.mdl': 'mdl 1.7; using .::OmniUe4Base import Translucent;',
        'OmniUe4Base.mdl': 'mdl 1.7; texture_2d("Textures/map.png"); using .::OmniUe4Function import *;',
        'Textures/map.png': 'texture bytes',
    }
    for name, content in files.items():
        if name != missing:
            (donor / name).write_text(content)
    monkeypatch.setattr(sm, 'sibling_material_donors', lambda _: [])
    # Even a stock module with the missing name cannot satisfy a relative import.
    stock = tmp_path / 'sdk'
    stock.mkdir()
    if missing and missing.endswith('.mdl'):
        (stock / missing).write_text('mdl 1.7;')
    monkeypatch.setattr(sm, 'stock_material_donors', lambda: [stock])
    kwargs = dict(extra_donors=(path for path in [donor]), extra_referenced=[next(iter(files))])
    if missing:
        with pytest.raises(RuntimeError, match='still missing'):
            sm.recover_missing_collection_materials(scene, **kwargs)
    else:
        result = sm.recover_missing_collection_materials(scene, **kwargs)
        assert set(result['recovered_mdl']) == set(files) - {'Textures/map.png'}
        assert (materials / 'Textures/map.png').read_text() == 'texture bytes'
