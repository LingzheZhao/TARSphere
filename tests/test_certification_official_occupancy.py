"""The compiled official map is usable without a custom-bake sidecar."""
from types import SimpleNamespace

import numpy as np
import pytest

from tarsphere.bench.certification.scene import load_episode_occupancy


def test_navigation_usd_uses_declared_official_npy(tmp_path):
    path = tmp_path / 'occupancy.npy'
    np.save(path, np.array([[0, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float))
    episode = SimpleNamespace(
        world_knowledge={'scene_map': {'occupancy': 'occupancy.npy'}},
        scene=SimpleNamespace(asset_path='scene/start_result_navigation.usd'),
    )
    result = load_episode_occupancy(episode, assets_root=tmp_path)
    assert result.occupancy_path == str(path)
    assert not path.with_suffix('.json').exists()
    path.unlink()
    with pytest.raises(FileNotFoundError, match='occupancy is missing'):
        load_episode_occupancy(episode, assets_root=tmp_path)
