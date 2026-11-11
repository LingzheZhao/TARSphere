"""Missing support-surface operations must fail before Isaac starts."""
import sys
from types import ModuleType

import pytest
from tarsphere.infra.simulation.scene_server_preflight import preload_support_surface_geometry


def test_old_shapely_preserves_the_missing_operation_error(monkeypatch):
    old = ModuleType('shapely')
    old.__version__ = '2.0.7'
    monkeypatch.setitem(sys.modules, 'shapely', old)
    with pytest.raises(RuntimeError, match='before Isaac starts') as error:
        preload_support_surface_geometry()
    assert isinstance(error.value.__cause__, ImportError)
