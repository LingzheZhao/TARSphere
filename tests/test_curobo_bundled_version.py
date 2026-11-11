"""Only the audited checkout's official fallback is accepted without metadata."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from tarsphere.infra.manipulation import g1_grasp


def test_official_fallback_requires_audited_bundle_path(monkeypatch):
    root = Path(g1_grasp.__file__).resolve().parents[3]
    module = SimpleNamespace(__version__='v0.8.0-no-tag',
        __file__=str(root / 'third_party/curobo/curobo/__init__.py'))
    monkeypatch.setitem(sys.modules, 'curobo', module)
    g1_grasp._require_curobo_v2()
    module.__file__ = '/unverified/curobo/__init__.py'
    with pytest.raises(RuntimeError, match='requires cuRobo'):
        g1_grasp._require_curobo_v2()
    module.__version__ = '0.8.0'
    g1_grasp._require_curobo_v2()


@pytest.mark.parametrize('release', ['0.7.0', '0.9.0', 'v0.9.0-no-tag', 'unknown'])
def test_wrong_releases_rejected_even_from_bundle(monkeypatch, release):
    root = Path(g1_grasp.__file__).resolve().parents[3]
    monkeypatch.setitem(sys.modules, 'curobo', SimpleNamespace(__version__=release,
        __file__=str(root / 'third_party/curobo/curobo/__init__.py')))
    with pytest.raises(RuntimeError, match='requires cuRobo'):
        g1_grasp._require_curobo_v2()
