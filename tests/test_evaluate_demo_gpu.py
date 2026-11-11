"""Evaluation workers retain the explicitly allocated demo CUDA identity."""
import pytest

from tarsphere.cli.evaluate import WorkerSlot, worker_environment
from tarsphere.infra.simulation import demo_gpu


UUID = "GPU-592eec6a-cc01-65a5-b880-fc4a40cd58c5"


@pytest.fixture
def allocated_demo(monkeypatch):
    monkeypatch.setenv("TARSPHERE_DEMO_GPU_UUID", UUID)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    monkeypatch.setenv("TARSPHERE_KIT_ACTIVE_GPU", "2")
    def lookup(index):
        assert index == 2
        return UUID
    monkeypatch.setattr(demo_gpu, "lookup_physical_gpu_uuid", lookup)


def test_worker_preserves_verified_demo_mapping(allocated_demo, tmp_path):
    environment = worker_environment(WorkerSlot(2, 1), 19001, tmp_path, 4)
    assert demo_gpu.resolve_demo_gpu_mapping(environment) == {
        "physical_gpu": 2, "uuid": UUID, "cuda_ordinal": 0,
    }
    assert environment["CUDA_VISIBLE_DEVICES"] == UUID
    assert environment["TARSPHERE_WORKER_CUDA_VISIBLE_DEVICES"] == UUID
    assert environment["TARSPHERE_KIT_ACTIVE_GPU"] == "2"
    assert environment["TARSPHERE_MODEL_GPU"] == "0"
    assert environment["TARSPHERE_FRONTIERNET_DEVICE"] == "cuda:0"
    assert environment["TARSPHERE_ISAAC_RUNTIME_SLOT"] == "1"


def test_worker_rejects_slot_outside_demo_allocation(allocated_demo, tmp_path):
    with pytest.raises(RuntimeError, match="differs from the verified demo GPU"):
        worker_environment(WorkerSlot(3, 0), 19000, tmp_path, 4)


def test_worker_validates_parent_before_rewriting_it(allocated_demo, monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(RuntimeError, match="exact TARSPHERE_DEMO_GPU_UUID"):
        worker_environment(WorkerSlot(2, 0), 19000, tmp_path, 4)


def test_worker_legacy_unmasked_mapping(monkeypatch, tmp_path):
    monkeypatch.delenv("TARSPHERE_DEMO_GPU_UUID", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    environment = worker_environment(WorkerSlot(2, 0), 19000, tmp_path, 4)
    assert demo_gpu.resolve_demo_gpu_mapping(environment) is None
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert environment["TARSPHERE_WORKER_CUDA_VISIBLE_DEVICES"] == "2"
    assert environment["TARSPHERE_KIT_ACTIVE_GPU"] == "2"
    assert environment["TARSPHERE_MODEL_GPU"] == "2"
    assert environment["TARSPHERE_FRONTIERNET_DEVICE"] == "cuda:2"
