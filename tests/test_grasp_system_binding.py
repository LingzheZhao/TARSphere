from types import SimpleNamespace
from copy import deepcopy
import pytest
from tarsphere.bench.certification.grasp_system import GraspSystemTrial
from tarsphere.bench.certification.grasp_system_episode import compile_grasp_system_episode, validate_grasp_system_world
from tarsphere.bench.grounding.schema import SceneMap, SceneSources
from tarsphere.bench.certification import physical_rollout
from tarsphere.harness.certification.oracle import GoldenOracleAgent


def compiled():
    trial = GraspSystemTrial.model_validate({
        "scene_id": "test_scene", "scene_server_id": "server", "scene_server_git_sha": "revision",
        "reference_object": {"center_xyz": [1, 0, .7], "local_unit_in_meters": .01},
        "grasp": {"target_object_id": "tarsphere_grasp_reference", "navigation_approach_xy": [0, 0],
                  "robot_stance_xy": [0, 0], "robot_yaw_rad": 0, "retreat_xy": [0, 0]},
    })
    scene = SimpleNamespace(scene_id="test_scene", scene_scale=(.01, .01, .01),
        sources=SceneSources(**{k: "scene.usd" for k in SceneSources.model_fields}),
        map=SceneMap(**{**{k: "asset" for k in ["rgb_bev", "room_bev", "projection", "room_annotations", "occupancy"]},
                      "resolution": 10, "bounds_min_xy": [0, 0], "bounds_max_xy": [2, 2]}))
    return compile_grasp_system_episode("binding_test", trial=trial, scene=scene)


def test_compiler_binding_passes_reader_and_oracle_without_fake_execution(monkeypatch):
    episode = compiled()
    trial = validate_grasp_system_world(episode.world_knowledge)
    phases = GoldenOracleAgent._build_phases(episode)
    assert phases and phases[-1]["kind"] == "report"
    class ReachedExecution(Exception): pass
    def execution(*args, **kwargs):
        assert kwargs["stage"] == "A"
        raise ReachedExecution
    monkeypatch.setattr(physical_rollout, "_run_certification_episode", execution)
    with pytest.raises(ReachedExecution):
        physical_rollout.run_grasp_system_trial(None, episode)
    assert episode.world_knowledge["physical_task"]["target_object_id"] == trial.reference_object.object_id


@pytest.mark.parametrize("mutation", ["target", "grasp", "extra", "route", "trial"])
def test_mixed_or_mismatched_task_rejected(mutation):
    episode = compiled()
    world = deepcopy(episode.world_knowledge)
    if mutation == "target": world["physical_task"]["target_object_id"] = "other"
    elif mutation == "grasp": world["physical_task"]["grasp"]["robot_yaw_rad"] = 1
    elif mutation == "extra": world["physical_task"]["place"] = {}
    elif mutation == "route": world["episode_route"] = {}
    else: world["physical_trial"] = {}
    episode.world_knowledge = world
    with pytest.raises(ValueError): validate_grasp_system_world(world)
    with pytest.raises(ValueError): GoldenOracleAgent._build_phases(episode)
    with pytest.raises(ValueError): physical_rollout.run_grasp_system_trial(None, episode)
