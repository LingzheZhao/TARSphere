"""Private live-state planner validation precedes policy execution."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from tarsphere.bench.case_loader import BenchmarkCaseLoader
from tarsphere.infra.episode.episode_runner import resolve_agent_action_space
from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv
from tarsphere.infra.simulation.scene_server import SceneServer

ROOT = Path(__file__).parents[1]


def test_repository_navigation_skips_fetch_initializes_once(tmp_path):
    calls=[]
    server=SceneServer.__new__(SceneServer)
    server.agent_harness_config=None
    server.output_dir=tmp_path
    server.env=NS(warmup_manipulation_planner=lambda path:calls.append(path))
    for family in ('navigate_and_report','fetch_and_deliver'):
        episode=BenchmarkCaseLoader().load_file(ROOT/f'episodes/case/{family}/base.yaml')
        # The public tool surface is intentionally shared by official tasks.
        assert 'grasp_object' in resolve_agent_action_space(episode)
        server._warmup_manipulation_planner(episode,family)
    assert calls == [tmp_path/'fetch_and_deliver.planner_warmup.json']


@pytest.mark.parametrize('fail',[False,True])
def test_private_receipt_uses_actual_getters_and_retains_failure(tmp_path,monkeypatch,fail):
    joints={'measured_joint':.123,'base_j_z':.81}
    seen=[]
    def initialize(**kw):
        seen.append(kw)
        if fail:raise RuntimeError('actual planner dependency failed')
        return {'status':'initialized','curobo_runtime_version':'0.8.0'}
    monkeypatch.setattr('tarsphere.infra.manipulation.g1_grasp.G1GraspPlannerClient',
        lambda:NS(initialize=initialize))
    env=IsaacAutonomousEnv.__new__(IsaacAutonomousEnv)
    env._g1_curobo_joint_positions=lambda:joints
    env._g1_body_height_command=lambda:.74
    path=tmp_path/'receipt.json'
    if fail:
        with pytest.raises(RuntimeError,match='actual planner dependency failed'):
            env.warmup_manipulation_planner(path)
    else:env.warmup_manipulation_planner(path)
    assert seen==[{'current_joint_positions':joints,'current_body_height_command_m':.74}]
    receipt=json.loads(path.read_text())
    assert receipt['current_joint_positions']==joints
    assert receipt['state_source']=='live_pre_rollout_articulation'
    assert receipt['status']==('failed' if fail else 'initialized')

