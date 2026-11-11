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


@pytest.mark.parametrize('primed',[False,True])
@pytest.mark.parametrize('fail',[False,True])
def test_real_cli_group_dispatch_after_live_reset_before_policy(tmp_path,monkeypatch,fail,primed):
    import tarsphere.cli.worker as worker
    episode=BenchmarkCaseLoader().load_file(ROOT/'episodes/case/fetch_and_deliver/base.yaml')
    calls=[]
    env=NS(load_episode=lambda e:calls.append('bind'),
        reset=lambda **kw:calls.append('reset') or {'actual':'reset'},
        configure_continuous_ego_video=lambda **kw:None,
        _start_continuous_ego_video=lambda:None,
        warmup_manipulation_planner=lambda path:initialize(path),
        close=lambda:calls.append('close'))
    def initialize(path):
        calls.append(('initialize',path))
        if fail:raise RuntimeError('planner dependency')
    class Runner:
        def __init__(self,**kw):pass
        def run(self,**kw):
            calls.append('policy')
            return {'episode_id':episode.episode_id,'success':False}
        def partial_result(self,**kw):
            calls.append('partial')
            return {'episode_id':episode.episode_id,'success':False}
    monkeypatch.setattr(worker,'AutonomousEpisodeRunner',Runner)
    monkeypatch.setattr('tarsphere.bench.execution_metrics.attach_execution_quality',lambda *a,**kw:None)
    class Saved(Exception):pass
    def persist(e,r,result,out):
        assert ('policy' in calls) is not fail
        if fail:assert result['outcome_status']=='infrastructure_invalid'
        raise Saved
    monkeypatch.setattr(worker,'_save_rollout_artifact',persist)
    with pytest.raises(Saved):
        worker._run_group([episode],agent=NS(),headless=True,image_dir='',viewer_mode='none',
            max_steps=1,output_dir=tmp_path,prebuilt_env=env,
            prebuilt_initial_observation={'actual':'already_reset'} if primed else None)
    if not primed:assert calls[:2]==['bind','reset']
    init=next(call for call in calls if isinstance(call,tuple))
    assert init[0]=='initialize'
    assert init[1].parent==tmp_path
    assert calls[-1]=='close'
