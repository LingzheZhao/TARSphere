from dataclasses import asdict
from copy import deepcopy
from types import SimpleNamespace

from tarsphere.harness.harness import ProtocolConfig
from tarsphere.harness.memory.episodic import EpisodicMemory
from tarsphere.harness.kernel.envelope_kernel import EnvelopeKernel
from tarsphere.harness.core.types import DecisionEnvelope, ToolCall


def test_inspection_survives_history_without_becoming_another_submission():
    memory = EpisodicMemory(protocol=ProtocolConfig(text_history_horizon=1))
    conclusion = "The bed has a raised blue upholstered headboard."
    later = DecisionEnvelope(thinking="Returning", actions=[ToolCall(tool="wait", args={})])
    for step in range(4):
        state = memory.begin_action_step(
            action_step=step, observation=SimpleNamespace(text="", public_events=[], pose={"body_posture": "standing"}),
            perception=None, record_visual_history=False,
        )
        envelope = later  # Accepted output may come from an earlier repair attempt.
        batches = [{"inspection": conclusion}] if step == 0 else []
        original_batches = deepcopy(batches)
        memory.record_policy_decision(
            policy_step=step, envelope=envelope, actions=[{"action": "wait", "args": {}}],
            accepted_memory_update_batches=[], accepted_event_output_batches=batches,
            plan_updates_accepted=False,
        )
        assert batches == original_batches
        if step == 3:
            assert all(r.action_step != 0 for r in state.text_history)
            assert asdict(state)["submitted_inspections"] == [
                {"text": conclusion, "action_step": 0, "policy_step": 0}
            ]
            assert conclusion in EnvelopeKernel._format_longterm_state(state)
            assert "inspection" not in later.as_dict()
        memory.record_executed_action(policy_step=step, action={"action": "wait"})
        memory.complete_action_step({})
    memory.reset()
    state = memory.begin_action_step(
        action_step=0, observation=SimpleNamespace(text="", public_events=[], pose={"body_posture": "standing"}),
        perception=None, record_visual_history=False,
    )
    assert state.submitted_inspections == []


def test_raw_unaccepted_inspection_is_not_retained():
    memory = EpisodicMemory(protocol=ProtocolConfig())
    memory.begin_action_step(
        action_step=0, observation=SimpleNamespace(text="", public_events=[], pose={"body_posture": "standing"}),
        perception=None, record_visual_history=False,
    )
    memory.record_policy_decision(
        policy_step=0,
        envelope=DecisionEnvelope(thinking="Unaccepted", actions=[ToolCall(tool="wait", args={})], inspection="not submitted"),
        actions=[], accepted_memory_update_batches=[], accepted_event_output_batches=[],
        plan_updates_accepted=False,
    )
    memory.record_executed_action(policy_step=0, action={"action": "wait"})
    memory.complete_action_step({})
    state = memory.begin_action_step(
        action_step=1, observation=SimpleNamespace(text="", public_events=[], pose={"body_posture": "standing"}),
        perception=None, record_visual_history=False,
    )
    assert state.submitted_inspections == []
