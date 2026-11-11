"""NPC topic authorization shares the bounded support-response budget."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tarsphere.configs.secrets import EvaluationSupportServiceError
from tarsphere.humans.llm_policy import HumanDialoguePolicy


@pytest.fixture(autouse=True)
def support_sdk_errors(monkeypatch):
    # No transport is exercised; keep these CPU tests independent of the SDK.
    import sys
    errors = {name: type(name, (Exception,), {}) for name in (
        'APIConnectionError', 'APITimeoutError', 'InternalServerError', 'RateLimitError')}
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(**errors))


def policy_with_responses(contents):
    policy = HumanDialoguePolicy.__new__(HumanDialoguePolicy)
    policy.config = SimpleNamespace(model='local-qwen3-4b')
    policy._npc_context = {}
    policy._model_calls = []
    create = Mock(side_effect=[SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=None
    ) for text in contents])
    policy._client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    return policy, create


def route(policy, topics):
    return policy.route_question(question='Where is the doorway?', topics=topics,
        recent_human_utterances=[], history=[], public_context={})


def test_empty_topics_does_not_call_support_model():
    policy, create = policy_with_responses([])
    assert route(policy, []).intent_id is None
    create.assert_not_called()
    assert policy._model_calls == []


def test_unauthorized_intent_retries_then_accepts_authorized_selection():
    policy, create = policy_with_responses([
        '{"intent_id":"one"}', '{"intent_id":"doorway"}'])
    assert route(policy, [{'intent_id': 'doorway'}]).intent_id == 'doorway'
    assert create.call_count == 2
    assert policy._model_calls[0]['provider_responses'] == 2


def test_exhausted_unauthorized_intents_raise_typed_support_error():
    policy, create = policy_with_responses(['{"intent_id":"one"}'] * 3)
    with pytest.raises(EvaluationSupportServiceError, match='after three attempts'):
        route(policy, [{'intent_id': 'doorway'}])
    assert create.call_count == 3
    assert policy._model_calls[0]['provider_responses'] == 3
