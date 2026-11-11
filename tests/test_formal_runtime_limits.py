from pathlib import Path

import pytest

from tarsphere.bench.case_loader import BenchmarkCaseLoader
from tarsphere.evaluation.official import validate_formal_episode_runtime


def released_episode():
    return BenchmarkCaseLoader().load_file(
        Path(__file__).resolve().parents[1]
        / "episodes/case/navigate_and_report/task_update.yaml"
    )


def test_released_task_update_preserves_formal_limit():
    episode = released_episode()
    assert episode.runtime.max_decision_steps == 100
    validate_formal_episode_runtime(episode)
    episode.runtime.max_decision_steps = None
    validate_formal_episode_runtime(episode)


@pytest.mark.parametrize("limit", [1, 99, 101, 1000])
def test_episode_cannot_change_formal_decision_limit(limit):
    episode = released_episode()
    episode.runtime.max_decision_steps = limit
    with pytest.raises(ValueError, match="formal limit"):
        validate_formal_episode_runtime(episode)


def test_episode_cannot_add_physics_limit():
    episode = released_episode()
    episode.runtime.max_physics_steps = 100
    with pytest.raises(ValueError, match="max_physics_steps"):
        validate_formal_episode_runtime(episode)
