"""Explicit startup warmup for heavyweight robot perception backends.

This module is only for entrypoint-level validation/bootstrap. It does not
change the planner-visible tool architecture: it simply asks
the already-wired agent backends to load using the first real ego observation,
after Isaac owns the CUDA context.
"""

from __future__ import annotations

import os
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def heavy_backend_warmup_requested() -> bool:
    return env_flag("TARSPHERE_AGENT_WARMUP_HEAVY_BACKENDS")


def _split_prompts(value: str | None) -> list[str]:
    prompts = [item.strip() for item in (value or "").split(",")]
    return [item for item in prompts if item]


def _to_agent_observation(agent: Any, observation: Any) -> Any:
    if hasattr(agent, "_to_agent_observation") and isinstance(observation, dict):
        return agent._to_agent_observation(observation)
    return observation


def _semantic_backend(agent: Any) -> Any:
    """The SAM3 backend held by the agent's ``locate_object`` tool, if wired.

    The tool is the single owner of the backend, so warming up through it
    exercises exactly the path the model will take at decision time.
    """
    registry = getattr(agent, "tools", None)
    if registry is None:
        return None
    return getattr(registry.handler("locate_object"), "backend", None)


def warmup_configured_heavy_backends(
    agent: Any,
    observation: Any,
    *,
    sam3_prompts: list[str] | None = None,
) -> dict[str, Any]:
    """Force-load configured SAM3 backend using one ego observation.

    Mapping and navigation backends require no heavyweight model warmup here.
    The remaining heavy perception backend is SAM3. It is reached through the
    ``locate_object`` tool that owns it — the same path the model uses. If
    the tool is absent, this raises so validation cannot silently pass on the
    wrong architecture.
    """
    agent_obs = _to_agent_observation(agent, observation)
    summary: dict[str, Any] = {}

    backend = _semantic_backend(agent)
    if backend is None:
        raise RuntimeError(
            "The configured agent has no locate_object "
            "tool backed by a semantic backend. Wire SAM3 via "
            "build_robot_agent_envelope."
        )
    prompts = (
        sam3_prompts
        or _split_prompts(os.environ.get("TARSPHERE_AGENT_SAM3_WARMUP_PROMPTS"))
        or ["person"]
    )
    from tarsphere.harness.perception.contracts import PerceptionInput

    output = backend.process(
        PerceptionInput(observation=agent_obs, text_prompts=prompts)
    )
    summary["sam3"] = {
        "status": "loaded",
        "prompts": prompts,
        "detections": len(getattr(output, "detections", []) or []),
    }

    return summary


def warmup_episode_manipulation_planner(
    environment: Any, episode: Any, receipt_path: Any, *, agent_harness_config: Any = None,
) -> bool:
    """Validate declared manipulation using private, currently live robot state.

    Official episodes share tools, so a public grasp tool alone is insufficient:
    the episode must also bind an actual pickable. Call with an initialized live
    world before policy execution; the receipt is implementation evidence only.
    """
    from tarsphere.infra.episode.episode_runner import resolve_agent_action_space
    from tarsphere.infra.manipulation.g1_attachment import manipulation_objects_from_episode

    if "grasp_object" not in resolve_agent_action_space(episode, agent_harness_config):
        return False
    if not any(item.pickable for item in manipulation_objects_from_episode(episode)):
        return False
    environment.warmup_manipulation_planner(receipt_path)
    return True


__all__ = [
    "warmup_episode_manipulation_planner",
    "env_flag",
    "heavy_backend_warmup_requested",
    "warmup_configured_heavy_backends",
]
