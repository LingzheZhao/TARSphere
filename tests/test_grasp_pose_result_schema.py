"""Current pose evidence does not manufacture native-contact certification."""
import pytest
from pydantic import ValidationError
from tarsphere.bench.certification.physical_rollout import GraspPoseCertificationEvidence


def test_pre_action_failure_is_valid_pose_evidence():
    evidence = dict(target_object_id='target', grasp_pose_established=False,
        attachment_evidence=None, minimum_aabb_surface_gap_m=None,
        carried_away=False, maximum_attached_displacement_m=None,
        required_displacement_m=0.05, carried_state=None, passed=False)
    result = GraspPoseCertificationEvidence.model_validate(evidence)
    assert not result.passed
    assert 'contact' not in result.model_dump()
    with pytest.raises(ValidationError):
        GraspPoseCertificationEvidence.model_validate({**evidence, 'passed': True})
