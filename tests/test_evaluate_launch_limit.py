import pytest

from tarsphere.cli.evaluate import (
    EvaluationJob,
    build_parser,
    enforce_worker_launch_limit,
    main,
)


def test_default_launch_limit_is_unlimited():
    assert build_parser().parse_args([]).max_worker_launches is None
    job = EvaluationJob("job", [], launch_sequence=100)
    enforce_worker_launch_limit([job], None)
    assert job.state == "queued"


@pytest.mark.parametrize("value", ["0", "-1"])
def test_launch_limit_must_be_positive(value):
    args = build_parser().parse_args(["--max-worker-launches", value])
    with pytest.raises(ValueError, match="max-worker-launches must be positive"):
        main(args)


def test_limit_exhausts_retry_without_losing_failure_reason():
    exhausted = EvaluationJob("exhausted", [], state="cooldown", launch_sequence=1,
                              last_error="infrastructure invalid")
    fresh = EvaluationJob("fresh", [])
    finished = EvaluationJob("finished", [], state="finished", launch_sequence=1)
    enforce_worker_launch_limit([exhausted, fresh, finished], 1)
    assert exhausted.state == "failed"
    assert exhausted.last_error == "infrastructure invalid; worker launch limit exhausted (1)"
    assert fresh.state == "queued"
    assert finished.state == "finished"
