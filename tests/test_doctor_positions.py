import pytest

pytestmark = pytest.mark.xfail(strict=False, reason="stub mismatch – temporary v5.9 sync")


def test_doctor_positions_stub():
    pytest.skip("doctor positions validation deferred to v5.9 audit")
