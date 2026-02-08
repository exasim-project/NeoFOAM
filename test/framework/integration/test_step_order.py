import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - uses old @Solver/@Model decorators from bkp_solver.py. New FastAPI-like syntax tested in dummy_solver tests. Old API imports removed."
)


def test_placeholder():
    """Placeholder test to prevent collection errors."""
    pass
