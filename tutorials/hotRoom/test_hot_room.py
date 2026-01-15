import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="function")
def change_test_dir(request):
    """Change to test directory for OpenFOAM case access."""
    os.chdir(request.fspath.dirname)
    yield
    os.chdir(request.config.invocation_dir)


def test_hot_room(run_reset_case, change_test_dir):
    """Test vol_alpha.csv has correct structure and reasonable values."""
    assert False
