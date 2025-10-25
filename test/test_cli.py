import subprocess
import sys
from pathlib import Path


def test_cli_hello():
    # Run the CLI using 'uv' to ensure correct environment
    result = subprocess.run(["foamadapter"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Hello, FoamAdapter user!" in result.stdout
