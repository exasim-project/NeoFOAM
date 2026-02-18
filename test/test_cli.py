import subprocess


def test_cli_hello() -> None:
    result = subprocess.run(
        ["neofoam", "solver", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
