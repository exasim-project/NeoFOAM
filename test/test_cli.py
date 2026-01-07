import subprocess


def test_cli_hello() -> None:
    result = subprocess.run(
        ["foamadapter", "solver", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0


def test_cli_incompressiblefluid_help() -> None:
    result = subprocess.run(
        ["foamadapter", "solver", "incompressiblefluid", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "incompressiblefluid" in result.stdout.lower()
    assert "Modular incompressible fluid solver" in result.stdout
