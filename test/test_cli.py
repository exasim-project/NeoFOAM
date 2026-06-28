import subprocess


def test_cli_hello() -> None:
    result = subprocess.run(
        ["neofoam", "solver", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0


def test_cli_mcp_help() -> None:
    result = subprocess.run(
        ["neofoam", "mcp", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0


def test_cli_mcp_serve_help_lists_host_and_port() -> None:
    result = subprocess.run(
        ["neofoam", "mcp", "serve", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--host" in result.stdout and "--port" in result.stdout
