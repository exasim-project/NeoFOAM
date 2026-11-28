import os
import re
import subprocess

# Typer renders --help via Rich. In a non-TTY CI run Rich still emits ANSI styling
# and wraps/truncates the options table to an 80-col width (so "--host" can render
# as "--ho…"), which breaks naive substring checks that pass locally on a wide TTY.
# Force a wide width + no color, then strip any residual ANSI before matching.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _help(*args: str) -> str:
    env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "200"}
    result = subprocess.run(["neofoam", *args], capture_output=True, text=True, env=env)
    assert result.returncode == 0
    return _ANSI.sub("", result.stdout)


def test_cli_hello() -> None:
    _help("solver", "--help")


def test_cli_mcp_help() -> None:
    _help("mcp", "--help")


def test_cli_mcp_serve_help_lists_host_and_port() -> None:
    out = _help("mcp", "serve", "--help")
    assert "--host" in out and "--port" in out
