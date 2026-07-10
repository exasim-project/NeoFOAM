# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from neofoam.mcp.app import build_fastapi_app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def client():
    app = build_fastapi_app()
    with TestClient(app, base_url="http://127.0.0.1:8000") as c:
        yield c


def test_health_route_returns_ok(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_mcp_endpoint_is_mounted(client) -> None:
    # Streamable-HTTP expects POST; a GET must NOT 404 (proves it is mounted).
    resp = client.get("/mcp")
    assert resp.status_code != 404


def test_mcp_endpoint_has_no_trailing_slash_redirect(client) -> None:
    # ``/mcp`` (no slash) is the canonical endpoint and must serve directly,
    # NOT 307-redirect to ``/mcp/`` — a redirect breaks clients (e.g. the MCP
    # Inspector) that don't re-send the body/Accept headers on the redirect.
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        headers={"Accept": "application/json, text/event-stream"},
        follow_redirects=False,
    )
    assert resp.status_code != 307


def test_mcp_initialize_handshake_over_mcp(client) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    }
    resp = client.post(
        "/mcp",
        json=payload,
        headers={"Accept": "application/json, text/event-stream"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert resp.headers.get("mcp-session-id")
    data = next(
        json.loads(line[len("data: ") :])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    )
    assert data["result"]["serverInfo"]["name"] == "neofoam"
    assert data["result"]["protocolVersion"]


def test_import_without_dependencies_raises_import_error() -> None:
    """Without the ``mcp`` extra, importing the app module raises a plain
    ``ImportError`` naming the missing dependency (no custom guard — the import
    is allowed to fail naturally)."""
    script = textwrap.dedent(
        """
        import sys, importlib.abc
        BLOCKED = {"fastapi", "fastmcp", "mcp"}

        class _Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"No module named {name!r}")
                return None

        sys.meta_path.insert(0, _Blocker())
        for m in list(sys.modules):
            if m.split(".")[0] in BLOCKED:
                del sys.modules[m]
        try:
            import neofoam.mcp.app  # noqa: F401
        except ImportError as exc:
            assert any(d in str(exc) for d in ("fastapi", "fastmcp", "mcp")), str(exc)
            print("OK:", exc)
            sys.exit(0)
        sys.exit(1)
        """
    )
    src = str(REPO_ROOT / "src")
    import os

    # Inherit the real environment (LD_LIBRARY_PATH so pybFoam's native libs load —
    # it is a hard dep that the eager neofoam package import pulls); only the mcp-extra
    # packages are blocked, via the in-process meta_path finder above.
    env = dict(os.environ)
    env["PYTHONPATH"] = src
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert any(d in proc.stdout for d in ("fastapi", "fastmcp", "mcp"))
