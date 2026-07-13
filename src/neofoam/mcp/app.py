# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The FastAPI app + uvicorn entry point wrapping the module-level FastMCP server.

Mounts FastMCP's Streamable-HTTP ASGI app (``mcp.http_app(path="/mcp")``) at the
root and adds a ``GET /health`` route ahead of it, so the public MCP endpoint is
exactly ``/mcp`` (no trailing-slash redirect). Imports the ``mcp`` extra — see
:mod:`neofoam.mcp.server`.
"""

from __future__ import annotations

from fastapi import FastAPI

from neofoam.mcp.server import configure_root, mcp


def build_fastapi_app(*, name: str = "neofoam", root: str | None = None) -> FastAPI:
    """FastAPI app exposing the MCP endpoint at ``/mcp`` plus a ``/health`` route.

    When ``root`` is given, every filesystem tool confines its path arguments under
    it (relative paths only, escapes rejected) — the trust boundary for an untrusted
    client. ``None`` leaves paths unconfined (trusted/local use).
    """
    configure_root(root)
    mcp_app = mcp.http_app(path="/mcp")  # Streamable-HTTP ASGI app, endpoint at /mcp

    app = FastAPI(title=name, lifespan=mcp_app.lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.mount("/", mcp_app)  # /mcp handled here; /health matched first (above)
    return app


def serve(
    *, host: str = "127.0.0.1", port: int = 8000, root: str | None = None
) -> None:
    """Blocking: run :func:`build_fastapi_app` under uvicorn (one process).

    ``root`` confines every filesystem tool's paths under that directory; omit it only
    for trusted/local use (see :func:`build_fastapi_app`).
    """
    import uvicorn

    uvicorn.run(build_fastapi_app(root=root), host=host, port=port)
