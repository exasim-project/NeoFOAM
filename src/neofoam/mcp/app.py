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

from neofoam.mcp.server import mcp


def build_fastapi_app(*, name: str = "neofoam") -> FastAPI:
    """FastAPI app exposing the MCP endpoint at ``/mcp`` plus a ``/health`` route."""
    mcp_app = mcp.http_app(path="/mcp")  # Streamable-HTTP ASGI app, endpoint at /mcp

    app = FastAPI(title=name, lifespan=mcp_app.lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.mount("/", mcp_app)  # /mcp handled here; /health matched first (above)
    return app


def serve(*, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Blocking: run :func:`build_fastapi_app` under uvicorn (one process)."""
    import uvicorn

    uvicorn.run(build_fastapi_app(), host=host, port=port)
