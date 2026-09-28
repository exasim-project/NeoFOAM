<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 NeoFOAM authors
-->

# NeoFOAM MCP server — try it out

The MCP server exposes NeoFOAM solver introspection + case scaffolding as MCP
tools/resources over Streamable-HTTP (FastAPI + FastMCP).

> **Note:** `neofoam.mcp` is newer than the last `pip install`, so run from the
> source tree with `PYTHONPATH=src`. Deps are in the `mcp` extra
> (`pip install '.[mcp]'` → `mcp`, `fastapi`, `uvicorn`).

## 1. Start the server

```bash
PYTHONPATH=src python -m neofoam.cli.app mcp serve            # http://127.0.0.1:8000
#   --host 0.0.0.0   --port 8000   --solver incompressibleFluid
```

Health check: `curl http://127.0.0.1:8000/health` → `{"status":"ok"}`.
The MCP endpoint is `http://127.0.0.1:8000/mcp`.

## 2a. Drive it from Python (this demo)

```bash
PYTHONPATH=src python examples/mcp/mcp_client_demo.py
```

It connects, lists tools, calls `list_solvers` / `model_catalog` /
`config_schema`, reads the `neofoam://solvers` resource, and does a
`load_case` → `save_case` round-trip on a real case into a temp dir.

## 2b. Drive it from raw HTTP (curl)

MCP is JSON-RPC over POST; the response is an SSE stream, and you must accept
both content types. First `initialize`, then call a tool:

```bash
# initialize (grab the mcp-session-id header from the response)
curl -sD - http://127.0.0.1:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
        "protocolVersion":"2025-06-18","capabilities":{},
        "clientInfo":{"name":"curl","version":"0"}}}'

# then list tools / call one, passing -H "mcp-session-id: <id>"
curl -s http://127.0.0.1:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'mcp-session-id: <id-from-above>' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call",
       "params":{"name":"list_solvers","arguments":{}}}'
```

## 2c. Add it to an MCP host (Claude Code / Desktop)

Point your host at the HTTP endpoint, e.g. Claude Code:

```bash
claude mcp add --transport http neofoam http://127.0.0.1:8000/mcp
```

## Tools exposed

`list_solvers`, `model_catalog`, `toggle_models`, `list_configs`,
`config_schema` (read) · `read_case`, `load_case`, `save_case` (scaffold) ·
`fill_case` (LLM; needs a model backend). Resources: `neofoam://solvers`,
`neofoam://{solver}/catalog`, `neofoam://{solver}/config/{name}/schema`.
There is intentionally **no run-the-solver tool**.
