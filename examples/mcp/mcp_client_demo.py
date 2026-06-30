# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Minimal MCP client that exercises the NeoFOAM MCP server.

Start the server in one terminal::

    PYTHONPATH=src python -m neofoam.cli.app mcp serve    # http://127.0.0.1:8000/mcp
    # (once `pip install '.[mcp]'` has shipped neofoam.mcp into the env, the
    #  console script works too:  `neofoam mcp serve`)

Then run this client in another::

    PYTHONPATH=src python examples/mcp/mcp_client_demo.py

It connects over Streamable-HTTP, lists the tools, then calls the read tools
(``list_solvers``, ``model_catalog``, ``config_schema``), reads an MCP resource,
and does a scaffolding round-trip (``load_case`` a real case -> ``save_case`` it
into a temp dir). No LLM is used (``fill_case`` is left out so the demo needs no
model credentials).
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = "http://127.0.0.1:8000/mcp"
# A real OpenFOAM case shipped in the repo, used for the load -> save round-trip.
SOURCE_CASE = "test/solver/incompressibleFluid/val_pitzDaily"


def _payload(result: Any) -> Any:
    """Pull the JSON payload out of an MCP tool/resource result.

    Tool results carry ``structuredContent`` and ``content``; resource reads
    carry ``contents``. Use ``getattr`` defaults so a missing attr is None
    rather than a Pydantic AttributeError.
    """
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    blocks = getattr(result, "content", None) or getattr(result, "contents", None)
    return json.loads(blocks[0].text)


async def main() -> None:
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print(f"connected to: {info.serverInfo.name}\n")

            tools = await session.list_tools()
            print("tools:", ", ".join(t.name for t in tools.tools), "\n")

            solvers = _payload(await session.call_tool("list_solvers", {}))
            print("list_solvers ->", solvers)

            catalog = _payload(await session.call_tool("model_catalog", {}))
            rows = catalog.get("result", catalog)  # tool returns a list
            print("model_catalog ->")
            for m in rows:
                flag = "required" if m["required"] else "optional"
                print(f"    {m['name']:<14} {flag:<8} owns={m['dicts'] + m['fields']}")

            schema = _payload(
                await session.call_tool("config_schema", {"name": "ControlDictConfig"})
            )
            print(
                "\nconfig_schema(ControlDictConfig) -> defaults:",
                schema["defaults"],
            )

            res = await session.read_resource("neofoam://solvers")
            print("\nresource neofoam://solvers ->", _payload(res))

            # scaffolding round-trip: load a real case, write it elsewhere
            loaded = _payload(
                await session.call_tool("load_case", {"case_dir": SOURCE_CASE})
            )
            keys = list(loaded["values"].keys())
            print(f"\nload_case({SOURCE_CASE}) -> config keys: {keys}")

            with tempfile.TemporaryDirectory() as tmp:
                saved = _payload(
                    await session.call_tool(
                        "save_case",
                        {"case_spec": loaded["values"], "target_dir": tmp},
                    )
                )
                written = saved["written"]
                print(f"save_case -> wrote {len(written)} files into a temp dir:")
                for w in written:
                    print("    ", w)
                # prove it actually hit disk
                print(
                    "    on-disk:",
                    [
                        str(p.relative_to(tmp))
                        for p in Path(tmp).rglob("*")
                        if p.is_file()
                    ],
                )


if __name__ == "__main__":
    asyncio.run(main())
