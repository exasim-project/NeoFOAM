# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for the pydantic-ai based AI agent scaffold (``neofoam.agent``).

The agent talks to an OpenAI-compatible endpoint. By default this is a local
Ollama server (``http://localhost:11434/v1``). The live test below actually
calls that server, so it is skipped automatically when no Ollama instance is
reachable, keeping the suite green in CI.
"""

import urllib.error
import urllib.request

import pytest

# The agent is an optional feature; skip the whole module if pydantic-ai
# (and therefore the agent extra) is not installed.
pytest.importorskip("pydantic_ai")

from neofoam.agent import ExampleResponse, build_model, create_agent  # noqa: E402


def _ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Return True if a local Ollama server answers within a short timeout."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="local Ollama server not reachable on http://localhost:11434",
)


def test_example_response_is_pydantic_schema() -> None:
    """The example output schema validates and exposes the expected field."""
    reply = ExampleResponse(summary="hello")
    assert reply.summary == "hello"
    # It must be a pydantic model so pydantic-ai can use it as output_type.
    assert hasattr(ExampleResponse, "model_validate")


def test_build_model_targets_ollama_by_default() -> None:
    """build_model returns a model wired to the local Ollama endpoint."""
    model = build_model()
    # No network call is made just by constructing the model.
    assert model is not None
    assert "llama" in model.model_name.lower()


def test_create_agent_uses_given_output_type() -> None:
    """create_agent returns a pydantic-ai Agent with our output schema."""
    from pydantic_ai import Agent

    agent = create_agent()
    assert isinstance(agent, Agent)
    assert agent.output_type is ExampleResponse


@requires_ollama
def test_agent_returns_structured_output_against_live_ollama() -> None:
    """End-to-end: the agent calls a real Ollama and returns ExampleResponse."""
    agent = create_agent()
    result = agent.run_sync(
        "Summarise in one short sentence: NeoFOAM is a CFD library."
    )
    assert isinstance(result.output, ExampleResponse)
    assert result.output.summary.strip()
