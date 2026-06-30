# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""A minimal pydantic-ai agent scaffold for NeoFOAM.

This is intentionally generic: it wires a :class:`pydantic_ai.Agent` to an
OpenAI-compatible endpoint (a local `Ollama <https://ollama.com>`_ server by
default) and returns structured output validated against a pydantic schema.
Replace :class:`ExampleResponse` with a domain-specific schema and pass it via
``create_agent(output_type=...)`` to build a real NeoFOAM assistant on top.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

#: Default model served by a local Ollama instance.
DEFAULT_MODEL = "llama3.2"
#: Default OpenAI-compatible endpoint exposed by a local Ollama server.
DEFAULT_BASE_URL = "http://localhost:11434/v1"


class ExampleResponse(BaseModel):
    """Placeholder structured output for the scaffold.

    Swap this out for a domain schema (e.g. solver settings, a case config)
    once the agent's task is defined.
    """

    summary: str = Field(description="A short, one-sentence summary of the prompt.")


def build_model(
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> OpenAIChatModel:
    """Build a chat model backed by a local Ollama server.

    Uses :class:`~pydantic_ai.providers.ollama.OllamaProvider`, so **no API key
    is required** — Ollama serves models locally. Constructing the model
    performs no network I/O. Point ``base_url`` at any Ollama endpoint.
    """
    provider = OllamaProvider(base_url=base_url)
    return OpenAIChatModel(model_name, provider=provider)


def create_agent(
    output_type: type[BaseModel] = ExampleResponse,
    *,
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    **kwargs: Any,
) -> Agent[None, BaseModel]:
    """Create a pydantic-ai :class:`Agent` returning structured ``output_type``.

    Extra keyword arguments (``instructions``, ``system_prompt``, ``retries``,
    ...) are forwarded to :class:`pydantic_ai.Agent`.
    """
    model = build_model(model_name=model_name, base_url=base_url)
    return Agent(model, output_type=output_type, **kwargs)
