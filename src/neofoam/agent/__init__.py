# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""pydantic-ai based AI agent scaffold for NeoFOAM.

Two layers live here:

- The generic scaffold (:mod:`neofoam.agent.agent`) wires a pydantic-ai
  :class:`~pydantic_ai.Agent` to an OpenAI-compatible endpoint (local
  Ollama by default).
- The case-fill API (:mod:`neofoam.agent.case_fill`) targets the
  ``incompressibleFluid`` solver: it builds an aggregate ``CaseSpec``
  pydantic model from every ``BaseConfig`` the solver may consume, and
  saves it back via the registered IO strategies — so an agent (or the
  no-LLM disk-roundtrip path used by tests / the ``--no-llm`` CLI flag)
  can fill a full case from a source case or natural-language prompt.
"""

from neofoam.agent.agent import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ExampleResponse,
    build_model,
    create_agent,
)
from neofoam.agent.case_fill import (
    DEFAULT_CASE_SYSTEM_PROMPT,
    build_case_agent,
    build_case_output_model,
    case_spec_to_configs,
    fill_case,
    load_case_from_disk,
    read_case_text,
    save_case,
)
from neofoam.agent.case_forms import (
    INPUT_KEYS,
    field_name,
    is_scheme_config,
    merge_field_config,
    split_field_dump,
)
from neofoam.agent.wizard_template import (
    NOTEBOOK_TEMPLATE,
    write_wizard_notebook,
)

__all__ = [
    # generic scaffold
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "ExampleResponse",
    "build_model",
    "create_agent",
    # case-fill API
    "DEFAULT_CASE_SYSTEM_PROMPT",
    "build_case_agent",
    "build_case_output_model",
    "case_spec_to_configs",
    "fill_case",
    "load_case_from_disk",
    "read_case_text",
    "save_case",
    # case form wiring
    "INPUT_KEYS",
    "field_name",
    "is_scheme_config",
    "merge_field_config",
    "split_field_dump",
    # notebook scaffolding
    "NOTEBOOK_TEMPLATE",
    "write_wizard_notebook",
]
