# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
End-to-end tests for the DummySolver 3-stage initialization.

Internal StagedInitSpec / StagedInitRunner behavior is covered under
``test/framework/initialization/staged/``; this file only exercises the
full LOAD → RESOLVE → BUILD flow via ``dummy_init.create_init``.
"""

from .dummy_init import create_init


def test_dummy_init_staged_full_run() -> None:
    """Complete 3-stage initialization produces a Context with fields/models/mesh."""
    init_instance = create_init()
    init_instance.argv = []

    ctx = init_instance.run()

    assert "field1" in ctx.fields
    assert "field2" in ctx.fields
    assert "field3" in ctx.fields
    assert ctx.fields["field1"] == 1.0
    assert ctx.fields["field2"] == 101325.0
    assert ctx.fields["field3"] == 0.01

    assert "algorithm" in ctx.models
    assert "core2" in ctx.models
    assert "config" in ctx.models

    assert ctx.mesh["name"] == "test_domain"
    assert ctx.mesh["nPoints"] == 500


def test_optional_models_integration() -> None:
    """Optional models are detected, resolved, and produce InitSteps."""
    init_instance = create_init()
    init_instance.argv = []

    ctx = init_instance.run()

    optional_model_keys = ["model_field1", "model_field2", "model_field3"]
    detected_optional_fields = [k for k in optional_model_keys if k in ctx.fields]

    # The DummySolver always registers the optional family, so these hold
    # unconditionally — no guard that could silently skip the assertion.
    optional_models = init_instance.optional_models
    assert len(optional_models) > 0
    assert len(detected_optional_fields) > 0

    algorithm = ctx.models.get("algorithm")
    assert algorithm is not None

    from neofoam.framework.model import ModelRuntime

    rt_m1 = next(
        (
            m
            for m in optional_models
            if isinstance(m, ModelRuntime) and "Model1" in m.spec.name
        ),
        None,
    )
    assert rt_m1 is not None
    assert rt_m1.config is not None
