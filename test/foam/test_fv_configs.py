# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Phase 4 + 5 regression tests for ``neofoam.foam.fvSchemes`` /
``fvSolution`` per-spec subclass synthesis and ``.add(...)`` decorator.

Locks in:
- ``spec.config(fvSchemes)`` returns a per-spec subclass (not the base).
- Two specs that register ``fvSchemes`` get independent subclasses —
  ``.add(...)`` on one does not affect the other.
- ``.add(div="div(phi,U)")`` injects a typed Pydantic field whose alias
  matches the OpenFOAM key; ``populate_by_name=True`` lets Python code
  use the sanitized attribute name.
- The value side is validated against the scheme unions
  (``DdtScheme``, ``DivScheme``, …) — bad values raise.
- ``fvSolution.add("U", "p")`` adds entries under the ``solvers``
  sub-dictionary.
- ``.add(...)`` is composable and idempotent (re-adding a key is a
  no-op).
"""

import pytest

from neofoam.foam import fvSchemes, fvSolution
from neofoam.framework.model import Model
from neofoam.framework.solver import Solver


# ---------------------------------------------------------------------------
# Subclass synthesis
# ---------------------------------------------------------------------------


def test_spec_config_returns_subclass_not_base() -> None:
    spec = Model("PerSpec")
    Sub = spec.config(fvSchemes)
    assert Sub is not fvSchemes
    assert issubclass(Sub, fvSchemes)


def test_two_specs_get_independent_subclasses() -> None:
    spec_a = Model("A")
    spec_b = Model("B")
    SubA = spec_a.config(fvSchemes)
    SubB = spec_b.config(fvSchemes)
    assert SubA is not SubB


def test_subclass_inherits_io_strategy_binding() -> None:
    spec = Model("IOInherit")
    Sub = spec.config(fvSchemes)
    assert Sub.io_config is fvSchemes.io_config


def test_solver_spec_works_too() -> None:
    """spec.config(fvSchemes) is identical on a SolverSpec."""
    spec = Solver("SolverWithFvSchemes")
    Sub = spec.config(fvSchemes)
    assert issubclass(Sub, fvSchemes)
    assert Sub in spec._config_classes


# ---------------------------------------------------------------------------
# .add(...) classmethod — fvSchemes
# ---------------------------------------------------------------------------


def test_add_injects_typed_field() -> None:
    spec = Model("AddInject")
    Sub = spec.config(fvSchemes)

    @Sub.add(div="div(phi,U)")
    def _op() -> None:
        pass

    assert "divSchemes" in Sub.model_fields
    sec = Sub.model_fields["divSchemes"].annotation
    assert "div_phi_U" in sec.model_fields
    # DivScheme is Annotated[Union[...], ...]; Pydantic unwraps to the Union.
    # Confirm by validating a known-good value through the field.
    field_alias = sec.model_fields["div_phi_U"].alias
    assert field_alias == "div(phi,U)"
    inst = sec.model_validate({"div(phi,U)": "Gauss linear"})
    assert inst.div_phi_U.type == "Gauss"


def test_add_multiple_entries_in_one_call() -> None:
    spec = Model("MultiKwarg")
    Sub = spec.config(fvSchemes)

    @Sub.add(div="div(phi,U)", grad="grad(U)", laplacian="laplacian(nuEff,U)")
    def _op() -> None:
        pass

    parsed = {
        "divSchemes": {"default": "none", "div(phi,U)": "Gauss linear"},
        "gradSchemes": {"default": "Gauss linear", "grad(U)": "Gauss linear"},
        "laplacianSchemes": {
            "default": "Gauss linear corrected",
            "laplacian(nuEff,U)": "Gauss linear corrected",
        },
    }
    inst = Sub.model_validate(parsed)
    assert inst.divSchemes.div_phi_U is not None
    assert inst.gradSchemes.grad_U is not None
    assert inst.laplacianSchemes.laplacian_nuEff_U is not None


def test_add_accumulates_across_calls() -> None:
    spec = Model("Accumulate")
    Sub = spec.config(fvSchemes)

    @Sub.add(div="div(phi,U)")
    def _op1() -> None:
        pass

    @Sub.add(div="div(phi,p)", grad="grad(p)")
    def _op2() -> None:
        pass

    sec = Sub.model_fields["divSchemes"].annotation
    assert "div_phi_U" in sec.model_fields
    assert "div_phi_p" in sec.model_fields


def test_add_is_idempotent() -> None:
    spec = Model("Idempotent")
    Sub = spec.config(fvSchemes)

    @Sub.add(div="div(phi,U)")
    def _op1() -> None:
        pass

    @Sub.add(div="div(phi,U)")
    def _op2() -> None:
        pass

    sec = Sub.model_fields["divSchemes"].annotation
    assert list(sec.model_fields).count("div_phi_U") == 1


def test_add_bad_value_raises_at_validation() -> None:
    spec = Model("BadValue")
    Sub = spec.config(fvSchemes)
    Sub.add(div="div(phi,U)")

    with pytest.raises(Exception):
        Sub.model_validate({"divSchemes": {"div(phi,U)": "notARealScheme garbage"}})


def test_add_typed_value_round_trips() -> None:
    spec = Model("Round")
    Sub = spec.config(fvSchemes)
    Sub.add(div="div(phi,U)")

    inst = Sub.model_validate({"divSchemes": {"div(phi,U)": "Gauss linear"}})
    assert inst.divSchemes.div_phi_U.type == "Gauss"


def test_add_unknown_section_passes_through_as_string() -> None:
    """Short names not in the table — e.g. wallDist — get str typing."""
    spec = Model("Unknown")
    Sub = spec.config(fvSchemes)
    Sub.add(wallDist="method")

    inst = Sub.model_validate({"wallDist": {"method": "meshWave"}})
    assert inst.wallDist.method == "meshWave"


def test_two_specs_dont_share_added_entries() -> None:
    spec_a = Model("Isolated_A")
    spec_b = Model("Isolated_B")
    SubA = spec_a.config(fvSchemes)
    SubB = spec_b.config(fvSchemes)

    SubA.add(div="div(phi,U)")
    SubB.add(grad="grad(p)")

    assert "divSchemes" in SubA.model_fields
    assert "divSchemes" not in SubB.model_fields
    assert "gradSchemes" in SubB.model_fields
    assert "gradSchemes" not in SubA.model_fields


# ---------------------------------------------------------------------------
# .add(...) classmethod — fvSolution
# ---------------------------------------------------------------------------


def test_fvsolution_add_creates_solvers_section() -> None:
    spec = Model("FvSolution")
    Sub = spec.config(fvSolution)
    Sub.add("U", "p")

    parsed = {"solvers": {"U": {"solver": "PBiCG"}, "p": {"solver": "PCG"}}}
    inst = Sub.model_validate(parsed)
    assert inst.solvers.U == {"solver": "PBiCG"}
    assert inst.solvers.p == {"solver": "PCG"}


def test_fvsolution_passes_through_extra_sections() -> None:
    """PIMPLE / SIMPLE / relaxationFactors etc. pass through via extra='allow'."""
    spec = Model("FvSolutionExtra")
    Sub = spec.config(fvSolution)
    Sub.add("p")

    parsed = {
        "solvers": {"p": {"solver": "PCG"}},
        "PIMPLE": {"nOuterCorrectors": 1, "nCorrectors": 2},
    }
    inst = Sub.model_validate(parsed)
    assert inst.solvers.p == {"solver": "PCG"}
    # PIMPLE captured via extra="allow"
    assert getattr(inst, "PIMPLE") == {"nOuterCorrectors": 1, "nCorrectors": 2}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
