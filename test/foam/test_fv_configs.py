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
from pydantic import ValidationError

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
    """Each field declares its entry plus the <field>Final companion.

    fvMatrix::solve selects the Final settings on the final outer
    iteration (in PISO mode: on every solve), so OpenFOAM needs both.
    """
    spec = Model("FvSolution")
    Sub = spec.config(fvSolution)
    Sub.add("U", "p")

    parsed = {
        "solvers": {
            "U": {"solver": "PBiCG"},
            "UFinal": {"solver": "PBiCG", "relTol": 0},
            "p": {"solver": "PCG"},
            "pFinal": {"solver": "PCG", "relTol": 0},
        }
    }
    inst = Sub.model_validate(parsed)
    assert inst.solvers.U == {"solver": "PBiCG"}
    assert inst.solvers.p == {"solver": "PCG"}
    assert inst.solvers.UFinal == {"solver": "PBiCG", "relTol": 0}
    assert inst.solvers.pFinal == {"solver": "PCG", "relTol": 0}


def test_fvsolution_add_requires_the_final_variant() -> None:
    spec = Model("FvSolutionNoFinal")
    Sub = spec.config(fvSolution)
    Sub.add("U")

    with pytest.raises(ValidationError):
        Sub.model_validate({"solvers": {"U": {"solver": "PBiCG"}}})


def test_fvsolution_passes_through_extra_sections() -> None:
    """PIMPLE / SIMPLE / relaxationFactors etc. pass through via extra='allow'."""
    spec = Model("FvSolutionExtra")
    Sub = spec.config(fvSolution)
    Sub.add("p")

    parsed = {
        "solvers": {"p": {"solver": "PCG"}, "pFinal": {"solver": "PCG"}},
        "PIMPLE": {"nOuterCorrectors": 1, "nCorrectors": 2},
    }
    inst = Sub.model_validate(parsed)
    assert inst.solvers.p == {"solver": "PCG"}
    # PIMPLE captured via extra="allow"
    assert getattr(inst, "PIMPLE") == {"nOuterCorrectors": 1, "nCorrectors": 2}


# ---------------------------------------------------------------------------
# form_defaults — runnable starter prefill for the required-but-defaultless fields
# ---------------------------------------------------------------------------


def test_fvschemes_form_defaults_is_a_valid_runnable_scaffold() -> None:
    # G2: the scheme fields are required-but-defaultless, so the form prefill would be
    # empty; form_defaults hands back a canonical scaffold that round-trips through the
    # subclass (a save-able starter), keyed by OpenFOAM alias.
    spec = Model("SchemeDefaults")
    Sub = spec.config(fvSchemes)
    Sub.add(
        ddt="ddt(U)",
        div=["div(phi,U)", "div((nuEff*dev2(T(grad(U)))))"],
        grad="grad(U)",
        laplacian="laplacian(nuEff,U)",
        snGrad="snGrad(p)",
        interpolation="interpolate(rAU)",
    )

    scaffold = Sub.form_defaults()
    assert scaffold is not None
    # a convection div (a flux) gets bounded upwind; the stress divergence stays linear
    assert scaffold["divSchemes"]["div(phi,U)"] == "Gauss upwind"
    assert scaffold["divSchemes"]["div((nuEff*dev2(T(grad(U)))))"] == "Gauss linear"
    assert scaffold["ddtSchemes"]["ddt(U)"] == "Euler"
    assert (
        scaffold["laplacianSchemes"]["laplacian(nuEff,U)"] == "Gauss linear corrected"
    )
    # and it actually validates as an instance of the config
    Sub.model_validate(scaffold)


def test_fvsolution_form_defaults_blocks_by_field_kind() -> None:
    spec = Model("SolverDefaults")
    Sub = spec.config(fvSolution)
    Sub.add("U", "p")

    scaffold = Sub.form_defaults()
    assert scaffold is not None
    solvers = scaffold["solvers"]
    # pressure → symmetric PCG/DIC; velocity → smoothSolver; Final tightens relTol to 0
    assert solvers["p"]["solver"] == "PCG"
    assert solvers["U"]["solver"] == "smoothSolver"
    assert solvers["pFinal"]["relTol"] == 0.0
    assert solvers["UFinal"]["relTol"] == 0.0
    Sub.model_validate(scaffold)


def test_fvsolution_form_defaults_includes_runnable_control_block() -> None:
    # A declared control section (PIMPLE) gets a runnable block — correctors keyed by
    # the section name plus the closed-domain pressure reference — so the starter runs
    # (the solver reads PIMPLE at run time though the schema models it as optional).
    spec = Model("SolverControls")
    Sub = spec.config(fvSolution)
    Sub.add("U", "p")
    Sub.add_controls("PIMPLE", pRefCell=int, pRefValue=float)

    scaffold = Sub.form_defaults()
    assert scaffold is not None
    pimple = scaffold["PIMPLE"]
    assert pimple["nCorrectors"] == 2
    assert pimple["pRefCell"] == 0 and pimple["pRefValue"] == 0
    Sub.model_validate(scaffold)


def test_form_defaults_none_when_nothing_declared() -> None:
    # An empty subclass has no required scheme/solver fields → no scaffold to offer.
    assert Model("EmptySchemes").config(fvSchemes).form_defaults() is None
    assert Model("EmptySolvers").config(fvSolution).form_defaults() is None


# ---------------------------------------------------------------------------
# OpenFOAM ``default`` shorthand — expands to the required per-operator keys
# ---------------------------------------------------------------------------


def test_default_shorthand_fills_missing_required_entries() -> None:
    """A section giving only ``default`` satisfies every required operator.

    OpenFOAM applies ``gradSchemes { default Gauss linear; }`` to every operator
    not spelled out. The typed per-operator fields are required, so without
    expansion a tutorial (or a minimally-authored case) that leans on ``default``
    fails to load. The before-validator fills the missing keys from ``default``.
    """
    spec = Model("DefaultFill")
    Sub = spec.config(fvSchemes)
    Sub.add(grad=["grad(U)", "grad(p)"])

    inst = Sub.model_validate({"gradSchemes": {"default": "Gauss linear"}})
    assert inst.gradSchemes.grad_U.type == "Gauss"
    assert inst.gradSchemes.grad_p.type == "Gauss"


def test_default_shorthand_does_not_override_explicit_entries() -> None:
    """Explicitly-named operators win; ``default`` only fills the gaps."""
    spec = Model("DefaultNoOverride")
    Sub = spec.config(fvSchemes)
    Sub.add(div=["div(phi,U)", "div(phi,T)"])

    inst = Sub.model_validate(
        {
            "divSchemes": {
                "default": "Gauss upwind",
                "div(phi,U)": "Gauss linearUpwind grad(U)",
            }
        }
    )
    # explicit override kept verbatim
    assert inst.divSchemes.div_phi_U.type == "Gauss"
    dumped = inst.model_dump(by_alias=True)["divSchemes"]
    assert dumped["div(phi,U)"] == "Gauss linearUpwind grad(U)"
    # the omitted operator is filled from the (real) default
    assert dumped["div(phi,T)"] == "Gauss upwind"


def test_default_none_sentinel_is_not_expanded() -> None:
    """``default none`` is OpenFOAM's *no-default* sentinel — it must not fabricate a
    ``none`` value for an unlisted operator; the missing required key still fails."""
    spec = Model("DefaultNoneSentinel")
    Sub = spec.config(fvSchemes)
    Sub.add(div=["div(phi,U)", "div(phi,T)"])

    with pytest.raises(ValidationError):
        # only div(phi,U) is spelled out; ``default none`` must NOT fill div(phi,T)
        Sub.model_validate(
            {"divSchemes": {"default": "none", "div(phi,U)": "Gauss upwind"}}
        )


def test_without_default_missing_required_entry_still_raises() -> None:
    """No ``default`` ⇒ a genuinely missing required operator still fails.

    Expansion must not weaken validation: a case that omits an operator and gives
    no ``default`` is incomplete and must be rejected, so validate_case keeps
    catching real gaps.
    """
    spec = Model("DefaultAbsent")
    Sub = spec.config(fvSchemes)
    Sub.add(grad=["grad(U)", "grad(p)"])

    with pytest.raises(ValidationError):
        Sub.model_validate({"gradSchemes": {"grad(U)": "Gauss linear"}})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
