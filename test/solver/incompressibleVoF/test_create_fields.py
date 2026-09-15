# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the incompressibleVoF staged-init pipeline (``create_fields.py``).

Two levels, because the three stages have very different costs:

**LOAD / BUILD are pure description** — they read ``system/fvSolution`` and
return :class:`InitStep` objects; nothing is constructed. So they are driven
in-process against the two real advection fvSolutions already checked in under
``models/alpha_advection/cases/`` (the interFoam damBreak one, no
``advectionScheme`` key, and this repo's ``damBreak_isoAdvector`` one). Those
directories hold *nothing but* ``system/fvSolution`` — no mesh, no ``0/`` — so
the fact that BUILD returns a complete 19-step graph there is itself the proof
that every step is deferred: an eager step would abort on the missing mesh.

**BUILD's steps are executed once**, in a worker subprocess (one ``Foam::Time``
per process), on ``cases/vofRow4/common`` — see ``conftest.py`` for the fixture and the
case. The expected values are hand-derived from that case:

* the row is a unit cube cut into 4 cells along x, so cell volume is 0.25,
  every x-face area is 1 and every cell centre sits at y = 0.5;
* ``0/alpha.water`` is the literal ``(0 0.25 0.75 1)`` and the transport
  properties are water ``rho = 1000`` / air ``rho = 1``, so
  ``rho = 999*alpha1 + 1 = (1, 250.75, 750.25, 1000)`` (asserted in
  ``models/alpha_advection/test_shared.py``, which owns those two steps);
* ``g = (0 -9.81 0)`` and ``ghRef = 0``, so ``gh = -9.81*0.5 = -4.905`` in every
  cell, and with ``0/p_rgh`` uniform 100,
  ``p = p_rgh + rho*gh = (95.095, -1129.92875, -3579.97625, -4805)``. The case is
  closed (``pRefCell 0``, ``pRefValue 0``), so init then applies
  ``createFields.H``'s level shift of ``-95.095`` to both fields:
  ``p = (0, -1225.02375, -3675.07125, -4900.095)`` and ``p_rgh`` uniform
  ``4.905``;
* ``0/U`` is uniform ``(2 0 0)``, so ``phi = 2`` on every internal x-face and
  ``-2`` on the inlet (``Sf`` points out of the domain there).

Tolerances are ``rtol=1e-12`` throughout: every expectation is exact in binary
floating point except for OpenFOAM's geometric cell-centre/face-area
computation, so only round-off is being allowed for.

What is asserted through the *written* time directory (``written_dimensions`` /
``written_boundary``) rather than the JSON dump: pybFoam binds neither
``dimensions()`` nor ``boundaryField()`` on a GeometricField, so the only view
of those is the solver's own write path (``runtime.write(True)``). The worker
advances the clock one step before writing, so those files are the constructed
fields — they cannot be the case input the pipeline read.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from neofoam.framework.initialization import ConfigContext
from neofoam.solver.incompressibleVoF.create_fields import create_init

from .conftest import BuiltCase

_ADVECTION_CASES = Path(__file__).parent / "models" / "alpha_advection" / "cases"
_MULES_CASE = _ADVECTION_CASES / "damBreak_mules"

# The graph ``create_init`` emits for a MULES case: the argList/runtime/mesh
# triple, the two core models, the alpha-advection family's fields, PIMPLE's
# fields + controls, and the two-phase turbulence model last.
MULES_STEPS = [
    "models.foam_arglist",
    "runtime",
    "mesh",
    "models.alpha_advection",
    "models.pressure_velocity",
    "fields.phi",
    "models.mixture",
    "fields.alpha1",
    "fields.alpha2",
    "fields.rho",
    "fields.rhoPhi",
    "fields.alphaPhiUn",
    "fields.alphaPhi10",
    "models.alphaPhi1Corr0",
    "fields.U",
    "fields.p_rgh",
    "fields.hRef",
    "fields.gh",
    "fields.ghf",
    "fields.p",
    "models.pimple_control",
    "models.dynamic_mesh_controls",
    "models.Uf",
    "models.cumulativeContErr",
    "models.last_rAU",
    "models.pressure_reference",
    "models.initial_flux_correction",
    "models.turbulence",
]


# --------------------------------------------------------------------------- #
# LOAD                                                                         #
# --------------------------------------------------------------------------- #


def test_create_init_is_the_solver_runner_with_all_three_stages() -> None:
    # LOAD and BUILD are mandatory (the runner raises without them); RESOLVE is
    # optional in the framework, and incompressibleVoF does register one.
    runner = create_init(case_dir=Path("."))
    assert runner.name == "incompressibleVoF"
    assert [
        runner.spec.load_fn is not None,
        runner.spec.resolve_fn is not None,
        runner.spec.build_fn is not None,
    ] == [True, True, True]


@pytest.mark.parametrize(
    "case_name, pass_case_dir, expected",
    [
        # The damBreak fvSolution has no advectionScheme key -> the MULES
        # default; VoF always couples with PIMPLE. Advection first: it runs
        # before PIMPLE in every outer corrector.
        pytest.param("damBreak_mules", True, ["MULES", "Pimple"], id="mules"),
        # ``advectionScheme isoAdvector;`` swaps the advection member, nothing else.
        pytest.param("damBreak_isoAdvector", True, ["isoAdvector", "Pimple"], id="isoAdvector"),
        # ``case_dir=None`` -> Path("."). Only optional-model detection is handed
        # that path, and no optional model is registered today, so this pins the
        # default resolving at all (not a case-dir-specific outcome).
        pytest.param("damBreak_mules", False, ["MULES", "Pimple"], id="cwd_default"),
    ],
)
def test_load_selects_the_advection_scheme_the_case_asks_for(
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    pass_case_dir: bool,
    expected: list[str],
) -> None:
    case = _ADVECTION_CASES / case_name
    monkeypatch.chdir(case)
    runner = create_init(case_dir=case) if pass_case_dir else create_init()
    assert [model.name for model in runner.run_load().core_models] == expected


def test_load_result_carries_the_config_classes_and_no_optional_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The with-a-case config surface: MULES declares its fvSchemes/fvSolution
    # slices and the 0/alpha.water field, PIMPLE its slices, the typed
    # ``PIMPLE`` block (loop controls + createDyMControls switches) the slices
    # pass through untyped, and U/p_rgh. Nothing in the tree registers with
    # incompressibleVoFModel yet, so the optional-model half of the LoadResult
    # is empty (not absent).
    monkeypatch.chdir(_MULES_CASE)
    load_result = create_init(case_dir=_MULES_CASE).run_load()
    assert [cls.__name__ for cls in load_result.config_classes] == [
        "MULES_fvSchemes",
        "MULES_fvSolution",
        "alpha.waterFieldConfig",
        "Pimple_fvSchemes",
        "Pimple_fvSolution",
        "VofPimpleAlgorithmConfig",
        "DynamicMeshControls",
        "UFieldConfig",
        "p_rghFieldConfig",
    ]
    assert load_result.optional_models == []


# --------------------------------------------------------------------------- #
# RESOLVE / BUILD (description only — nothing is constructed)                  #
# --------------------------------------------------------------------------- #


def test_build_describes_the_whole_graph_without_constructing_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``cases/damBreak_mules`` is *only* a system/fvSolution — no polyMesh, no
    # 0/ directory. A BUILD that constructed anything eagerly (Foam::Time, mesh,
    # a field read) could not survive here; a fully lazy one returns the graph.
    #
    # RESOLVE forwards the ConfigContext to every *optional* model; the two core
    # models are wired by their own specs. With no optional model registered it
    # is a no-op, so BUILD still yields the whole graph afterwards.
    assert not (_MULES_CASE / "constant").exists()
    monkeypatch.chdir(_MULES_CASE)
    runner = create_init(case_dir=_MULES_CASE)
    runner.run_load()
    assert [step.name for step in runner.run_build()] == MULES_STEPS
    runner.run_resolve(ConfigContext())
    assert [step.name for step in runner.run_build()] == MULES_STEPS


def test_build_of_an_isoadvector_case_adds_the_advector_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # isoAdvector owns the same shared fields plus one persistent model, and
    # none of MULES' own ``alphaPhi1Corr0`` compression-flux cache.
    case = _ADVECTION_CASES / "damBreak_isoAdvector"
    monkeypatch.chdir(case)
    runner = create_init(case_dir=case)
    runner.run_load()
    names = [step.name for step in runner.run_build()]
    assert MULES_STEPS[13] == "models.alphaPhi1Corr0"
    assert names == MULES_STEPS[:13] + ["models.advector"] + MULES_STEPS[14:]


def test_turbulence_step_depends_on_the_fields_it_wraps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TwoPhaseTransportModel(rho, U, phi, rhoPhi, mixture) — the declared
    # dependencies are what orders it behind both core models' fields.
    monkeypatch.chdir(_MULES_CASE)
    runner = create_init(case_dir=_MULES_CASE)
    runner.run_load()
    step = next(s for s in runner.run_build() if s.name == "models.turbulence")
    assert step.depends_on == [
        "fields.rho",
        "fields.U",
        "fields.phi",
        "fields.rhoPhi",
        "models.mixture",
    ]


# --------------------------------------------------------------------------- #
# The executed pipeline on cases/vofRow4/common                               #
# --------------------------------------------------------------------------- #


def test_the_pipeline_builds_every_field_and_model_of_the_vof_solver(
    vof_row4: BuiltCase,
) -> None:
    # The models are both core models under their solver-facing names, the
    # mixture and turbulence models, the PIMPLE controls (including the
    # mesh-motion switches and the face velocity ``Uf``, which is None on this
    # static case), the Foam::Time (which has no Context slot of its own and
    # lands in models["runtime"]) and the argList it was built from (held so the
    # MPI session outlives the run — see ``foam.initialization.create_arglist``).
    #
    # ctx.fields is keyed by the solver's names; the phase fractions carry the
    # case's phase names (transportProperties: ``phases (water air)``).
    assert vof_row4.result["field_keys"] == [
        "U",
        "alpha1",
        "alpha2",
        "alphaPhi10",
        "alphaPhiUn",
        "gh",
        "ghf",
        "hRef",
        "p",
        "p_rgh",
        "phi",
        "rho",
        "rhoPhi",
    ]
    assert vof_row4.result["model_keys"] == [
        "Uf",
        "alphaPhi1Corr0",
        "alpha_advection",
        "cumulativeContErr",
        "dynamic_mesh_controls",
        "foam_arglist",
        "initial_flux_correction",
        "last_rAU",
        "mixture",
        "pimple_control",
        "pressure_reference",
        "pressure_velocity",
        "runtime",
        "turbulence",
    ]
    assert vof_row4.result["registered_names"] == {
        "U": "U",
        "alpha1": "alpha.water",
        "alpha2": "alpha.air",
        "alphaPhi10": "alphaPhi10",
        "alphaPhiUn": "alphaPhiUn",
        "gh": "gh",
        "ghf": "ghf",
        "hRef": "hRef",
        "p": "p",
        "p_rgh": "p_rgh",
        "phi": "phi",
        "rho": "rho",
        "rhoPhi": "rhoPhi",
    }


def test_alpha_water_is_read_from_the_case_with_its_boundary_conditions(
    vof_row4: BuiltCase,
) -> None:
    # 0/alpha.water is the literal (0 0.25 0.75 1) — the mixture reads it — and
    # gives the inlet an inletOutlet BC and the walls zeroGradient; the
    # constructed field carries both, dimensionless.
    assert vof_row4.internal("alpha1") == [0.0, 0.25, 0.75, 1.0]
    inlet = vof_row4.written_boundary("alpha.water", "inlet")
    walls = vof_row4.written_boundary("alpha.water", "walls")
    assert (
        str(inlet.getOrDefault[str]("type", "")),
        str(walls.getOrDefault[str]("type", "")),
    ) == ("inletOutlet", "zeroGradient")
    assert vof_row4.written_dimensions("alpha.water") == [0, 0, 0, 0, 0, 0, 0]


def test_velocity_is_read_from_the_cases_zero_directory(
    vof_row4: BuiltCase,
) -> None:
    # 0/U is uniform (2 0 0) with dimensions [0 1 -1 0 0 0 0].
    assert vof_row4.internal("U") == [[2.0, 0.0, 0.0]] * 4
    assert vof_row4.written_dimensions("U") == [0, 1, -1, 0, 0, 0, 0]


def test_p_rgh_is_read_from_the_cases_zero_directory(vof_row4: BuiltCase) -> None:
    # 0/p_rgh is uniform 100 with dimensions [1 -1 -2 0 0 0 0]. The case is
    # closed, so createFields.H's start-up levelling then shifts it uniformly by
    # pRefValue - p[0] = -95.095, to 100 - 95.095 = 4.905 — a value only a field
    # that was actually read can reach (an unread p_rgh would level to 0).
    assert_allclose(
        vof_row4.internal("p_rgh"),
        [4.905] * 4,
        rtol=1e-12,
        err_msg="vofRow4: p_rgh must be 0/p_rgh, levelled against the reference cell",
    )
    assert vof_row4.written_dimensions("p_rgh") == [1, -1, -2, 0, 0, 0, 0]


def test_phi_is_the_face_flux_of_the_velocity_field(vof_row4: BuiltCase) -> None:
    # U = (2 0 0) on x-faces of area 1: every internal face carries +2, and the
    # inlet face carries -2 (Sf points out of the domain).
    assert vof_row4.internal("phi") == [2.0, 2.0, 2.0]
    assert vof_row4.written_dimensions("phi") == [0, 3, -1, 0, 0, 0, 0]
    inlet = vof_row4.written_boundary("phi", "inlet")
    assert str(inlet.getOrDefault[str]("value", "")) == "uniform -2"


@pytest.mark.parametrize(
    "case_fixture, expected_href, expected_gh",
    [
        # vofRow4 ships no constant/hRef -> readhRef.H's READ_IF_PRESENT default
        # (dimensions dimLength, value 0), so ghRef is 0 and gh = g&C is
        # -9.81*0.5 at every cell centre.
        pytest.param("vof_row4", 0.0, -4.905, id="hRef_absent"),
        # The `href` overlay ships constant/hRef = 0.3, and gh.H folds it in as
        # ghRef = g & (cmptMag(g)/mag(g))*hRef = (0,-9.81,0) & (0,0.3,0)
        # = -2.943, so gh = g&C - ghRef = -4.905 - (-2.943) = -1.962 everywhere.
        pytest.param("vof_row4_href", 0.3, -1.962, id="hRef_from_constant"),
    ],
)
def test_href_and_the_gravitational_head_follow_constant_hRef(
    request: pytest.FixtureRequest,
    case_fixture: str,
    expected_href: float,
    expected_gh: float,
) -> None:
    built: BuiltCase = request.getfixturevalue(case_fixture)
    # hRef must be registered under its own name for BC lookups
    # (prghPermeableAlphaTotalPressure etc.).
    assert built.result["registered_names"]["hRef"] == "hRef"
    assert built.internal("hRef") == expected_href
    assert_allclose(
        built.internal("gh"),
        [expected_gh] * 4,
        rtol=1e-12,
        err_msg=f"{case_fixture}: gh must be g&C folded against ghRef",
    )


def test_ghf_is_the_gravitational_head_at_the_internal_faces(
    vof_row4: BuiltCase,
) -> None:
    # The three internal faces are x-normal, all centred at y = 0.5.
    assert_allclose(
        vof_row4.internal("ghf"),
        [-4.905] * 3,
        rtol=1e-12,
        err_msg="vofRow4: ghf must be g&Cf on the x-normal internal faces",
    )


def test_absolute_pressure_is_p_rgh_plus_the_hydrostatic_head(
    vof_row4: BuiltCase,
) -> None:
    # p = p_rgh + rho*gh = 100 - 4.905*(1, 250.75, 750.25, 1000), then levelled
    # against the reference cell (createFields.H): pRefCell 0 / pRefValue 0, so
    # the whole field shifts by -95.095.
    assert_allclose(
        vof_row4.internal("p"),
        [0.0, -1225.02375, -3675.07125, -4900.095],
        rtol=1e-12,
        err_msg="vofRow4: p must be p_rgh + rho*gh, not p_rgh or rho*gh alone",
    )


def test_the_pimple_dict_is_read_into_the_controls_and_the_pressure_reference(
    vof_row4: BuiltCase,
) -> None:
    # Every p_rgh patch is zeroGradient, so the closed domain needs a reference
    # cell (``needsRef``); system/fvSolution's PIMPLE dict carries pRefCell 0 /
    # pRefValue 0, plus nOuterCorrectors 1, nCorrectors 3, momentumPredictor no.
    # The open/closed distinction itself is pinned in
    # ``models/pressure_velocity/test_pressure_reference.py``.
    assert vof_row4.result["pressure_reference"] == {
        "cell": 0,
        "value": 0.0,
        "needs_ref": True,
    }
    assert vof_row4.result["pimple_control"] == {
        "nOuterCorrectors": 1,
        "nCorrectors": 3,
        "momentumPredictor": False,
    }


def test_the_pipeline_seeds_the_turbulence_model_and_the_continuity_error(
    vof_row4: BuiltCase,
) -> None:
    assert vof_row4.result["turbulence_type"] == "TwoPhaseTransportModel"
    # A one-element list so the continuity operation can accumulate in place.
    assert vof_row4.result["cumulativeContErr"] == [0.0]
    # incompressibleVoF writes through OpenFOAM (``runtime.write(True)`` in
    # write_output), not through the Python per-field writer, so no init step
    # sets write=True and ctx.write_fields stays empty. (The write=True flags on
    # the ModelSpec field *declarations* are the case-wizard schema, a different
    # thing.)
    assert vof_row4.result["write_fields"] == []
