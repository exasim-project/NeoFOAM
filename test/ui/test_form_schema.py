# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless tests for the JSON-Schema → JSONForms transform (no trame, no JS)."""

from __future__ import annotations

import pytest

from neofoam.mcp import tools
from neofoam.mcp.registry import list_solver_names, resolve_solver
from neofoam.ui.forms import build_field_forms, build_forms, build_mesh_forms


def _titles(node, path=""):
    """Every ``(json-pointer, title)`` in a schema, however deeply nested."""
    if isinstance(node, dict):
        title = node.get("title")
        if isinstance(title, str):
            yield path, title
        for key, value in node.items():
            yield from _titles(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _titles(value, f"{path}/{index}")


def test_jsonforms_transform_unwraps_optional_and_titles_unions(solver):
    from neofoam.io.pydantic_schema import slice_schema  # noqa: PLC0415
    from neofoam.ui.form_schema import jsonforms_schema  # noqa: PLC0415

    # Optional[X] (anyOf[X, null]) collapses to X — turbulence RAS/LES.
    turb = jsonforms_schema(tools.config_schema(solver, "turbulence_properties_config").json_schema)
    assert "anyOf" not in turb["properties"]["RAS"]
    assert "anyOf" not in turb["properties"]["LES"]
    # Discriminated BC union becomes a oneOf titled by the const `type`.
    bc = jsonforms_schema(
        slice_schema(tools.config_schema(solver, "u_field_config").json_schema, ["boundaryField"])
    )
    arms = bc["properties"]["boundaryField"]["additionalProperties"]["oneOf"]
    titles = {a.get("title") for a in arms}
    assert {"fixedValue", "noSlip", "zeroGradient"} <= titles


def test_openfoam_dict_keys_are_titled_verbatim(solver):
    # Keys of a synthesized fvSchemes/fvSolution section are OpenFOAM entries the
    # case author wrote, not an API surface: humanizing them destroys information a
    # user cannot recover (p_rghFinal read as "P Rghfinal") and no longer names
    # anything in the written case file.
    entries = {e.key: e for e in build_forms(solver)}
    # The buoyant pressure solvers sit on the Boussinesq model's own slice.
    solvers = {
        **entries["dict:Pimple_fvSolution"].schema["properties"]["solvers"]["properties"],
        **entries["dict:boussinesq_fvSolution"].schema["properties"]["solvers"]["properties"],
    }
    assert [solvers[k]["title"] for k in ("p", "pFinal", "p_rgh", "p_rghFinal")] == [
        "p",
        "pFinal",
        "p_rgh",
        "p_rghFinal",
    ]
    div = entries["dict:Pimple_fvSchemes"].schema["properties"]["divSchemes"]["properties"]
    assert div["div(phi,U)"]["title"] == "div(phi,U)"
    # A dotted phase field survives too ("Alpha.water" before).
    vof = {e.key: e for e in build_forms(resolve_solver("incompressibleVoF"))}
    alpha = vof["dict:MULES_fvSolution"].schema["properties"]["solvers"]["properties"]
    assert alpha["alpha.water"]["title"] == "alpha.water"


def test_section_headings_drop_the_synthesized_class_name(solver):
    # fv_configs._rebuild_sections names each section model "_<section>"; that
    # leading underscore must not reach the panel ("_ddtSchemes", "_PIMPLE").
    entries = {e.key: e for e in build_forms(solver)}
    schemes = entries["dict:Pimple_fvSchemes"].schema["properties"]
    assert schemes["ddtSchemes"]["title"] == "ddtSchemes"
    solution = entries["dict:Pimple_fvSolution"].schema["properties"]
    assert solution["PIMPLE"]["title"] == "PIMPLE"
    assert solution["solvers"]["title"] == "solvers"
    # No private class name leaks anywhere, in any solver.
    for name in list_solver_names():
        for entry in build_forms(resolve_solver(name)):
            for path, title in _titles(entry.schema):
                assert not title.startswith("_"), f"{name}/{entry.key}{path}: {title}"


def test_sections_of_like_entries_keep_the_add_a_key_row(solver):
    # JSONForms offers an "add a key" row for `additionalProperties`. A section is a
    # collection — divSchemes gains a div(phi,k), solvers a new solver block — so the
    # row stays there, single-entry sections included.
    entries = {e.key: e for e in build_forms(solver)}
    schemes = entries["dict:Pimple_fvSchemes"].schema["properties"]
    assert schemes["divSchemes"]["additionalProperties"] is True
    assert schemes["ddtSchemes"]["additionalProperties"] is True
    solvers = entries["dict:Pimple_fvSolution"].schema["properties"]["solvers"]
    assert solvers["additionalProperties"] is True


def test_fixed_shape_cards_offer_no_add_a_key_row(solver):
    # `additionalProperties: true` on a card of differently-shaped entries can only
    # add an untyped value — noise. Dropping the keyword hides the row and still
    # validates the same data (absent means "allowed").
    entries = {e.key: e for e in build_forms(solver)}
    solution = entries["dict:Pimple_fvSolution"].schema
    assert "additionalProperties" not in solution
    assert "additionalProperties" not in solution["properties"]["PIMPLE"]
    assert "additionalProperties" not in entries["dict:Pimple_fvSchemes"].schema
    vof = {e.key: e for e in build_forms(resolve_solver("incompressibleVoF"))}
    water = vof["dict:TransportPropertiesConfig"].schema["properties"]["water"]
    assert "additionalProperties" not in water


def test_open_dicts_without_known_keys_keep_the_add_a_key_row(solver):
    # A solver block (and MRFProperties / fvOptions) is an open dict in the config: the
    # form pins the standard solver controls, every other option is an additional
    # property, so the row is the only way to add one.
    entries = {e.key: e for e in build_forms(solver)}
    block = entries["dict:Pimple_fvSolution"].schema["properties"]["solvers"]["properties"]["p"]
    assert block["additionalProperties"] is True
    # fvOptions takes sub-dictionaries only, each open in turn.
    source = entries["dict:FvOptionsConfig"].schema["additionalProperties"]
    assert (source["type"], source["additionalProperties"]) == ("object", True)


def _keyword_sites(node, keyword, path=""):
    """``{json-pointer: value}`` for every schema node carrying ``keyword``."""
    found = {}
    if isinstance(node, dict):
        if keyword in node:
            found[path] = node[keyword]
        for key, value in node.items():
            found.update(_keyword_sites(value, keyword, f"{path}/{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.update(_keyword_sites(value, keyword, f"{path}/{index}"))
    return found


def test_add_a_key_rows_are_labelled_for_what_they_add(solver):
    # JSONForms labels every such row "Property Name"; its `i18n` schema keyword picks
    # the translation prefix, so each kind of collection says what is typed there.
    entries = {e.key: e for e in build_forms(solver)}
    assert _keyword_sites(entries["field_bc:UFieldConfig"].schema, "i18n") == {
        "/properties/boundaryField": "nf.patch"
    }
    schemes = _keyword_sites(entries["dict:Simple_fvSchemes"].schema, "i18n")
    assert schemes["/properties/divSchemes"] == "nf.entry"
    assert set(schemes.values()) == {"nf.entry"}
    assert _keyword_sites(entries["dict:Simple_fvSolution"].schema, "i18n") == {
        "/properties/solvers": "nf.solver",
        "/properties/solvers/properties/U": "nf.option",
        "/properties/solvers/properties/UFinal": "nf.option",
        "/properties/solvers/properties/p": "nf.option",
        "/properties/solvers/properties/pFinal": "nf.option",
    }


@pytest.mark.parametrize(
    ("entry_key", "prefix", "label"),
    [
        ("dict:MRFPropertiesConfig", "nf.zone", "Zone name, e.g. MRF1"),
        ("dict:FvOptionsConfig", "nf.source", "Source name, e.g. momentumSource"),
    ],
)
def test_open_config_renders_as_named_cards_of_keywords(entry_key, prefix, label, solver):
    # A config of free-form sub-dictionaries declares no property, so JSONForms'
    # generated layout is empty and the panel showed nothing. The bundled section
    # renderer draws it instead: one card per sub-dictionary (`nfDicts`), each a grid
    # of its keywords (`nfDict`), both adders labelled.
    from neofoam.ui.form_schema import ADDER_TRANSLATIONS  # noqa: PLC0415

    entry = {e.key: e for e in build_forms(solver)}[entry_key]
    assert _keyword_sites(entry.schema, "i18n") == {
        "": prefix,
        "/additionalProperties": "nf.keyword",
    }
    assert _keyword_sites(entry.schema, "nfDicts") == {"": True}
    assert _keyword_sites(entry.schema, "nfDict") == {"/additionalProperties": True}
    assert ADDER_TRANSLATIONS[f"{prefix}.propertyNameLabel"] == label
    assert ADDER_TRANSLATIONS["nf.keyword.propertyNameLabel"] == "Keyword, e.g. cellZone"


@pytest.mark.parametrize("solver_name", list_solver_names())
def test_every_add_a_key_row_of_the_wizard_is_labelled(solver_name):
    # An object that takes new keys and carries no `i18n` prefix would fall back to
    # JSONForms' "Property Name".
    for entry in build_forms(resolve_solver(solver_name)):
        open_sites = {
            path
            for path, value in _keyword_sites(entry.schema, "additionalProperties").items()
            if value is not False
        }
        assert open_sites <= set(_keyword_sites(entry.schema, "i18n")), entry.key


@pytest.mark.parametrize(
    ("prefix", "text"),
    [
        ("nf.patch", "Add patch"),
        ("nf.entry", "Add entry"),
        ("nf.solver", "Add solver"),
        ("nf.option", "Add option"),
        ("nf.zone", "Add zone"),
        ("nf.source", "Add source"),
        ("nf.keyword", "Add keyword"),
    ],
)
def test_adder_button_names_what_it_adds(prefix, text):
    from neofoam.ui.form_schema import ADDER_TRANSLATIONS  # noqa: PLC0415

    assert ADDER_TRANSLATIONS[f"{prefix}.addLabel"] == text


def test_adder_translations_cover_every_emitted_prefix():
    from neofoam.ui.form_schema import ADDER_TRANSLATIONS  # noqa: PLC0415

    assert ADDER_TRANSLATIONS["nf.patch.propertyNameLabel"] == "Patch name, e.g. inlet"
    assert ADDER_TRANSLATIONS["nf.entry.propertyNameLabel"] == "Entry, e.g. div(phi,k)"
    assert ADDER_TRANSLATIONS["nf.option.propertyNameLabel"] == "Option, e.g. maxIter"
    for name in list_solver_names():
        for entry in build_forms(resolve_solver(name)):
            for prefix in _keyword_sites(entry.schema, "i18n").values():
                assert f"{prefix}.propertyNameLabel" in ADDER_TRANSLATIONS
                assert f"{prefix}.addLabel" in ADDER_TRANSLATIONS


@pytest.mark.parametrize("solver_name", list_solver_names())
def test_only_scheme_sections_render_as_compact_rows(solver_name):
    # `nfCompact` switches a section to the one-row-per-entry renderer. It belongs on
    # the fvSchemes sections (every entry a scheme union) and nowhere else — BC maps and
    # the model dicts keep the stock rendering, fvSolution solvers have their own flags.
    for entry in build_forms(resolve_solver(solver_name)):
        compact = _keyword_sites(entry.schema, "nfCompact")
        if not entry.cls_name.endswith("fvSchemes"):
            assert compact == {}, entry.key
            continue
        sections = {f"/properties/{name}" for name in entry.schema["properties"]}
        assert set(compact) == sections, entry.key
        assert set(compact.values()) == {True}


_SCALAR_GRIDS = {
    "incompressibleFluid": {
        "dict:Pimple_fvSolution": ["/properties/PIMPLE"],
        "dict:PimpleAlgorithmConfig": [""],
        "dict:DynamicMeshControls": [""],
        "dict:Simple_fvSolution": ["/properties/SIMPLE"],
        "dict:SimpleAlgorithmConfig": [""],
        "dict:KEpsilonCoeffs": [""],
        "dict:KOmegaSSTCoeffs": [""],
        "dict:SpalartAllmarasCoeffs": [""],
        "dict:BoussinesqConfig": [""],
    },
    "incompressibleFluidNeoN": {
        "dict:PimpleNeoN_fvSolution": ["/properties/PIMPLE"],
        "dict:SimpleNeoN_fvSolution": ["/properties/SIMPLE"],
        "dict:KEpsilonCoeffs": [""],
        "dict:KOmegaSSTCoeffs": [""],
        "dict:SpalartAllmarasCoeffs": [""],
    },
    "incompressibleVoF": {
        "dict:Pimple_fvSolution": ["/properties/PIMPLE"],
        "dict:VofPimpleAlgorithmConfig": [""],
        "dict:DynamicMeshControls": [""],
    },
}


@pytest.mark.parametrize("solver_name", list_solver_names())
def test_objects_of_scalars_only_render_as_a_grid(solver_name):
    # `nfGrid` lays an object's controls side by side. It goes on every object of two or
    # more properties that are all a number, a checkbox or an enum (the algorithm
    # controls, a model's coefficients) — a text field, a nested dictionary or a single
    # property keeps the stock one-per-line layout (controlDict, transportProperties).
    solver = resolve_solver(solver_name)
    entries = [*build_forms(solver), *build_mesh_forms(solver), *build_field_forms(solver)]
    grids = {e.key: sorted(_keyword_sites(e.schema, "nfGrid")) for e in entries}
    assert {key: sites for key, sites in grids.items() if sites} == _SCALAR_GRIDS[solver_name]


def _solver_blocks():
    """Every ``(id, block schema)`` under a ``solvers`` section, across all solvers."""
    for name in list_solver_names():
        for entry in build_forms(resolve_solver(name)):
            section = entry.schema.get("properties", {}).get("solvers")
            if entry.cls_name.endswith("fvSolution") and section:
                for field, block in section["properties"].items():
                    yield pytest.param(section, block, id=f"{name}-{entry.cls_name}-{field}")


@pytest.mark.parametrize(("section", "block"), list(_solver_blocks()))
def test_solver_blocks_pin_the_standard_controls(section, block):
    # The config leaves a linear-solver block schemaless; the form pins the keys every
    # block has so they render as a grid of dropdowns / number fields (`nfSolvers` on the
    # section, `nfSolver` on the block), titled verbatim like every OpenFOAM key.
    assert section["nfSolvers"] is True
    props = block["properties"]
    assert list(props) == ["solver", "preconditioner", "smoother", "tolerance", "relTol"]
    assert [props[key]["title"] for key in props] == list(props)
    assert props["tolerance"]["type"] == props["relTol"]["type"] == "number"
    assert {"PCG", "PBiCGStab", "smoothSolver", "GAMG"} <= set(props["solver"]["examples"])
    assert {"DIC", "DILU"} <= set(props["preconditioner"]["examples"])
    assert {"symGaussSeidel", "GaussSeidel"} <= set(props["smoother"]["examples"])
    # Extra options (minIter, nSweeps, GAMG's agglomerator, …) are still added by hand.
    assert block["additionalProperties"] is True


def test_solver_block_names_the_companion_key_of_every_suggested_solver(solver):
    # A Krylov solver takes a preconditioner, a smoothing one a smoother: the block's
    # `nfSolver` map tells the renderer which of the two sits next to `solver`.
    entries = {e.key: e for e in build_forms(solver)}
    block = entries["dict:Pimple_fvSolution"].schema["properties"]["solvers"]["properties"]["p"]
    assert block["nfSolver"]["PCG"] == "preconditioner"
    assert block["nfSolver"]["GAMG"] == block["nfSolver"]["smoothSolver"] == "smoother"
    assert set(block["nfSolver"]) == set(block["properties"]["solver"]["examples"])


@pytest.mark.parametrize(
    "data",
    [
        # NeoN/Ginkgo names are not OpenFOAM's; a loaded case must not turn red.
        {"solver": "Ginkgo", "type": "solver::Bicgstab", "tolerance": 1e-6, "relTol": 0},
        # OpenFOAM's GAMG-preconditioned PCG nests a dictionary under `preconditioner`.
        {"solver": "PCG", "preconditioner": {"preconditioner": "GAMG", "nVcycles": 2}},
    ],
)
def test_solver_choices_are_suggestions_not_a_closed_enum(data, solver):
    jsonschema = pytest.importorskip("jsonschema")
    entries = {e.key: e for e in build_forms(solver)}
    block = entries["dict:Pimple_fvSolution"].schema["properties"]["solvers"]["properties"]["p"]
    jsonschema.Draft202012Validator(block).validate(data)


@pytest.mark.parametrize(
    ("data", "missing"),
    [
        ({"solver": "GAMG", "tolerance": 1e-6, "relTol": 0.05}, "smoother"),
        ({"solver": "smoothSolver", "preconditioner": "DIC"}, "smoother"),
        ({"solver": "PBiCGStab", "smoother": "DILU"}, "preconditioner"),
    ],
)
def test_solver_block_requires_the_companion_key_of_its_solver(data, missing, solver):
    # OpenFOAM aborts on a Krylov solver without `preconditioner` or a smoothing one
    # without `smoother`, so the card shows the empty companion as a required field.
    jsonschema = pytest.importorskip("jsonschema")
    entries = {e.key: e for e in build_forms(solver)}
    block = entries["dict:Pimple_fvSolution"].schema["properties"]["solvers"]["properties"]["p"]
    errors = [error.message for error in jsonschema.Draft202012Validator(block).iter_errors(data)]
    assert errors == [f"'{missing}' is a required property"]


def test_nested_model_is_titled_by_its_key_not_its_class_name():
    # A `water: PhaseTransport` property inherits the *class* name as its title via
    # the $ref, so both phase cards read "PhaseTransport"; the key is what tells
    # them apart.
    vof = {e.key: e for e in build_forms(resolve_solver("incompressibleVoF"))}
    transport = vof["dict:TransportPropertiesConfig"].schema["properties"]
    assert [transport[k]["title"] for k in ("water", "air")] == ["Water", "Air"]


@pytest.mark.parametrize("solver_name", ["incompressibleFluid", "incompressibleVoF"])
def test_foamfile_header_stays_in_the_data_but_is_not_rendered(solver_name):
    # version / format / class / object are boilerplate the writer needs and the
    # user never edits.
    entries = {e.key: e for e in build_forms(resolve_solver(solver_name))}
    gravity = entries["dict:GravityConfig"]
    assert "FoamFile" not in gravity.schema["properties"]
    assert gravity.defaults["FoamFile"]["object"] == "g"


def test_prose_property_keys_are_still_humanized(solver):
    # Hand-written config models keep their human labels — only OpenFOAM blocks
    # are shown verbatim.
    entries = {e.key: e for e in build_forms(solver)}
    control = entries["dict:ControlDictConfig"].schema["properties"]
    assert control["writeControl"]["title"] == "Write Control"
    assert control["deltaT"]["title"] == "Delta T"
    pimple = entries["dict:PimpleAlgorithmConfig"].schema["properties"]
    assert pimple["momentumPredictor"]["title"] == "Momentum Predictor"


def test_inline_refs_makes_every_schema_self_contained():
    # JSONForms' combinator renderers ajv.compile() every oneOf/anyOf arm
    # STANDALONE; an arm containing a $ref throws, is never cached and
    # recompiles on every reactive re-evaluation — the big scheme schemas
    # (nested discriminated unions) hard-froze the page that way.
    import json  # noqa: PLC0415

    for name in list_solver_names():
        for e in build_forms(resolve_solver(name)):
            text = json.dumps(e.schema)
            assert "$ref" not in text, f"{name}/{e.key} still contains a $ref"
            assert "$defs" not in e.schema, f"{name}/{e.key} still carries $defs"
            assert "discriminator" not in text, f"{name}/{e.key} keeps a dangling discriminator"


def test_inline_refs_bounds_a_self_referential_union(solver):
    # cellLimited nests a GradScheme, which includes cellLimited again. The cycle
    # cannot be inlined, so the inner scheme offers the non-recursive arms only.
    entries = {e.key: e for e in build_forms(solver)}
    grad = entries["dict:Pimple_fvSchemes"].schema["properties"]["gradSchemes"]["properties"]
    arms = {a["title"]: a for a in grad["grad(U)"]["oneOf"]}
    assert set(arms) == {"Gauss", "pointCellsLeastSquares", "cellLimited"}
    inner = arms["cellLimited"]["properties"]["inner_scheme"]["oneOf"]
    assert [a["title"] for a in inner] == ["Gauss", "pointCellsLeastSquares"]


def _discriminators(node):
    """Every ``type`` property schema that pins a ``const``, however deeply nested."""
    if isinstance(node, dict):
        prop = node.get("properties", {}).get("type")
        if isinstance(prop, dict) and "const" in prop:
            yield prop
        for value in node.values():
            yield from _discriminators(value)
    elif isinstance(node, list):
        for value in node:
            yield from _discriminators(value)


def test_union_arm_discriminator_is_kept_in_the_data_but_not_rendered(solver):
    # The arm selector already names the type; a second "Type" dropdown holding that one
    # value under every (nested) scheme made the Numerics step 7000-12000 px tall.
    # JSONForms generates a control only for a property it can derive a JSON type for
    # (`type`/`enum`/`properties`/`items`), and fills a newly selected arm's data from
    # `default` — so exactly `const` + `default` hides the control and keeps the value.
    for name in list_solver_names():
        for entry in build_forms(resolve_solver(name)):
            for prop in _discriminators(entry.schema):
                assert prop == {"const": prop["const"], "default": prop["const"]}, (
                    f"{name}/{entry.key}: {prop}"
                )
    div = {e.key: e for e in build_forms(solver)}["dict:Pimple_fvSchemes"].schema
    arms = div["properties"]["divSchemes"]["properties"]["div(phi,U)"]["oneOf"]
    assert [(a["title"], list(a["properties"])) for a in arms] == [
        ("none", ["type"]),
        ("Gauss", ["type", "interpolation"]),
        ("bounded", ["type", "interpolation"]),
    ]


def test_inline_refs_merges_siblings_and_keeps_cycles():
    from neofoam.ui.form_schema import inline_refs  # noqa: PLC0415

    schema = {
        "$defs": {
            "Leaf": {"type": "object", "title": "leaf-title"},
            "Loop": {"properties": {"next": {"$ref": "#/$defs/Loop"}}},
        },
        "properties": {
            "a": {"$ref": "#/$defs/Leaf", "title": "arm-title"},
            "b": {"$ref": "#/$defs/Loop"},
        },
    }
    out = inline_refs(schema)
    # Sibling keys override the resolved definition.
    assert out["properties"]["a"] == {"type": "object", "title": "arm-title"}
    # The cyclic ref survives, so its definitions are kept.
    assert out["properties"]["b"]["properties"]["next"]["$ref"] == "#/$defs/Loop"
    assert "Loop" in out["$defs"]


def test_field_in_dimensions_and_internalfield_collapse_to_text(solver):
    # The fixed 7-element dimension vector and the FieldValue union both render as a
    # single string field, not a growable integer list / ANYOF combinator tabs.
    entries = {e.key: e for e in build_forms(solver)}
    props = entries["field_in:UFieldConfig"].schema["properties"]

    dims = props["dimensions"]
    assert dims["type"] == "string"
    assert dims["default"] == "[0 1 -1 0 0 0 0]"

    internal = props["internalField"]
    assert internal["type"] == "string"
    assert "anyOf" not in internal and "oneOf" not in internal
    assert internal["default"] == "uniform (0 0 0)"


def test_bc_value_arm_collapses_to_text_but_type_union_stays(solver):
    from neofoam.io.pydantic_schema import slice_schema  # noqa: PLC0415
    from neofoam.ui.form_schema import jsonforms_schema  # noqa: PLC0415

    bc = jsonforms_schema(
        slice_schema(
            tools.config_schema(solver, "u_field_config").json_schema,
            ["boundaryField"],
        )
    )
    # The discriminated BC-type union is preserved (a titled oneOf) …
    fixed = next(
        a
        for a in bc["properties"]["boundaryField"]["additionalProperties"]["oneOf"]
        if a.get("title") == "fixedValue"
    )
    # … every arm is self-contained (inline_refs: no $ref/$defs — JSONForms'
    # combinator renderer must be able to hand each arm to AJV standalone) …
    assert "$ref" not in fixed and "$defs" not in bc
    # … and the fixedValue `value` (scalar|vector|uniform-string) collapses to a string.
    assert fixed["properties"]["value"]["type"] == "string"


@pytest.mark.parametrize("solver_name", list_solver_names())
def test_boundary_field_maps_render_as_patch_rows(solver_name):
    # `nfPatches` hands a `boundaryField` map to the bundled row renderer, which binds
    # each patch through the map's data: a patch named `wall.left` never enters a dotted
    # JSONForms path. Both the wizard's BC half and the Parameters whole-field form.
    solver = resolve_solver(solver_name)
    entries = [e for e in build_forms(solver) if e.kind == "field_bc"]
    for entry in [*entries, *build_field_forms(solver)]:
        assert _keyword_sites(entry.schema, "nfPatches") == {"/properties/boundaryField": True}


def test_turbulence_properties_uischema_gates_ras_and_les_on_simulation_type(solver):
    entries = {e.config_name: e for e in build_forms(solver)}
    uischema = entries["turbulence_properties_config"].uischema
    assert uischema is not None
    rules = {
        e["scope"].rsplit("/", 1)[-1]: e.get("rule")
        for e in uischema["elements"]
        if isinstance(e, dict)
    }
    # simulationType picks; RAS/LES render only when it names them.
    assert rules["simulationType"] is None
    for block in ("RAS", "LES"):
        assert rules[block] == {
            "effect": "SHOW",
            "condition": {
                "scope": "#/properties/simulationType",
                "schema": {"const": block},
            },
        }
    # No other form needs a hand-written layout (JSONForms auto-generates it).
    assert [e.config_name for e in build_forms(solver) if e.uischema is not None] == [
        "turbulence_properties_config"
    ]


@pytest.mark.parametrize(
    ("key", "label"),
    [
        ("deltaT", "Delta T"),
        ("writeControl", "Write Control"),
        # "NeoN" is a name, not two camelCase words.
        ("PressureVelocityAlgorithmNeoN", "Pressure Velocity Algorithm NeoN"),
        ("NeoNControl", "NeoN Control"),
    ],
)
def test_humanize_splits_words_but_keeps_neon_whole(key, label):
    from neofoam.ui.form_schema import humanize  # noqa: PLC0415

    assert humanize(key) == label


def test_transport_properties_form_requires_nu_for_a_newtonian_fluid(solver):
    # JSONForms validates the form with AJV, so the rule has to survive the transform
    # for the missing `nu` to be flagged in the form and not only on save.
    entry = next(e for e in build_forms(solver) if e.config_name == "transport_properties_config")
    assert entry.schema["if"] == {"properties": {"transportModel": {"const": "Newtonian"}}}
    assert entry.schema["then"] == {"required": ["nu"]}
