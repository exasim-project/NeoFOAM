Case-wizard UI architecture
===========================

The case wizard (``src/neofoam/ui/``) is a `trame <https://kitware.github.io/trame/>`_
web app: pick a solver's models, fill one JSON-schema form per config (or let the AI
chat fill them), then save a runnable, validated case. No form is written by hand —
each is generated from a pydantic config and drawn in the browser by
`JSONForms <https://jsonforms.io>`_. For a user's walk-through see
:doc:`/tutorials/case-wizard`.

Start the wizard
----------------

The UI needs the ``ui`` extra (``pip install -e ".[ui]"``; part of ``all``):

.. code-block:: bash

    neofoam ui                                # http://localhost:8080/, opens a browser
    neofoam ui --solver incompressibleVoF --port 8081 --no-browser
    neofoam ui --host 0.0.0.0 --no-browser    # reachable from a phone on the LAN
    python -m neofoam.ui                      # same app, trame's own flags (--host, --port, --server)

``--solver`` takes ``incompressibleFluid`` (default), ``incompressibleFluidNeoN`` or
``incompressibleVoF``. ``NEOFOAM_WIZARD_STL_DIR`` pre-fills the Geometry step's STL
folder; ``NEOFOAM_CASE_MODEL`` overrides the chat's model, which needs
``ANTHROPIC_API_KEY``. The target directory must be an absolute path
(``_paths.py``), so nothing is ever written into the launch directory.

Module map
----------

Listed bottom-up. **A module imports only from rows above its own**; two references
point the other way (``boundary_forms`` → ``FormEntry``, ``sweep_view`` →
``SweepPanel``) and are ``TYPE_CHECKING``-only.

``_paths``, ``_responsive``, ``_markdown``, ``review``, ``scaffold``, ``geometry``, ``sweep_model``, ``plugins/``, ``jsonforms_module/``
    Leaves: path-field check; phone/desktop drawer props; chat markdown → escaped
    HTML; findings → alert rows;
    ``Allrun``/``Allclean``; STL scan and mesh dicts; the sweep data model; the
    step-plugin interface; the bundled ``<json-forms>`` client module.
``boundary_forms``
    Seeds a field's boundary conditions from scanned patches; ``patch_bc_schema``.
``form_schema``
    The JSON-Schema → JSONForms transform and the renderer keywords (below).
``geometry_agent``
    The LLM helper that assigns patch roles from prose.
``forms``
    The form registry: ``FormEntry``, ``build_forms`` (+ ``build_mesh_forms`` and
    ``build_field_forms`` for the sweep), ``schema_key``, ``uischema_key``.
``steps``, ``case_spec``, ``geometry_panel``
    Steps, model choices/families and ``selection_key``/``choice_key``; form state
    ↔ ``save_case`` spec; the Geometry step.
``case_load``, ``sweep_view``
    Reopen a case from disk (``case_algorithm``, ``case_advection_model``,
    ``models_to_select``, ``apply_configs_to_forms``); the Parameters step's layout.
``agent_panel``, ``sweep_panel``
    ``AgentPanel`` (the AI chat drawer) and ``SweepPanel`` (the Parameters step).
``app``
    The orchestrator: ``build_app``, ``_Wizard``, ``_seed_state``,
    ``_register_case_controllers`` and the layout functions.

``import neofoam.ui`` must work without the optional dependencies, so these imports
are function-local: trame in ``app`` (``build_app``, ``_json_forms_class``) and
``sweep_view``; ``pydantic_ai`` in ``geometry_agent``; and ``neofoam.ui.app`` itself
in ``neofoam/ui/__init__.py``. The panels never import trame — they are handed its
widget modules in ``render``.

How a form is produced
----------------------

#. ``neofoam.io.schema.config_schema`` returns a config class's
   ``model_json_schema()`` and its default values.
#. ``forms.build_forms`` runs the schema through ``form_schema.jsonforms_schema``
   (a ``0/<field>`` config is sliced into an *initial value* and a *boundary
   conditions* form first).
#. ``inline_refs`` resolves every ``$ref``: JSONForms compiles each union arm on its
   own, and an arm holding a ``$ref`` throws on every re-evaluation.
#. Each node is rewritten — unions first (``Optional[X]`` → ``X``, value unions → one
   text field, discriminated unions → titled ``oneOf``), then its children, then
   ``_NODE_PASSES`` in order: ``_drop_useless_adder``, ``_tag_adders``,
   ``_pin_solver_controls``, ``_patch_adder``, ``_tag_scalar_grid``,
   ``_hide_discriminator``. Two ordering constraints; the other passes commute:

   * ``_drop_useless_adder`` before ``_tag_adders``: a label goes on a row that is
     still there (``_tag_adders`` reads ``additionalProperties``); a fixed card such
     as ``PIMPLE`` gets none.
   * ``_tag_adders`` before ``_pin_solver_controls``: a solver block is recognised as
     an open dict (``nf.option``) by having no ``properties``, which pinning then
     gives it.

#. ``_dictionary_cards`` runs last, on the whole schema (``MRFProperties``,
   ``fvOptions``).
#. The result becomes a ``FormEntry``; ``_seed_state`` copies its schema, UISchema and
   defaults into trame state, and ``_form_panel`` binds them to ``<json-forms>``.

JSONForms gets a config's schema and its data at once, so the two must agree. A scheme
variant (``src/neofoam/foam/schemes/_variant.py``) therefore dumps structured data by
default and its OpenFOAM token (``Gauss upwind``) only under ``OPENFOAM_CONTEXT``,
which the file writers pass; an unconditional token matched no ``oneOf`` arm and the
form showed the first arm instead. A scheme entry added in the form under an
undeclared key bypasses the typed fields, so ``_tokenize_extras_validator``
(``src/neofoam/foam/fv_configs.py``) turns it into its token when the config is
validated on save.

The Python ↔ JS renderer contract
---------------------------------

The bundled renderers match schema keywords **by name**; a keyword renamed on one side
falls back to the stock renderer without any error. Python emits them in
``form_schema.py`` (``RENDERER_KEYWORDS``); the JS side reads them in
``sectionSchema.mjs`` (``LAYOUT_BY_KEYWORD``) and ``entry.mjs``.

.. list-table::
   :header-rows: 1
   :widths: 14 20 30 36

   * - Keyword
     - Value
     - Emitted by
     - Renderer / layout
   * - ``nfCompact``
     - ``true``
     - ``_tag_adders``: an OpenFOAM section of scheme unions only (``fvSchemes``)
     - ``NfCompactSection``, ``rows`` (key left, ``NfInlineOneOf`` selects inline)
   * - ``nfPatches``
     - ``true``
     - ``_patch_adder``: a ``boundaryField`` map
     - ``NfCompactSection``, ``rows``
   * - ``nfSolvers``
     - ``true``
     - ``_pin_solver_controls``: the ``solvers`` section
     - ``NfCompactSection``, ``cards`` (one nested form per block)
   * - ``nfSolver``
     - ``{solver: companion key}``, i.e. ``SOLVER_COMPANION``
     - ``_pin_solver_controls``: each solver block
     - ``NfCompactSection``, ``cells``
   * - ``nfDicts``
     - ``true``
     - ``_dictionary_cards``: the whole schema
     - ``NfCompactSection``, ``cards``
   * - ``nfDict``
     - ``true``
     - ``_dictionary_cards``: its ``additionalProperties``
     - ``NfCompactSection``, ``cells``
   * - ``nfGrid``
     - ``true``
     - ``_tag_scalar_grid``: an object of two or more scalars only
     - ``NfGridLayout`` (wrapping grid of the stock controls)

``nfInline`` and ``nfRowKey`` are UISchema options the JS sets for itself. The contract
has implicit parts as well:

* **Discriminator** — a union arm's ``type`` is ``{"const": t, "default": t}``: no
  control is drawn, ``default`` puts it into the data when the arm is picked, and
  ``NfInlineOneOf`` selects the arm by it.
* **Pinned solver keys** — ``solver``, the companion key its ``nfSolver`` map names,
  ``tolerance`` and ``relTol`` cannot be deleted and come first; switching the solver
  family drops the stale companion key.
* **Suggestions** — JSON-schema ``examples`` become a combobox's suggestions: free
  text stays valid (a loaded case may name ``Ginkgo``), unlike with an ``enum``.
* **New entries** — a hand-added map entry starts from the ``default`` of the map's
  ``additionalProperties`` (a patch: the wall boundary condition).
* **i18n ids** — a node's ``i18n`` prefix (``nf.patch``, ``nf.entry``, ``nf.solver``,
  ``nf.option``, ``nf.zone``, ``nf.source``, ``nf.keyword``) picks
  ``<prefix>.propertyNameLabel`` and ``<prefix>.addLabel`` for the adder's name field
  and button; the texts are ``ADDER_TRANSLATIONS``, sent as ``form_translations``.
* **Conditional asterisk** — while a top-level ``if`` holds for the data, the keys its
  ``then`` requires are starred (``nu`` of a Newtonian fluid); validation is unchanged.
* **Path-unsafe keys** — JSONForms data paths are dot-separated, so an entry whose key
  holds ``.``, ``[`` or ``]`` (``wall.left``, ``alpha.water``) is a nested form rooted
  at the entry and written back through the parent's data.

``test/ui/test_jsonforms_module.py`` guards this: the keywords in the ``.mjs`` sources
equal ``RENDERER_KEYWORDS`` and are in the bundle, the i18n ids the JS asks for equal
the ones Python sends, the committed bundle equals a rebuild byte for byte, and the
``bun`` unit tests pass. Rebuilding is described in
``src/neofoam/ui/jsonforms_module/README.md``.

Shared trame state
------------------

Everything the modules share at run time is a trame state variable. Build the
computed names with their helper, which makes them valid JS identifiers.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Key
     - Owner (seeds it)
     - Other readers and writers
   * - ``current_step``
     - ``app``
     - ``agent_panel`` reads it to pick the step's chat handler
   * - ``target_dir``
     - ``app`` (toolbar)
     - ``geometry_panel`` fills it after a scan when empty; ``sweep_panel`` sets it on
       restoring a base case; ``agent_panel`` auto-saves into it and loads the case
       in it (``ctrl.load_target_case``)
   * - ``sel_<model>`` (``selection_key``), ``choice_<family>`` (``choice_key``)
     - ``app``
     - written through ``steps.select_model_state`` only (radio group,
       ``case_load``); ``sweep_panel`` and ``sweep_view`` read ``sel_*``
   * - ``form_*`` (``FormEntry.state_key``)
     - ``app``; the browser's ``change`` event
     - ``case_load.apply_configs_to_forms`` (chat, "Load case", sweep), ``geometry_panel``
       (patch seeding), ``steps.turbulence_form_state``; ``sweep_panel`` reads them
   * - ``schema_*`` / ``uischema_*`` (``schema_key`` / ``uischema_key``)
     - ``app``
     - ``geometry_panel`` replaces a ``field_bc`` schema with ``patch_bc_schema``
   * - ``form_translations``
     - ``app``
     - read-only; ``sweep_view`` binds it to its own forms
   * - ``geometry_patches``; ``stl_dir``, ``role_names``, ``geo_*``,
       ``geometry_status`` / ``_severity`` / ``_busy``, ``mesh_written``
     - ``geometry_panel``
     - ``geometry_patches`` only: ``agent_panel`` assigns roles, ``app`` shows the
       "no patches" hint — unless a ``field_bc`` form already holds patches, as
       after "Load case" (a template condition on the ``form_*`` data, no key of
       its own)
   * - ``validation_ok``, ``findings``, ``save_report``, ``scaffolded``
     - ``app``
     - ``sweep_panel`` reads ``scaffolded`` before an export
   * - ``main_drawer`` / ``main_drawer_mobile``, ``ai_panel`` / ``ai_panel_mobile``
     - trame's layout (``main_drawer``), ``app``
     - ``agent_panel`` binds the ``ai_panel`` pair and opens it for a "Load case" report
   * - ``chat_log``, ``chat_html``, ``chat_input``, ``ai_busy``, ``suggested_prompts``
     - ``agent_panel``
     - ``app`` reads ``ai_busy`` to disable the toolbar's "Load case"
   * - ``sweep_*``
     - ``sweep_panel``
     - ``sweep_view`` only (which also reads ``target_dir``)

Load, save, validate, review
----------------------------

The toolbar's "Load case" calls ``ctrl.load_target_case``
(``AgentPanel.load_target_case``): the read of the assistant's ``load_case`` tool
(``case_load.read_case_configs``) on ``target_dir``, applied at once with
``case_load.apply_configs_to_forms``. No agent is built, so it needs no API key. The
report — the loaded configs, the selected models, a file that is present but does not
validate, fields that exist as ``0.orig/`` only (they are read from ``0/``), or why
nothing was loaded — goes to ``chat_log`` and the AI drawer is opened to show it. It is dropped while ``ai_busy``: a chat turn applies its own load.

Several configs slice one file, so ``load_case_from_disk`` takes a slice for absent —
and stays silent — when the file spells out none of the required keys that only this
slice declares (``div(phi,T)`` of the Boussinesq ``fvSchemes``). A slice with some of
them is reported as ``missing <key>, <key>``. A regex or grouped key (``"(U|nuTilda)"``)
is not such a spelling, so pitzDaily reports ``missing solvers.U, solvers.UFinal``.

An assistant message is drawn as markdown: ``_markdown._chat_html`` escapes the text
first and then turns the subset the wizard emits (bold, italics, inline code, ``-``
lists, line breaks) into tags, kept per message in ``chat_html`` and bound with
``v-html``. A reply or a path is untrusted, so no HTML of its own survives; a user's
message stays plain text.

A pick-one family follows the case the way the solver would run it:
``case_algorithm`` reads the ``system/fvSolution`` control block, and for
``incompressibleVoF`` ``case_advection_model`` mirrors the solver's
``select_from_case`` — the ``advectionScheme`` key, else isoAdvector controls in the
alpha solver block, else ``MULES``, so a case without evidence selects ``MULES``.


``ctrl.save_case`` (``app._save_case``) switches to the Review step, then:
``case_spec.state_to_case_spec`` merges the two halves of each field config and skips
forms of unselected models; ``neofoam.mcp.tools.save_case`` validates the spec against
the solver's aggregate model and writes the files; ``scaffold.scaffold_runnable_case``
adds ``Allrun`` and ``Allclean``; ``tools.validate_case`` runs the checks of
:mod:`neofoam.framework.validation` on the written case — e.g. ``solver-companion``: a
``PCG`` without a ``preconditioner``. ``review.findings_to_rows`` turns the findings
into ``findings``; an exception on the way becomes rows too (``save_error_rows``), so a
failed save never crashes the UI. ``ctrl.revalidate`` repeats the last step alone.

Test and inspect
----------------

.. code-block:: bash

    pytest test/ui                                    # headless; browser tests deselected
    pytest test/ui/test_browser_forms.py -m browser   # the forms in headless Chromium
    cd src/neofoam/ui/jsonforms_module && bun test    # the pure renderer rules

The browser suite needs ``playwright`` (``dev`` extra) and ``playwright install
chromium``. ``test/ui/conftest.py`` deselects it unless ``-m`` names ``browser``; it
serves the wizard from a subprocess and asserts on ``window.trame.state``. Below Vuetify's ``md``
breakpoint (960 px) both drawers become closed-by-default overlays with open flags of
their own (the ``*_mobile`` keys, ``_responsive.py``), and the target directory moves
to a second toolbar row.

Step plugins
------------

Another package can add a wizard step through the ``neofoam.ui.steps`` entry-point
group. The interface (``StepPlugin``, ``StepContext``, ``AT_START``) is documented in
the docstring of ``src/neofoam/ui/plugins/__init__.py``.
