# `jsonforms_module` — the wizard's `<json-forms>` client bundle

The case wizard's forms are drawn in the browser by JSONForms + Vuetify. This directory
holds the JS sources, the checked-in Vite build (`static/`) and the trame module
(`module.py`) that serves it. `bun` is needed to rebuild it, never to run the wizard.

## Rebuild and commit

```bash
cd src/neofoam/ui/jsonforms_module
bun install --frozen-lockfile   # only when node_modules/ is missing
bunx vite build                 # writes static/neofoam_jsonforms.{umd.js,css}
```

- Commit `static/` in the **same commit** as the source change, exactly as built:
  `static/` is excluded from the end-of-file hook, and
  `test_committed_bundle_equals_a_rebuild` compares it byte for byte with a fresh build
  (it is skipped without `bun` or `node_modules/`, so run it locally).
- Merge conflict in `static/`: take either side, then rebuild.

## File map

- `entry.mjs` — registers `<json-forms>` (`NeoFoamJsonForms`: injects the renderers and
  translations, stars the keys of a holding top-level `if`/`then`) and the renderers:
  - `NfNumberControl` — every number as a text field that parses `1e-5` (the stock
    control masks input to one decimal);
  - `NfCompactSection` — a keyed section as `rows` (fvSchemes, patches), `cards`
    (`solvers`, MRFProperties, fvOptions) or `cells` (a solver block, a sub-dictionary),
    each with its own "add" row;
  - `NfInlineOneOf` — a discriminated union as one select, its arm's fields inline;
  - `NfGridLayout` — an all-scalar object as a wrapping grid of the stock controls.
- `sectionSchema.mjs` — the pure rules those renderers decide by (layout per keyword,
  entry schemas, pinned keys, stale companion keys, number parsing). It imports no UI
  library; `sectionSchema.test.mjs` tests it with `bun test`.
- `compact.css` — the styles of the renderers above, bundled into the CSS asset.
- `vite.config.mjs` — one UMD library build. Vue and Vuetify are **externals** mapped to
  trame's globals: a second copy would break Vue's provide/inject. The Vuetify globals
  are lazy proxies, because trame loads module scripts async, possibly before Vuetify.
- `module.py` — what `server.enable_module` reads: serve `static/`, load the script and
  style, `vue_use` the plugin.

## The contract with Python

The schema keywords, their value shapes and the implicit rules (hidden discriminator,
`examples`, i18n ids, path-unsafe keys) are in `doc/reference/ui-architecture.rst`.
To add a renderer:

1. Emit a keyword from a pass in `../form_schema.py` and add it to `RENDERER_KEYWORDS`.
2. Add a tester to `renderers` in `entry.mjs` that ranks above the stock renderers (the
   custom ones use 30–40), or a `LAYOUT_BY_KEYWORD` entry for a new section layout.
3. Put what it decides from schema and data alone in `sectionSchema.mjs`, with a test.
4. Rebuild. `test_renderer_keywords_match_the_js_sources_and_the_bundle` fails while
   either side, or the bundle, lacks the keyword.

Two rules every renderer keeps: a schema or UISchema handed to a nested `JsonForms` must
be **identity-stable** (a fresh object restarts that form on every edit — see the caches
in `NfCompactSection` and `rowsSchema`), and a key holding `.`, `[` or `]` cannot be part
of a data path, so such an entry is a nested form written back through its parent's data
(`PATH_UNSAFE`).

## Look at it

```bash
neofoam ui --no-browser                                  # http://localhost:8080/
pytest test/ui/test_jsonforms_module.py                  # contract, rebuild, bun test
pytest test/ui/test_browser_forms.py -m browser          # the forms in headless Chromium
```

The browser tests need `playwright` (`dev` extra) and `playwright install chromium`.
