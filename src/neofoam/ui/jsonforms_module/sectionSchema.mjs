// The data -> data rules of the section and number renderers (entry.mjs), free of Vue,
// Vuetify and JSONForms so `bun test` runs them without a browser.

const JSON_TYPES = ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string']
const OPEN_DICT = { type: 'object', additionalProperties: true }
export const isDict = (value) =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

// JSONForms and lodash read these as separators, so such a key cannot be part of a data path.
export const PATH_UNSAFE = /[.[\]]/

// How a keyed section is drawn, by the schema keyword that flags it.
export const LAYOUTS = ['rows', 'cards', 'cells']
export const LAYOUT_BY_KEYWORD = {
  nfCompact: 'rows',
  nfPatches: 'rows',
  nfSolvers: 'cards',
  nfSolver: 'cells',
  nfDicts: 'cards',
  nfDict: 'cells',
}
export const layoutOf = (schema) =>
  LAYOUT_BY_KEYWORD[Object.keys(LAYOUT_BY_KEYWORD).find((flag) => schema[flag])]

// The shape a new entry takes: that of its declared siblings, else the map's own.
export const entryShape = (schema) => {
  const open = schema.additionalProperties
  return Object.values(schema.properties ?? {})[0] ?? (isDict(open) ? open : undefined)
}

// An entry only present in the data (a loaded case, or one added in the form) is drawn
// like its declared siblings when it has their object shape, else as a raw value.
const addedEntrySchema = (key, value, shape) => {
  // A scheme or BC union has no `type` of its own, yet every arm is a dictionary.
  const dicts = shape?.type === 'object' || Array.isArray(shape?.oneOf)
  const dictShape = dicts ? shape : OPEN_DICT
  return { ...(isDict(value) ? dictShape : { type: JSON_TYPES }), title: key }
}

// A declared string-or-dictionary (`preconditioner`) is drawn as what the data holds,
// and as a name while it is empty.
const eitherEntrySchema = (declared, value) => {
  const types = declared.type.filter((type) => type !== 'object')
  const type = types.length > 1 ? types : types[0]
  return { ...declared, ...(isDict(value) ? OPEN_DICT : { type }) }
}

// A solver block's companion key reads as required: the solver does not run without.
const requiredKeys = (schema, data) => {
  const companion = schema.nfSolver?.[data?.solver]
  return [...(schema.required ?? []), ...(companion ? [companion] : [])]
}

// The section's schema with one property per entry, declared or only in the data.
// `cache` must outlive the call: an added entry gets the SAME schema object every time,
// because a fresh schema object would restart that entry's own form on every edit.
export const rowsSchema = (schema, data, cache) => {
  const properties = { ...(schema.properties ?? {}) }
  const entries = data ?? {}
  const shape = entryShape(schema)
  for (const key of Object.keys({ ...properties, ...entries })) {
    const value = entries[key]
    if (!(key in properties))
      properties[key] = cache[key + ':' + isDict(value)] ??= addedEntrySchema(key, value, shape)
    else if (Array.isArray(properties[key].type))
      properties[key] = eitherEntrySchema(properties[key], value)
  }
  return { ...schema, properties, required: requiredKeys(schema, entries) }
}

// The keys that cannot be deleted: every declared one, or for a solver block `solver`,
// the key that solver takes, `tolerance` and `relTol`.
export const pinnedKeys = (schema, data) => {
  if (!schema.nfSolver) return Object.keys(schema.properties ?? {})
  return ['solver', schema.nfSolver[data?.solver], 'tolerance', 'relTol'].filter(Boolean)
}

// Cells show the pinned keys first, then what else the data holds; rows and cards keep
// the schema's order.
export const entryKeys = (layout, sectionSchema, pinned, data) => {
  if (layout !== 'cells') return Object.keys(sectionSchema.properties)
  return [...pinned, ...Object.keys(data ?? {}).filter((key) => !pinned.includes(key))]
}

const besidesSolver = (block) => JSON.stringify({ ...block, solver: 0 })

// Picking a solver of the other family drops the companion key it does not take; the
// one it takes then shows empty. Only an edit of `solver` alone counts (not a loaded
// case), and a solver the map does not know (`Ginkgo`) drops nothing. Returns the block
// to write back, or null when nothing is to drop.
export const withoutStaleCompanions = (now, before, companions) => {
  const keep = companions?.[now?.solver]
  if (!keep || !isDict(before) || now.solver === before.solver) return null
  if (besidesSolver(now) !== besidesSolver(before)) return null
  const kept = { ...now }
  for (const key of Object.values(companions)) if (key !== keep) delete kept[key]
  return Object.keys(kept).length === Object.keys(now).length ? null : kept
}

// A tolerance reads `1e-6` as in a case file, not `0.000001`; the data stays a number.
export const formatNumber = (value) => {
  const size = Math.abs(value)
  if (typeof value !== 'number' || size === 0 || (size >= 1e-4 && size < 1e6)) return String(value)
  return value.toExponential().replace('e+', 'e')
}

// What a number looks like while it is still being typed: `-`, `.`, `1e`, `1e-`.
export const PARTIAL_NUMBER = /^[-+]?\d*\.?\d*(e[-+]?)?$/i

// What typed text stores: `{ value }` to write (undefined clears the key), `{ error }`
// for text that can never become a number (`abc`, `0,5`), and null for partial input
// (`1e-`, `-`), which keeps the last valid value until it parses. An untyped option
// (`nSweeps 2`, `cacheAgglomeration on`) accepts text and stores it as typed.
export const parseNumberInput = (text, acceptsText) => {
  const trimmed = (text ?? '').trim()
  if (trimmed !== '' && Number.isFinite(Number(trimmed))) return { value: Number(trimmed) }
  if (acceptsText) return { value: trimmed }
  if (trimmed === '') return { value: undefined }
  if (PARTIAL_NUMBER.test(trimmed)) return null
  return { error: `'${trimmed}' is not a number (write 0.5 or 1e-5)` }
}
