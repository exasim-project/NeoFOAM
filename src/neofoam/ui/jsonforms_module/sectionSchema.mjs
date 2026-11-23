// The data -> data rules of the section and number renderers (entry.mjs), free of Vue,
// Vuetify and JSONForms so `bun test` runs them without a browser.

const JSON_TYPES = ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string']
const OPEN_DICT = { type: 'object', additionalProperties: true }
export const isDict = (value) =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

// Inside a compact row or grid the error text shows only when there is one.
const INLINE_FIELD = { hideDetails: 'auto' }
export const INLINE_OPTIONS = {
  nfInline: true,
  vuetify: {
    'v-text-field': INLINE_FIELD,
    'v-select': INLINE_FIELD,
    'v-combobox': INLINE_FIELD,
    'v-checkbox': { ...INLINE_FIELD, density: 'compact' },
  },
}

// A solver block is a form of its own, rooted at the block.
export const BLOCK_UISCHEMA = { type: 'Control', scope: '#' }

// JSONForms and lodash read these as separators, so such a key cannot be part of a data path.
export const PATH_UNSAFE = /[.[\]]/

// How a keyed section is drawn, by the schema keyword that flags it.
export const LAYOUTS = ['rows', 'cards', 'cells']
export const LAYOUT_BY_KEYWORD = {
  nfCompact: 'rows',
  nfPatches: 'rows',
  nfUnions: 'rows',
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
  if (schema.entryLike === 'self') return { ...schema, title: undefined }
  return Object.values(schema.properties ?? {})[0] ?? (isDict(open) ? open : undefined)
}

// The property a union names its arm by. Python sends `nfDiscriminator` only when it is
// not `type`.
export const discriminatorOf = (schema) => schema.nfDiscriminator ?? 'type'

// A union's discriminator as a label: `solver` -> `Solver`. Null for the default `type`,
// whose rows are labelled by their key alone.
export const discriminatorLabel = (schema) => {
  const name = schema.nfDiscriminator
  return name ? name[0].toUpperCase() + name.slice(1) : null
}

// What survives a switch from arm `from` to arm `to`: a value both declare that
// `fits(propertySchema, value)` for the new arm. The discriminator itself never does, nor
// a value the new arm's model refuses on its own (`nfSwitchDrops`), which would block Save.
export const keptValues = (from, to, data, fits) => {
  const kept = {}
  const dropped = to.nfSwitchDrops ?? []
  for (const [key, property] of Object.entries(to.properties ?? {})) {
    const shared = from?.properties?.[key] !== undefined && data?.[key] !== undefined
    if (!shared || dropped.includes(key)) continue
    if (property.const === undefined && fits(property, data[key])) kept[key] = data[key]
  }
  return kept
}

// A control's errors without the failed `if`/`then` wrapper of its "required" message.
export const withoutRuleErrors = (errors) =>
  (errors ?? '')
    .split('\n')
    .filter((message) => message !== 'must match "then" schema')
    .join('\n')

// An entry shown only under "All entries".
export const isAdvanced = (property) => property?.nfLevel === 'advanced'

// The fields an arm draws. In a union naming its discriminator, a key only this arm
// declares is its own tuning: drawn when required or set, so the row stays compact.
export const armFields = (arm, armSchemas, data, named) => {
  const keys = Object.keys(arm?.properties ?? {}).filter(
    (key) => arm.properties[key].const === undefined && !isAdvanced(arm.properties[key]),
  )
  if (!named) return keys
  const declaredElsewhere = (key) =>
    armSchemas.some((other) => other !== arm && other.properties?.[key] !== undefined)
  // `solverName`: the only place a block kept as written can name its solver.
  const discriminator = Object.keys(arm.properties).find((key) => arm.properties[key].const)
  return keys.filter(
    (key) =>
      key === discriminator + 'Name' ||
      declaredElsewhere(key) ||
      arm.required?.includes(key) ||
      data?.[key] !== undefined,
  )
}

// A dictionary no arm of a named union claims (a turbulence block next to typed ones).
const isFreeBlock = (shape, value) => {
  const name = shape?.nfDiscriminator
  return (
    name !== undefined &&
    isDict(value) &&
    !shape.oneOf.some((arm) => arm.properties?.[name]?.const === value[name])
  )
}
const FREE_BLOCK = { ...OPEN_DICT, nfDict: true, i18n: 'nf.option' }

// An entry only present in the data (a loaded case, or one added in the form) is drawn
// like its declared siblings when it has their object shape, else as a raw value.
const addedEntrySchema = (key, value, shape) => {
  if (isFreeBlock(shape, value)) return { ...FREE_BLOCK, title: key }
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
      properties[key] = cache[key + ':' + isDict(value) + ':' + isFreeBlock(shape, value)] ??=
        addedEntrySchema(key, value, shape)
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

// The arm a union's data selects, or undefined.
export const selectedArm = (schema, data) => {
  const key = discriminatorOf(schema)
  return schema.oneOf?.find((arm) => arm.properties?.[key]?.const === data?.[key])
}

// An arm as its select shows it.
export const armLabel = (arm) => arm?.nfArmLabel ?? arm?.title

const claimed = (section) =>
  Object.entries(section.properties ?? {}).filter(([, entry]) => entry.nfField !== undefined)

const schemeRows = (sections) => {
  const rows = new Map()
  for (const [section, schema] of sections) {
    if (schema.nfFieldGroups) continue
    for (const [key, entry] of claimed(schema)) {
      if (!rows.has(entry.nfField))
        rows.set(entry.nfField, { kind: 'scheme', name: entry.nfField, main: [], advanced: [] })
      rows.get(entry.nfField)[isAdvanced(entry) ? 'advanced' : 'main'].push({ section, key })
    }
  }
  return [...rows.values()]
}

// A per-field form's collapsible rows: control blocks, scheme fields, then solver fields.
export const fieldRows = (schema) => {
  const sections = Object.entries(schema.properties ?? {})
  const controls = sections
    .filter(([, section]) => claimed(section).length === 0)
    .map(([name]) => ({ kind: 'section', name, section: name }))
  const solvers = sections.flatMap(([section, entry]) =>
    Object.entries(entry.nfFieldGroups ?? {}).map(([name, blocks]) => ({
      kind: 'solver',
      name,
      section,
      blocks,
      finalOf: entry.nfFinalOf ?? {},
    })),
  )
  return [...controls, ...schemeRows(sections), ...solvers]
}

export const deepEqual = (a, b) => {
  if (a === b) return true
  if (!isDict(a) || !isDict(b)) return false
  const keys = Object.keys(a)
  return keys.length === Object.keys(b).length && keys.every((key) => deepEqual(a[key], b[key]))
}

// A detached copy; `structuredClone` throws on the reactive proxies Vue hands out.
export const plainCopy = (value) =>
  value === undefined ? undefined : JSON.parse(JSON.stringify(value))

// Each `Final` block that repeats its base block, by name.
export const sameFinals = (rows, solvers) => {
  const same = {}
  for (const row of rows.filter(({ kind }) => kind === 'solver'))
    for (const name of row.blocks.filter((block) => row.finalOf[block]))
      same[name] = solvers?.[name] !== undefined && deepEqual(solvers[name], solvers[row.finalOf[name]])
  return same
}

const NESTED_LABEL = (key) => key[0].toUpperCase() + key.slice(1)

const advancedControls = (properties, prefix) =>
  Object.entries(properties)
    .filter(([, property]) => isAdvanced(property))
    .flatMap(([key, property]) =>
      property.type === 'object' && property.properties
        ? advancedControls(property.properties, [...prefix, key]).map((control) => ({
            ...control,
            group: property.nfGroup,
          }))
        : [{ path: [...prefix, key], schema: property, group: property.nfGroup }],
    )

const grouped = (controls, title, groupTitle) => {
  const groups = new Map()
  for (const control of controls) {
    const heading = control.group ? groupTitle(control.group) : title
    if (!groups.has(heading)) groups.set(heading, { title: heading, controls: [] })
    groups.get(heading).controls.push(control)
  }
  return [...groups.values()]
}

// The advanced entries of the block's current solver and nested selections, under headings.
export const advancedGroups = (schema, data) => {
  const arm = selectedArm(schema, data)
  if (!arm) return []
  const name = armLabel(arm)
  const own = grouped(advancedControls(arm.properties, []), `Solver: ${name}`, (group) => `${name}: ${group}`)
  const nested = Object.entries(arm.properties)
    .filter(([, property]) => Array.isArray(property.oneOf))
    .flatMap(([key, property]) => {
      const inner = selectedArm(property, data?.[key])
      if (!inner) return []
      const type = armLabel(inner)
      return grouped(
        advancedControls(inner.properties, [key]),
        `${NESTED_LABEL(key)}: ${type}`,
        (group) => `${type}: ${group}`,
      )
    })
  return [...own, ...nested]
}

export const valueAt = (data, path) => path.reduce((value, key) => value?.[key], data)

// `data` with the value at `path` set; undefined removes it and any dictionary left empty,
// so an option the user clears is not written.
export const withValueAt = (data, [key, ...rest], value) => {
  const next = { ...data }
  const child = rest.length ? withValueAt(data?.[key], rest, value) : value
  if (child === undefined || (rest.length && Object.keys(child).length === 0)) delete next[key]
  else next[key] = child
  return next
}

// An entry's hint, shown as a placeholder and never written. `false` is a hint too.
export const hintText = (schema) =>
  'nfDefaultHint' in schema ? String(schema.nfDefaultHint) : 'none'

const summaryNumber = (value) => {
  if (typeof value !== 'number') return String(value)
  const text = formatNumber(value)
  return text.replace(/e(-?)(\d)$/, 'e$10$2')
}

// A scheme value as the words of its file entry: `Gauss linear corrected`.
export const schemeWords = (value) => {
  if (!isDict(value)) return value === undefined ? '' : String(value)
  return Object.values(value).map(schemeWords).filter(Boolean).join(' ')
}

// One line about a solver block: `Cg · Multigrid · tol 1e-07 · relTol 0.01`.
export const solverSummary = (schema, data) => {
  const arm = selectedArm(schema, data)
  if (arm?.nfArmLabel) return arm.nfArmLabel
  const parts = [data?.solverName ?? arm?.title ?? data?.solver, data?.preconditioner?.type]
  if (data?.tolerance !== undefined) parts.push('tol ' + summaryNumber(data.tolerance))
  if (data?.relTol !== undefined) parts.push('relTol ' + summaryNumber(data.relTol))
  return parts.filter((part) => part !== undefined).join(' · ')
}

// One line about a control block: its main entries with their values.
export const sectionSummary = (section, data) =>
  Object.entries(section.properties ?? {})
    .filter(([key, property]) => !isAdvanced(property) && data?.[key] !== undefined)
    .map(([key]) => `${key} ${summaryNumber(data[key])}`)
    .join(' · ')

const withoutWords = (property, words = []) =>
  property?.examples
    ? { ...property, examples: property.examples.filter((word) => !words.includes(word)) }
    : property

const narrowedPreconditioner = (property, refused, preferred) => {
  const oneOf = property.oneOf
    .filter((arm) => !refused.preconditioner?.includes(arm.properties?.type?.const))
    .map((arm) =>
      arm.properties.smoother
        ? {
            ...arm,
            properties: {
              ...arm.properties,
              smoother: withoutWords(arm.properties.smoother, refused.smoother),
            },
          }
        : arm,
    )
  const offered = oneOf.some((arm) => arm.properties?.type?.const === preferred)
  return { ...property, oneOf, ...(offered ? { default: { type: preferred } } : {}) }
}

const narrowedArm = (arm, refused, preferred) => {
  const properties = { ...arm.properties }
  if (properties.smoother) properties.smoother = withoutWords(properties.smoother, refused.smoother)
  if (properties.preconditioner?.oneOf)
    properties.preconditioner = narrowedPreconditioner(properties.preconditioner, refused, preferred)
  return { ...arm, properties }
}

// A block's arms for its field: the choices its matrix refuses are gone, and a new block
// starts from the field's default preconditioner.
export const withFieldChoices = (block) => {
  if (!block.nfRefusedChoices && !block.nfDefaultChoices) return block
  const refused = block.nfRefusedChoices ?? {}
  const preferred = block.nfDefaultChoices?.preconditioner
  const key = discriminatorOf(block)
  const oneOf = block.oneOf
    .filter((arm) => !refused[key]?.includes(arm.properties?.[key]?.const))
    .map((arm) => narrowedArm(arm, refused, preferred))
  return { ...block, oneOf }
}

const undeclaredKeys = (arm, data) =>
  arm.additionalProperties === false
    ? null
    : Object.keys(data ?? {}).filter((key) => arm.properties?.[key] === undefined)

// The dictionaries of a block that accept keys no plugin models, with those it holds.
export const openKeyGroups = (schema, data) => {
  const arm = selectedArm(schema, data)
  if (!arm) return []
  const groups = []
  const own = undeclaredKeys(arm, data)
  if (own) groups.push({ title: 'Other keys', path: [], keys: own })
  for (const [key, property] of Object.entries(arm.properties)) {
    const inner = Array.isArray(property.oneOf) ? selectedArm(property, data?.[key]) : undefined
    const keys = inner && undeclaredKeys(inner, data?.[key])
    if (keys) groups.push({ title: `${armLabel(inner)}: other keys`, path: [key], keys })
  }
  return groups
}
