// Unit tests of the renderers' pure rules: `bun test` (run by test/ui/test_jsonforms_module.py).

import { expect, test } from 'bun:test'
import {
  LAYOUTS,
  PATH_UNSAFE,
  VARIANTS,
  entryKeys,
  formatNumber,
  parseNumberInput,
  pinnedKeys,
  rowsSchema,
  variantOf,
  withoutStaleCompanions,
} from './sectionSchema.mjs'

const COMPANIONS = {
  PCG: 'preconditioner',
  PBiCGStab: 'preconditioner',
  GAMG: 'smoother',
  smoothSolver: 'smoother',
}
const PCG = { solver: 'PCG', preconditioner: 'DIC', tolerance: 1e-6, relTol: 0.05 }
const SOLVER_BLOCK = {
  type: 'object',
  nfSolver: COMPANIONS,
  required: ['solver'],
  properties: {
    solver: { type: 'string' },
    preconditioner: { type: ['string', 'object'] },
    smoother: { type: 'string' },
    tolerance: { type: 'number' },
    relTol: { type: 'number' },
  },
}
const SCHEMES = {
  type: 'object',
  nfCompact: true,
  properties: { default: { oneOf: [{ type: 'object' }] } },
}

test('a solver of the other family drops the companion it does not take', () => {
  const kept = withoutStaleCompanions({ ...PCG, solver: 'GAMG' }, PCG, COMPANIONS)

  expect(kept).toEqual({ solver: 'GAMG', tolerance: 1e-6, relTol: 0.05 })
})

test('a solver of the same family keeps its companion', () => {
  expect(withoutStaleCompanions({ ...PCG, solver: 'PBiCGStab' }, PCG, COMPANIONS)).toBeNull()
})

test('a solver the map does not know drops nothing', () => {
  expect(withoutStaleCompanions({ ...PCG, solver: 'Ginkgo' }, PCG, COMPANIONS)).toBeNull()
})

test('a loaded block that differs in several keys keeps both companions', () => {
  const loaded = { solver: 'GAMG', smoother: 'DIC', preconditioner: 'DILU', tolerance: 1e-8 }

  expect(withoutStaleCompanions(loaded, PCG, COMPANIONS)).toBeNull()
})

// Pinned, not endorsed: a fill that changes `solver` and one more key is
// indistinguishable from a load, so the stale `preconditioner` stays.
test('a fill that changes the solver and one other key is read as a load', () => {
  const filled = { ...PCG, solver: 'GAMG', relTol: 0.01 }

  expect(withoutStaleCompanions(filled, PCG, COMPANIONS)).toBeNull()
})

test('a block without a previous value or a companion map drops nothing', () => {
  expect(withoutStaleCompanions({ ...PCG, solver: 'GAMG' }, undefined, COMPANIONS)).toBeNull()
  expect(withoutStaleCompanions({ ...PCG, solver: 'GAMG' }, PCG, undefined)).toBeNull()
})

test.each([
  [1e-6, '1e-6'],
  [2.5e-7, '2.5e-7'],
  [1e6, '1e6'],
  [0.05, '0.05'],
  [0, '0'],
  [-1e-5, '-1e-5'],
  [999999, '999999'],
  ['on', 'on'],
  ['1e-6', '1e-6'],
])('formatNumber(%p) reads %p', (value, text) => {
  expect(formatNumber(value)).toBe(text)
})

test.each([
  ['1e-5', { value: 1e-5 }],
  [' 0.5 ', { value: 0.5 }],
  ['', { value: undefined }],
  ['-', null],
  ['1e', null],
  ['1e-', null],
  ['.', null],
])('typed %p parses to %p', (text, parsed) => {
  expect(parseNumberInput(text, false)).toEqual(parsed)
})

test.each(['0,5', 'abc', '1e-5x'])('typed %p is an error and stores nothing', (text) => {
  const parsed = parseNumberInput(text, false)

  expect(parsed.error).toContain(`'${text}' is not a number`)
  expect('value' in parsed).toBe(false)
})

test('an untyped option stores a number as a number and a word as text', () => {
  expect(parseNumberInput('2', true)).toEqual({ value: 2 })
  expect(parseNumberInput('on', true)).toEqual({ value: 'on' })
})

test('a solver block pins solver, the companion its solver takes, tolerance and relTol', () => {
  expect(pinnedKeys(SOLVER_BLOCK, PCG)).toEqual(['solver', 'preconditioner', 'tolerance', 'relTol'])
  expect(pinnedKeys(SOLVER_BLOCK, { solver: 'GAMG' })).toEqual([
    'solver',
    'smoother',
    'tolerance',
    'relTol',
  ])
  expect(pinnedKeys(SOLVER_BLOCK, { solver: 'Ginkgo' })).toEqual(['solver', 'tolerance', 'relTol'])
})

test('any other section pins its declared keys', () => {
  expect(pinnedKeys(SCHEMES, { default: {}, 'div(phi,U)': {} })).toEqual(['default'])
})

test('cells list the pinned keys before the extra options of the data', () => {
  const data = { nSweeps: 2, ...PCG }
  const keys = entryKeys('cells', rowsSchema(SOLVER_BLOCK, data, {}), pinnedKeys(SOLVER_BLOCK, data), data)

  expect(keys).toEqual(['solver', 'preconditioner', 'tolerance', 'relTol', 'nSweeps'])
})

test('rows list the declared keys, then the entries only the data holds', () => {
  const data = { 'div(phi,U)': { type: 'Gauss' } }

  expect(entryKeys('rows', rowsSchema(SCHEMES, data, {}), ['default'], data)).toEqual([
    'default',
    'div(phi,U)',
  ])
})

test('an added entry keeps the same schema object across edits', () => {
  const cache = {}
  const before = rowsSchema(SCHEMES, { 'div(phi,k)': { type: 'Gauss' } }, cache)
  const after = rowsSchema(SCHEMES, { 'div(phi,k)': { type: 'Gauss', scheme: 'upwind' } }, cache)

  expect(after.properties['div(phi,k)']).toBe(before.properties['div(phi,k)'])
  expect(after.properties['div(phi,k)']).toEqual({ ...SCHEMES.properties.default, title: 'div(phi,k)' })
})

test('an added entry that turns from a value into a dictionary gets a new schema', () => {
  const cache = {}
  const asValue = rowsSchema(SOLVER_BLOCK, { nSweeps: 2 }, cache).properties.nSweeps
  const asDict = rowsSchema(SOLVER_BLOCK, { nSweeps: {} }, cache).properties.nSweeps

  expect(asValue.type).toContain('number')
  expect(asDict.type).toBe('object')
})

test('a string-or-dictionary entry is drawn as what the data holds', () => {
  const named = rowsSchema(SOLVER_BLOCK, PCG, {}).properties.preconditioner
  const nested = rowsSchema(SOLVER_BLOCK, { ...PCG, preconditioner: {} }, {}).properties.preconditioner

  expect(named.type).toBe('string')
  expect(nested).toMatchObject({ type: 'object', additionalProperties: true })
})

test('the companion of the chosen solver reads as required', () => {
  expect(rowsSchema(SOLVER_BLOCK, { solver: 'GAMG' }, {}).required).toEqual(['solver', 'smoother'])
  expect(rowsSchema(SOLVER_BLOCK, { solver: 'Ginkgo' }, {}).required).toEqual(['solver'])
})

test('every section keyword names a layout the renderer draws', () => {
  for (const { layout } of Object.values(VARIANTS)) expect(LAYOUTS).toContain(layout)
  expect(variantOf({ nfSolvers: true }).layout).toBe('cards')
  expect(variantOf({ type: 'object' })).toBeUndefined()
})

test.each(['wall.left', 'div(phi,alpha.water)', 'a[0]'])('%p cannot be part of a data path', (key) => {
  expect(PATH_UNSAFE.test(key)).toBe(true)
})

test('a plain key can be part of a data path', () => {
  expect(PATH_UNSAFE.test('div(phi,U)')).toBe(false)
})
