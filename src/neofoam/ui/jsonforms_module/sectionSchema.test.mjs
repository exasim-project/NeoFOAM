// Unit tests of the renderers' pure rules: `bun test` (run by test/ui/test_jsonforms_module.py).

import * as per from './sectionSchema.mjs'
import { describe, expect, test } from 'bun:test'
import {
  LAYOUTS,
  PATH_UNSAFE,
  LAYOUT_BY_KEYWORD,
  armFields,
  discriminatorLabel,
  discriminatorOf,
  entryKeys,
  formatNumber,
  keptValues,
  withoutRuleErrors,
  parseNumberInput,
  pinnedKeys,
  rowsSchema,
  layoutOf,
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

// Here a fill that changes `solver` and one more key is indistinguishable from a load,
// so the agent panel drops the stale `preconditioner` itself before it fills the form
// (`_drop_stale_companions`, agent_panel.py).
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
  for (const layout of Object.values(LAYOUT_BY_KEYWORD)) expect(LAYOUTS).toContain(layout)
  expect(layoutOf({ nfSolvers: true })).toBe('cards')
  expect(layoutOf({ type: 'object' })).toBeUndefined()
})

test.each(['wall.left', 'div(phi,alpha.water)', 'a[0]'])('%p cannot be part of a data path', (key) => {
  expect(PATH_UNSAFE.test(key)).toBe(true)
})

test('a plain key can be part of a data path', () => {
  expect(PATH_UNSAFE.test('div(phi,U)')).toBe(false)
})

const NUMBER = { type: 'number' }
const CHOICE = (...values) => ({ type: 'string', enum: values })
const flatArm = (solver, properties) => ({
  title: solver,
  properties: { solver: { const: solver, default: solver }, ...properties },
})
const CG = flatArm('Cg', { preconditioner: CHOICE('none', 'Ic'), tolerance: NUMBER })
const GMRES = flatArm('Gmres', { preconditioner: CHOICE('none'), tolerance: NUMBER })
const FILE = flatArm('configFile', { path: { type: 'string' } })
const enumFits = (property, value) => !property.enum || property.enum.includes(value)

test('a union names its arm by type unless the schema says otherwise', () => {
  expect(discriminatorOf({ oneOf: [] })).toBe('type')
  expect(discriminatorOf({ nfDiscriminator: 'solver' })).toBe('solver')
})

test('only a named discriminator titles the select', () => {
  expect(discriminatorLabel({ nfDiscriminator: 'solver' })).toBe('Solver')
  expect(discriminatorLabel({ oneOf: [] })).toBeNull()
})

test('a switch keeps a shared value the new arm accepts', () => {
  const data = { solver: 'Cg', preconditioner: 'none', tolerance: 1e-3 }
  expect(keptValues(CG, GMRES, data, enumFits)).toEqual({ preconditioner: 'none', tolerance: 1e-3 })
})

test('a switch drops a shared value the new arm rejects', () => {
  const data = { solver: 'Cg', preconditioner: 'Ic', tolerance: 1e-3 }
  expect(keptValues(CG, GMRES, data, enumFits)).toEqual({ tolerance: 1e-3 })
})

test('a switch to an arm sharing no field keeps nothing', () => {
  expect(keptValues(CG, FILE, { solver: 'Cg', tolerance: 1e-3 }, enumFits)).toEqual({})
})

test('a dictionary no arm of a named union claims is a free-form block', () => {
  const union = { nfDiscriminator: 'solver', oneOf: [CG, GMRES] }
  const section = { nfUnions: true, properties: { p: union } }
  const schema = rowsSchema(section, { k: { solver: 'PBiCGStab' } }, {})
  expect(schema.properties.k.nfDict).toBe(true)
})

test('an added entry naming an arm of a named union keeps the union', () => {
  const union = { nfDiscriminator: 'solver', oneOf: [CG, GMRES] }
  const section = { nfUnions: true, properties: { p: union } }
  const schema = rowsSchema(section, { k: { solver: 'Cg' } }, {})
  expect(schema.properties.k.oneOf).toBe(union.oneOf)
})

const TUNED = flatArm('Tuned', { tolerance: NUMBER, depth: NUMBER })
const NEEDS = { ...flatArm('Needs', { path: { type: 'string' }, tolerance: NUMBER }), required: ['path'] }

test('a named union hides a field only its arm declares while it is unset', () => {
  expect(armFields(TUNED, [TUNED, CG], { solver: 'Tuned' }, true)).toEqual(['tolerance'])
})

test('a named union draws an arm-only field once the data sets it', () => {
  expect(armFields(TUNED, [TUNED, CG], { depth: 2 }, true)).toEqual(['tolerance', 'depth'])
})

test('a named union always draws a required arm-only field', () => {
  expect(armFields(NEEDS, [NEEDS, CG], {}, true)).toEqual(['path', 'tolerance'])
})

test('a union on type draws every field of its arm', () => {
  expect(armFields(TUNED, [TUNED, CG], {}, false)).toEqual(['tolerance', 'depth'])
})

describe('per-field form rules', () => {
  const multigrid = {
    title: 'Multigrid',
    properties: {
      type: { const: 'Multigrid' },
      max_levels: { type: 'integer', nfLevel: 'advanced', nfGroup: 'Hierarchy', nfDefaultHint: 10 },
      coarsening: {
        type: 'object',
        nfLevel: 'advanced',
        nfGroup: 'Pgm coarsening',
        properties: { skip_sorting: { type: 'boolean', nfLevel: 'advanced', nfDefaultHint: false } },
      },
    },
  }
  const none = { title: 'none', properties: { type: { const: 'none' } } }
  const block = {
    nfDiscriminator: 'solver',
    oneOf: [
      {
        title: 'Cg',
        properties: {
          solver: { const: 'Cg' },
          preconditioner: { oneOf: [none, multigrid] },
          tolerance: { type: 'number' },
          minIter: { type: 'integer', nfLevel: 'advanced' },
        },
      },
    ],
  }

  test('advanced entries follow the selected arms', () => {
    const titles = (data) => per.advancedGroups(block, data).map((group) => group.title)
    expect(titles({ solver: 'Cg', preconditioner: { type: 'none' } })).toEqual(['Solver: Cg'])
    expect(titles({ solver: 'Cg', preconditioner: { type: 'Multigrid' } })).toEqual([
      'Solver: Cg',
      'Multigrid: Hierarchy',
      'Multigrid: Pgm coarsening',
    ])
  })

  test('an arm draws no advanced entry inline, even when set', () => {
    expect(per.armFields(block.oneOf[0], block.oneOf, { minIter: 2 }, true)).not.toContain('minIter')
  })

  test('clearing the last nested option removes its dictionary', () => {
    const data = { type: 'Multigrid', coarsening: { skip_sorting: true } }
    expect(per.withValueAt(data, ['coarsening', 'skip_sorting'], undefined)).toEqual({
      type: 'Multigrid',
    })
  })

  test('a false hint is a hint', () => {
    expect(per.hintText({ nfDefaultHint: false })).toBe('false')
    expect(per.hintText({})).toBe('none')
  })

  test('a switch drops what the new arm refuses', () => {
    const from = { properties: { tolerance: {} } }
    const to = { nfSwitchDrops: ['tolerance'], properties: { tolerance: {} } }
    expect(per.keptValues(from, to, { tolerance: 1 }, () => true)).toEqual({})
  })

  test('a Final block counts as same only when it equals its base', () => {
    const rows = [{ kind: 'solver', blocks: ['p', 'pFinal'], finalOf: { pFinal: 'p' } }]
    const p = { solver: 'Cg', preconditioner: { type: 'Ic' } }
    expect(per.sameFinals(rows, { p, pFinal: structuredClone(p) })).toEqual({ pFinal: true })
    expect(per.sameFinals(rows, { p, pFinal: { solver: 'Bicg' } })).toEqual({ pFinal: false })
    expect(per.sameFinals(rows, { p })).toEqual({ pFinal: false })
  })

  test('summaries read as in a case file', () => {
    const data = { solver: 'Cg', preconditioner: { type: 'Multigrid' }, tolerance: 1e-7, relTol: 0.01 }
    expect(per.solverSummary(block, data)).toBe('Cg · Multigrid · tol 1e-07 · relTol 0.01')
    expect(per.schemeWords({ type: 'Gauss', interpolation: { type: 'linear' } })).toBe('Gauss linear')
  })
})

describe('field choices of a typed OpenFOAM block', () => {
  const smoother = { type: 'string', examples: ['DIC', 'DILU'] }
  const gamg = { title: 'GAMG', properties: { type: { const: 'GAMG' }, smoother } }
  const dilu = { title: 'DILU', properties: { type: { const: 'DILU' } } }
  const pcg = {
    title: 'PCG',
    properties: { solver: { const: 'PCG' }, preconditioner: { oneOf: [gamg, dilu] } },
  }
  const pbicg = { title: 'PBiCG', properties: { solver: { const: 'PBiCG' } } }
  const other = {
    title: 'Other',
    properties: { solver: { const: 'Other' }, solverName: { type: 'string' } },
  }
  const block = {
    nfDiscriminator: 'solver',
    nfRefusedChoices: { solver: ['PBiCG'], preconditioner: ['DILU'], smoother: ['DILU'] },
    oneOf: [pcg, pbicg, other],
  }

  test('refused solvers, preconditioners and smoothers are gone', () => {
    const narrowed = per.withFieldChoices(block)
    expect(narrowed.oneOf.map((arm) => arm.title)).toEqual(['PCG', 'Other'])
    const arms = narrowed.oneOf[0].properties.preconditioner.oneOf
    expect(arms.map((arm) => arm.title)).toEqual(['GAMG'])
    expect(arms[0].properties.smoother.examples).toEqual(['DIC'])
  })

  test('a field default names the preconditioner of a new block', () => {
    const narrowed = per.withFieldChoices({
      nfDiscriminator: 'solver',
      nfDefaultChoices: { preconditioner: 'DILU' },
      oneOf: [pcg],
    })
    expect(narrowed.oneOf[0].properties.preconditioner.default).toEqual({ type: 'DILU' })
  })

  test('a block without field keywords is the same object', () => {
    const plain = { nfDiscriminator: 'solver', oneOf: [pcg] }
    expect(per.withFieldChoices(plain)).toBe(plain)
  })

  test('undeclared keys stay reachable, at both levels', () => {
    const data = {
      solver: 'PCG',
      custom: 1,
      preconditioner: { type: 'GAMG', mine: 2 },
    }
    expect(per.openKeyGroups(block, data)).toEqual([
      { title: 'Other keys', path: [], keys: ['custom'] },
      { title: 'GAMG: other keys', path: ['preconditioner'], keys: ['mine'] },
    ])
  })

  test('the kept block can always name its solver', () => {
    expect(per.armFields(other, block.oneOf, { solver: 'Other' }, true)).toContain('solverName')
    expect(per.solverSummary(block, { solver: 'Other', solverName: 'petsc' })).toBe('petsc')
  })
})

test('a required companion reports only its own message', () => {
  const errors = 'must match "then" schema\nis a required property'
  expect(withoutRuleErrors(errors)).toBe('is a required property')
  expect(withoutRuleErrors(undefined)).toBe('')
})
