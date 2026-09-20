// Trame client entry: registers a <json-forms> Vue component backed by JSONForms +
// the Vuetify renderers. Vue and Vuetify are externalized (globals provided by
// trame's client), so this bundle binds to trame's single Vue app / Vuetify plugin
// instance rather than a second copy.
//
// The wrapper injects `vuetifyRenderers` by default and re-emits JSONForms' `change`
// event, so trame templates only pass :schema / :uischema / :data and listen on
// @change — matching src/neofoam/ui/app.py.
//
// Number fields use a custom renderer: the stock @jsonforms/vue-vuetify one derives
// `precision` from its default `step` (0.1), so VNumberInput masks typed input to ONE
// decimal — "0.001" becomes "0.0". A CFD case needs deltaT=1e-3 and nu=1e-5, so we
// render a plain text field instead and parse any finite JS number (incl. "1e-5").
//
// `translations` (id -> text, built in form_schema.py) feeds JSONForms' i18n: a schema node's
// `i18n` prefix picks e.g. "nf.patch.propertyNameLabel" for its "add a key" row.
//
// A schema node flagged `nfCompact` (the fvSchemes sections, see form_schema.py `_tag_adders`)
// renders as one row per entry — key left, variant + nested selects inline — instead
// of JSONForms' stacked full-width selects. Both renderers below only lay out: data
// goes through the same JSONForms bindings and `createDefaultValue` as the stock ones.
//
// The same section renderer draws fvSolution's `solvers` (`nfSolvers`) as one card per
// linear solver, and a block (`nfSolver`, see form_schema.py `_pin_solver_controls`) as cells:
// `solver`, its preconditioner/smoother, `tolerance`, `relTol`, then the case's extra
// options, "+ add option" at the foot. Each block is its own JSONForms instance, so its
// paths start at the block: a solver named `alpha.water` never enters a dotted path.
//
// What these renderers decide from schema and data alone (an entry's schema, the pinned
// keys, the companion key to drop, how a number reads and parses) lives in
// sectionSchema.mjs, which imports no UI library and is unit-tested with `bun test`.
//
// A `boundaryField` map (`nfPatches`, see form_schema.py `_patch_adder`) is drawn as rows too, one
// per patch. In every variant an entry whose key holds `.`, `[` or `]` (`wall.left`,
// `div(phi,alpha.water)`) is such a form of its own as well, written back through the
// section's data; all other entries keep the cheap path binding.
//
// A config of free-form sub-dictionaries (`nfDicts`, see form_schema.py `_dictionary_cards`:
// MRFProperties, fvOptions) reuses the cards, each entry (`nfDict`) the cells of its keywords.
//
// An object of scalars only (`nfGrid`, see form_schema.py `_tag_scalar_grid`) keeps JSONForms'
// generated layout and controls; its layout renderer just sets them in a wrapping grid.

import { computed, defineComponent, h, nextTick, ref, watch } from 'vue'
import {
  DispatchRenderer,
  JsonForms,
  rendererProps,
  useJsonFormsControl,
  useJsonFormsLayout,
  useJsonFormsOneOfControl,
  useTranslator,
} from '@jsonforms/vue'
import { vuetifyRenderers } from '@jsonforms/vue-vuetify'
import {
  and,
  createAjv,
  createCombinatorRenderInfos,
  createDefaultValue,
  encode,
  getI18nKeyPrefix,
  isLayout,
  isNumberControl,
  isObjectControl,
  optionIs,
  rankWith,
  schemaMatches,
} from '@jsonforms/core'
import { VBtn, VCard, VCardText, VCardTitle, VSelect, VTextField } from 'vuetify/components'
import '@jsonforms/vue-vuetify/lib/jsonforms-vue-vuetify.css'
import './compact.css'
import {
  PATH_UNSAFE,
  entryKeys,
  entryShape,
  formatNumber,
  parseNumberInput,
  pinnedKeys,
  rowsSchema as sectionRowsSchema,
  variantOf,
  withoutStaleCompanions,
} from './sectionSchema.mjs'

// Inside a compact row or grid the error text shows only when there is one.
const INLINE_FIELD = { on: { hideDetails: 'auto' }, off: {} }
const INLINE_OPTIONS = {
  nfInline: true,
  vuetify: {
    'v-text-field': INLINE_FIELD.on,
    'v-select': INLINE_FIELD.on,
    'v-combobox': INLINE_FIELD.on,
    'v-checkbox': { ...INLINE_FIELD.on, density: 'compact' },
  },
}

// One control of `schema` (the property `key`), laid out inline.
const inlineChild = (control, schema, key) =>
  h(DispatchRenderer, {
    key,
    schema,
    uischema: { type: 'Control', scope: '#/properties/' + encode(key), options: INLINE_OPTIONS },
    path: control.path,
    enabled: control.enabled,
    renderers: control.renderers,
    cells: control.cells,
  })

const NfNumberControl = defineComponent({
  name: 'nf-number-control',
  props: { ...rendererProps() },
  setup(props) {
    const { control, handleChange } = useJsonFormsControl(props)
    const editing = ref(null) // raw text while the field has focus
    return { control, handleChange, editing }
  },
  computed: {
    display() {
      if (this.editing !== null) return this.editing
      const v = this.control.data
      return v === undefined || v === null ? '' : formatNumber(v)
    },
    // Text that can never become a number (`abc`, `0,5`) is not stored, so say so.
    inputError() {
      return parseNumberInput(this.editing, this.acceptsText)?.error ?? null
    },
    // An untyped option (`nSweeps 2`, `cacheAgglomeration on`) holds either.
    acceptsText() {
      return [this.control.schema.type].flat().includes('string')
    },
  },
  methods: {
    onInput(txt) {
      this.editing = txt
      const parsed = parseNumberInput(txt, this.acceptsText)
      if (parsed && 'value' in parsed) this.handleChange(this.control.path, parsed.value)
    },
  },
  render() {
    const c = this.control
    if (!c.visible) return null
    return h(VTextField, {
      id: c.id + '-input',
      modelValue: this.display,
      label: c.label + (c.required ? '*' : ''),
      disabled: !c.enabled,
      errorMessages: this.inputError ?? c.errors,
      hint: c.description,
      class: 'nf-number',
      ...INLINE_FIELD[c.uischema.options?.nfInline ? 'on' : 'off'],
      'onUpdate:modelValue': this.onInput,
      onBlur: () => {
        this.editing = null
      },
    })
  },
})

// A discriminated union as one select, its arm's own fields following on the same line.
const NfInlineOneOf = defineComponent({
  name: 'nf-inline-one-of',
  props: { ...rendererProps() },
  setup(props) {
    const { control, handleChange } = useJsonFormsOneOfControl(props)
    const arms = computed(() => {
      const c = control.value
      return createCombinatorRenderInfos(
        c.schema.oneOf, c.rootSchema, 'oneOf', c.uischema, c.path, c.uischemas,
      )
    })
    // The `type` const names the arm outright. JSONForms' own "fitting schema" probe
    // finds none while a required field of a freshly picked arm is still empty.
    const selected = computed(() => {
      const c = control.value
      const byType = arms.value.findIndex(
        (arm) => c.data && arm.schema.properties?.type?.const === c.data.type,
      )
      if (byType >= 0) return byType
      return c.indexOfFittingSchema ?? (c.data === undefined ? null : 0)
    })
    return { control, handleChange, arms, selected, t: useTranslator() }
  },
  render() {
    const c = this.control
    if (!c.visible) return null
    const arm = this.selected === null ? null : this.arms[this.selected]
    const topLevel = c.uischema.options.nfRowKey !== undefined
    const label = c.label + (c.required ? '*' : '')
    const fields = Object.keys(arm?.schema.properties ?? {}).filter(
      (key) => arm.schema.properties[key].const === undefined,
    )
    return h('div', { class: 'nf-inline' }, [
      h(VSelect, {
        id: c.id + '-input',
        // The row already shows the entry's key next to its first select.
        label: topLevel ? undefined : label,
        'aria-label': label,
        items: this.arms.map((info, index) => ({
          title: this.t(info.label, info.label),
          value: index,
        })),
        modelValue: this.selected,
        disabled: !c.enabled,
        errorMessages: c.errors,
        hideDetails: 'auto',
        'onUpdate:modelValue': (index) =>
          this.handleChange(c.path, createDefaultValue(this.arms[index].schema, c.rootSchema)),
      }),
      ...fields.map((key) => inlineChild(c, arm.schema, key)),
    ])
  },
})

// A solver block is a form of its own, rooted at the block.
const BLOCK_UISCHEMA = { type: 'Control', scope: '#' }
const blockAjv = createAjv()

// A section of like entries as a dense card: one row per entry, "+ add entry" at the foot.
const NfCompactSection = defineComponent({
  name: 'nf-compact-section',
  props: { ...rendererProps() },
  setup(props) {
    const { control, handleChange } = useJsonFormsControl(props)
    const like = computed(() => entryShape(control.value.schema))
    // Kept per entry: a fresh schema or uischema object would restart an entry's own form
    // on every edit.
    const added = {}
    const ownUischemas = {}
    const rowsSchema = computed(() =>
      sectionRowsSchema(control.value.schema, control.value.data, added),
    )
    watch(
      () => control.value.data,
      (now, before) => {
        const kept = withoutStaleCompanions(now, before, control.value.schema.nfSolver)
        if (kept) handleChange(control.value.path, kept)
      },
    )
    const newName = ref(null) // null: adder closed
    const t = useTranslator()
    const i18n = { translate: (...args) => t.value(...args) }
    return { control, handleChange, like, rowsSchema, newName, t, i18n, ownUischemas }
  },
  computed: {
    variant() {
      return variantOf(this.control.schema)
    },
    pinned() {
      return pinnedKeys(this.control.schema, this.control.data)
    },
    keys() {
      return entryKeys(this.variant.layout, this.rowsSchema, this.pinned, this.control.data)
    },
    i18nPrefix() {
      const c = this.control
      return getI18nKeyPrefix(c.schema, c.uischema, c.path + '.additionalProperties')
    },
    nameError() {
      const name = this.newName
      if (!name) return null
      if (this.keys.includes(name))
        return this.t(this.i18nPrefix + '.propertyAlreadyDefined', `'${name}' already defined`)
      return null
    },
  },
  methods: {
    add() {
      if (!this.newName || this.nameError) return
      // A keyword of a free-form dictionary starts as a text value.
      const value = createDefaultValue(this.like ?? { type: 'string' }, this.control.rootSchema)
      this.handleChange(this.control.path, { ...this.control.data, [this.newName]: value })
      this.closeAdder()
    },
    // The name field unmounts on close; without this the focus falls to <body>.
    closeAdder() {
      this.newName = null
      nextTick(() => this.$refs.addButton?.$el.focus())
    },
    remove(key) {
      const data = { ...this.control.data }
      delete data[key]
      this.handleChange(this.control.path, data)
    },
    setBlock(key, block) {
      const c = this.control
      // Every form reports its data once on mount; only an edit is written back.
      if (JSON.stringify(block) === JSON.stringify(c.data?.[key])) return
      this.handleChange(c.path, { ...c.data, [key]: block })
    },
    // An entry as a form of its own, rooted at the entry and written back through `setBlock`.
    ownForm(key, uischema) {
      const c = this.control
      return h(JsonForms, {
        schema: this.rowsSchema.properties[key],
        uischema,
        data: c.data?.[key],
        renderers: c.renderers,
        cells: c.cells,
        readonly: !c.enabled,
        ajv: blockAjv,
        i18n: this.i18n,
        onChange: (event) => this.setBlock(key, event.data),
      })
    },
    child(key, options) {
      const c = this.control
      if (PATH_UNSAFE.test(key))
        return this.ownForm(key, (this.ownUischemas[key] ??= { ...BLOCK_UISCHEMA, options }))
      return h(DispatchRenderer, {
        schema: this.rowsSchema,
        uischema: { type: 'Control', scope: '#/properties/' + encode(key), options },
        path: c.path,
        enabled: c.enabled,
        renderers: c.renderers,
        cells: c.cells,
      })
    },
    trash(key) {
      if (this.pinned.includes(key)) return null
      return h(VBtn, {
        icon: 'mdi-delete',
        variant: 'text',
        size: 'small',
        'aria-label': 'Delete ' + key,
        disabled: !this.control.enabled,
        onClick: () => this.remove(key),
      })
    },
    rowEntry(key) {
      const required = (this.control.schema.required ?? []).includes(key)
      return h('div', { class: 'nf-compact-row', key }, [
        h('span', { class: 'nf-compact-key' }, key + (required ? '*' : '')),
        h('div', { class: 'nf-compact-controls' }, [
          this.child(key, { ...INLINE_OPTIONS, nfRowKey: key }),
          this.trash(key),
        ]),
      ])
    },
    cardEntry(key) {
      return h('div', { class: 'nf-card-entry', key }, [
        this.ownForm(key, BLOCK_UISCHEMA),
        this.trash(key),
      ])
    },
    cellEntry(key) {
      const { examples: suggestion, type } = this.rowsSchema.properties[key]
      // A number takes one column, a name two, a nested dictionary the whole row.
      const size = { 'nf-cell-number': type === 'number', 'nf-cell-dict': type === 'object' }
      return h('div', { class: ['nf-cell', size], key }, [
        this.child(key, { ...INLINE_OPTIONS, suggestion }),
        this.trash(key),
      ])
    },
    adder() {
      if (this.newName === null)
        return h(
          VBtn,
          {
            ref: 'addButton',
            variant: 'text',
            size: 'small',
            color: 'primary',
            prependIcon: 'mdi-plus',
            disabled: !this.control.enabled,
            onClick: () => (this.newName = ''),
          },
          () => this.variant.add,
        )
      return h('div', { class: 'nf-compact-adder' }, [
        h(VTextField, {
          modelValue: this.newName,
          label: this.t(this.i18nPrefix + '.propertyNameLabel', 'Property Name'),
          autofocus: true,
          hideDetails: 'auto',
          errorMessages: this.nameError ?? [],
          'onUpdate:modelValue': (name) => (this.newName = name ?? ''),
          onKeydown: (event) => {
            if (event.key === 'Enter') {
              // Prevented, or the key press goes on to click the refocused button.
              event.preventDefault()
              this.add()
            }
            if (event.key === 'Escape') this.closeAdder()
          },
        }),
        h(
          VBtn,
          {
            variant: 'text',
            color: 'primary',
            disabled: !this.newName || !!this.nameError,
            onClick: this.add,
          },
          () => 'Add',
        ),
      ])
    },
  },
  render() {
    const c = this.control
    if (!c.visible) return null
    const { layout } = this.variant
    // Not `this[layout]`: `cells` is also a JSONForms prop of every renderer.
    const entry = { rows: this.rowEntry, cards: this.cardEntry, cells: this.cellEntry }[layout]
    return h(VCard, { class: 'nf-compact mb-2', flat: true, border: true }, () => [
      c.label ? h(VCardTitle, { class: 'text-subtitle-1' }, () => c.label) : null,
      h(VCardText, { class: { 'pt-0': c.label, 'pb-1': layout === 'cells' } }, () => [
        h('div', { class: 'nf-' + layout }, this.keys.map(entry)),
        this.adder(),
      ]),
    ])
  },
})

// The controls of an all-scalar object side by side; a nested one keeps its titled card.
const NfGridLayout = defineComponent({
  name: 'nf-grid-layout',
  props: { ...rendererProps() },
  setup(props) {
    return useJsonFormsLayout(props)
  },
  render() {
    const l = this.layout
    if (!l.visible) return null
    const nested = l.uischema.type === 'Group'
    const grid = h(
      'div',
      { class: ['nf-form-grid', { 'px-4': !nested }] },
      l.uischema.elements.map((element, index) =>
        h(DispatchRenderer, {
          key: index,
          schema: l.schema,
          uischema: { ...element, options: { ...element.options, ...INLINE_OPTIONS } },
          path: l.path,
          enabled: l.enabled,
          renderers: l.renderers,
          cells: l.cells,
        }),
      ),
    )
    if (!nested) return grid
    return h(VCard, { class: 'nf-compact mb-2', flat: true, border: true }, () => [
      h(VCardTitle, { class: 'text-subtitle-1' }, () => l.label),
      h(VCardText, { class: 'pt-0' }, () => grid),
    ])
  },
})

const isInline = optionIs('nfInline', true)
const renderers = [
  ...vuetifyRenderers,
  { tester: rankWith(30, isNumberControl), renderer: NfNumberControl },
  {
    tester: rankWith(40, and(isObjectControl, schemaMatches((s) => variantOf(s) !== undefined))),
    renderer: NfCompactSection,
  },
  {
    tester: rankWith(40, and(isInline, schemaMatches((s) => Array.isArray(s.oneOf)))),
    renderer: NfInlineOneOf,
  },
  { tester: rankWith(40, and(isLayout, (_, schema) => schema.nfGrid === true)), renderer: NfGridLayout },
]

// JSONForms stars a label from the static `required` list alone. While the schema's own
// `if` holds for the data, the keys its `then` requires join that list, so the label
// follows (`Nu*`); what validates stays the same.
const formAjv = createAjv()
const conditionalRequired = (schema, data) => {
  const { if: condition, then: { required, ...then } = {} } = schema
  if (!condition || !required || !formAjv.validate(condition, data)) return null
  return { ...schema, required: [...(schema.required ?? []), ...required], then }
}

const NeoFoamJsonForms = defineComponent({
  name: 'json-forms',
  props: {
    schema: { type: Object, default: () => ({}) },
    uischema: { type: Object, default: null },
    data: { type: Object, default: () => ({}) },
    translations: { type: Object, default: () => ({}) },
  },
  emits: ['change'],
  setup(props, { emit }) {
    const i18n = computed(() => ({
      translate: (id, defaultMessage) => props.translations[id] ?? defaultMessage,
    }))
    // Kept until the schema or the outcome of its `if` changes: a fresh schema object
    // would restart the form on every edit.
    let kept = {}
    const schema = computed(() => {
      const starred = conditionalRequired(props.schema, props.data)
      if (kept.source !== props.schema || kept.on !== !!starred)
        kept = { source: props.schema, on: !!starred, schema: starred ?? props.schema }
      return kept.schema
    })
    return () =>
      h(JsonForms, {
        schema: schema.value,
        // A schema that is itself a section has no property for a generated layout to show.
        uischema: props.uischema || (variantOf(props.schema) ? BLOCK_UISCHEMA : undefined),
        data: props.data,
        renderers,
        i18n: i18n.value,
        onChange: (event) => emit('change', event),
      })
  },
})

export function install(app) {
  app.component('json-forms', NeoFoamJsonForms)
}

export default { install }
