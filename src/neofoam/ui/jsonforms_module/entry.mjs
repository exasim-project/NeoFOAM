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
// `translations` (id -> text, built in forms.py) feeds JSONForms' i18n: a schema node's
// `i18n` prefix picks e.g. "nf.patch.propertyNameLabel" for its "add a key" row.
//
// A schema node flagged `nfCompact` (the fvSchemes sections, see forms.py `_tag_adders`)
// renders as one row per entry — key left, variant + nested selects inline — instead
// of JSONForms' stacked full-width selects. Both renderers below only lay out: data
// goes through the same JSONForms bindings and `createDefaultValue` as the stock ones.

import { computed, defineComponent, h, ref } from 'vue'
import {
  DispatchRenderer,
  JsonForms,
  rendererProps,
  useJsonFormsControl,
  useJsonFormsOneOfControl,
  useTranslator,
} from '@jsonforms/vue'
import { vuetifyRenderers } from '@jsonforms/vue-vuetify'
import {
  and,
  createCombinatorRenderInfos,
  createDefaultValue,
  encode,
  getI18nKeyPrefix,
  isNumberControl,
  isObjectControl,
  optionIs,
  rankWith,
  schemaMatches,
} from '@jsonforms/core'
import { VBtn, VCard, VCardText, VCardTitle, VSelect, VTextField } from 'vuetify/components'
import '@jsonforms/vue-vuetify/lib/jsonforms-vue-vuetify.css'
import './compact.css'

// Inside a compact row the error text shows only when there is one.
const INLINE_FIELD = { on: { hideDetails: 'auto' }, off: {} }
const INLINE_OPTIONS = {
  nfInline: true,
  vuetify: { 'v-text-field': INLINE_FIELD.on, 'v-select': INLINE_FIELD.on },
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
      return v === undefined || v === null ? '' : String(v)
    },
  },
  methods: {
    onInput(txt) {
      this.editing = txt
      const t = (txt ?? '').trim()
      if (t === '') {
        this.handleChange(this.control.path, undefined)
        return
      }
      const n = Number(t)
      if (Number.isFinite(n)) this.handleChange(this.control.path, n)
      // Partial input ("1e-", "-") keeps the last valid value until it parses.
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
      errorMessages: c.errors,
      hint: c.description,
      inputmode: 'decimal',
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

const JSON_TYPES = ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string']

// A section of like entries as a dense card: one row per entry, "+ add entry" at the foot.
const NfCompactSection = defineComponent({
  name: 'nf-compact-section',
  props: { ...rendererProps() },
  setup(props) {
    const { control, handleChange } = useJsonFormsControl(props)
    const declared = computed(() => control.value.schema.properties ?? {})
    // An entry only present in the data (a loaded case, or one added here) is drawn
    // like its declared siblings when it has their object shape, else as a raw value.
    const rowsSchema = computed(() => {
      const like = Object.values(declared.value)[0]
      const properties = { ...declared.value }
      for (const [key, value] of Object.entries(control.value.data ?? {})) {
        if (key in properties) continue
        const isObject = value !== null && typeof value === 'object' && !Array.isArray(value)
        properties[key] = { ...(isObject ? like : { type: JSON_TYPES }), title: key }
      }
      return { ...control.value.schema, properties }
    })
    const newName = ref(null) // null: adder closed
    return { control, handleChange, declared, rowsSchema, newName, t: useTranslator() }
  },
  computed: {
    i18nPrefix() {
      const c = this.control
      return getI18nKeyPrefix(c.schema, c.uischema, c.path + '.additionalProperties')
    },
    nameError() {
      const name = this.newName
      if (!name) return null
      if (name in this.rowsSchema.properties)
        return this.t(this.i18nPrefix + '.propertyAlreadyDefined', `'${name}' already defined`)
      if (/[.[\]]/.test(name))
        return this.t(this.i18nPrefix + '.propertyNameInvalid', `'${name}' is invalid`)
      return null
    },
  },
  methods: {
    add() {
      if (!this.newName || this.nameError) return
      const like = Object.values(this.declared)[0]
      const value = createDefaultValue(like, this.control.rootSchema)
      this.handleChange(this.control.path, { ...this.control.data, [this.newName]: value })
      this.newName = null
    },
    remove(key) {
      const data = { ...this.control.data }
      delete data[key]
      this.handleChange(this.control.path, data)
    },
    row(key) {
      const c = this.control
      const required = (c.schema.required ?? []).includes(key)
      return h('div', { class: 'nf-compact-row', key }, [
        h('span', { class: 'nf-compact-key' }, key + (required ? '*' : '')),
        h('div', { class: 'nf-compact-controls' }, [
          h(DispatchRenderer, {
            schema: this.rowsSchema,
            uischema: {
              type: 'Control',
              scope: '#/properties/' + encode(key),
              options: { ...INLINE_OPTIONS, nfRowKey: key },
            },
            path: c.path,
            enabled: c.enabled,
            renderers: c.renderers,
            cells: c.cells,
          }),
          key in this.declared
            ? null
            : h(VBtn, {
                icon: 'mdi-delete',
                variant: 'text',
                size: 'small',
                'aria-label': 'Delete ' + key,
                disabled: !c.enabled,
                onClick: () => this.remove(key),
              }),
        ]),
      ])
    },
    adder() {
      if (this.newName === null)
        return h(
          VBtn,
          {
            variant: 'text',
            size: 'small',
            color: 'primary',
            prependIcon: 'mdi-plus',
            disabled: !this.control.enabled,
            onClick: () => (this.newName = ''),
          },
          () => 'Add entry',
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
            if (event.key === 'Enter') this.add()
            if (event.key === 'Escape') this.newName = null
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
    return h(VCard, { class: 'nf-compact mb-2', flat: true, border: true }, () => [
      h(VCardTitle, { class: 'text-subtitle-1' }, () => c.label),
      h(VCardText, { class: 'pt-0' }, () => [
        ...Object.keys(this.rowsSchema.properties).map(this.row),
        this.adder(),
      ]),
    ])
  },
})

const isInline = optionIs('nfInline', true)
const renderers = [
  ...vuetifyRenderers,
  { tester: rankWith(30, isNumberControl), renderer: NfNumberControl },
  {
    tester: rankWith(40, and(isObjectControl, schemaMatches((s) => s.nfCompact === true))),
    renderer: NfCompactSection,
  },
  {
    tester: rankWith(40, and(isInline, schemaMatches((s) => Array.isArray(s.oneOf)))),
    renderer: NfInlineOneOf,
  },
]

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
    return () =>
      h(JsonForms, {
        schema: props.schema,
        uischema: props.uischema || undefined,
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
