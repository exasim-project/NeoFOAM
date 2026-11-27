// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// The per-field numerics form (`nfFieldForm`, see field_form.py): one collapsible row per
// field, its main entries inline and the rest under "All entries". What it decides from
// schema and data alone lives in sectionSchema.mjs.

import { computed, defineComponent, h, reactive, ref, watch } from 'vue'
import { JsonForms, rendererProps, useJsonFormsControl, useTranslator } from '@jsonforms/vue'
import { createAjv } from '@jsonforms/core'
import { VCombobox, VIcon, VSelect, VSwitch, VTextField } from 'vuetify/components'
import {
  BLOCK_UISCHEMA,
  INLINE_OPTIONS,
  advancedGroups,
  deepEqual,
  fieldRows,
  formatNumber,
  hintText,
  isAdvanced,
  openKeyGroups,
  parseNumberInput,
  plainCopy,
  sameFinals,
  schemeWords,
  sectionSummary,
  solverSummary,
  valueAt,
  withFieldChoices,
  withValueAt,
} from './sectionSchema.mjs'

const blockAjv = createAjv()
const INLINE_BLOCK_UISCHEMA = { ...BLOCK_UISCHEMA, options: INLINE_OPTIONS }
// Keys no plugin models, each editable and removable, nested dictionaries included.
const OPEN_KEYS_SCHEMA = {
  type: 'object',
  additionalProperties: true,
  nfDict: true,
  i18n: 'nf.option',
  entryLike: 'self',
}
const BOOLEAN_CHOICES = [
  { title: 'On', value: true },
  { title: 'Off', value: false },
]

// One advanced option: empty means unset, so it is not written; the hint is a placeholder.
const NfOptionField = defineComponent({
  name: 'nf-option-field',
  props: { schema: Object, value: {}, label: String, disabled: Boolean },
  emits: ['update'],
  setup() {
    return { editing: ref(null) }
  },
  computed: {
    choices() {
      if (this.schema.type === 'boolean') return BOOLEAN_CHOICES
      return this.schema.enum?.map((value) => ({ title: String(value), value }))
    },
    parsed() {
      if (this.schema.type === 'string') return { value: this.editing?.trim() || undefined }
      const parsed = parseNumberInput(this.editing, false)
      if (this.schema.type !== 'integer' || !Number.isFinite(parsed?.value)) return parsed
      if (Number.isInteger(parsed.value)) return parsed
      return { error: `'${this.editing.trim()}' is not a whole number` }
    },
    display() {
      if (this.editing !== null) return this.editing
      if (this.value === undefined) return ''
      return this.schema.type === 'string' ? this.value : formatNumber(this.value)
    },
  },
  methods: {
    onInput(text) {
      this.editing = text
      if (this.parsed && 'value' in this.parsed) this.$emit('update', this.parsed.value)
    },
  },
  render() {
    const common = {
      label: this.label,
      placeholder: hintText(this.schema),
      persistentPlaceholder: true,
      disabled: this.disabled,
      hideDetails: 'auto',
    }
    if (this.choices)
      return h(VSelect, {
        ...common,
        items: this.choices,
        modelValue: this.value ?? null,
        clearable: true,
        'onUpdate:modelValue': (value) => this.$emit('update', value ?? undefined),
      })
    if (this.schema.examples)
      return h(VCombobox, {
        ...common,
        items: this.schema.examples,
        modelValue: this.value ?? null,
        clearable: true,
        'onUpdate:modelValue': (value) => this.$emit('update', value || undefined),
      })
    return h(VTextField, {
      ...common,
      class: 'nf-number',
      modelValue: this.display,
      errorMessages: this.parsed?.error ?? [],
      'onUpdate:modelValue': this.onInput,
      onBlur: () => (this.editing = null),
    })
  },
})

const chevron = (open) =>
  h(VIcon, { class: ['nf-chevron', { 'nf-chevron-open': open }], icon: 'mdi-chevron-down', size: 18 })

const solverRows = (rows) => rows.filter(({ kind }) => kind === 'solver')

export const NfFieldForm = defineComponent({
  name: 'nf-field-form',
  props: { ...rendererProps() },
  setup(props) {
    const { control, handleChange } = useJsonFormsControl(props)
    const rows = computed(() => fieldRows(control.value.schema))
    const same = reactive(sameFinals(rows.value, control.value.data?.solvers))
    // A block changed from outside no longer repeats its base, so its switch must not
    // overwrite it on the next edit of the base.
    watch(
      () => control.value.data,
      (data) => {
        for (const row of solverRows(rows.value))
          for (const name of row.blocks.filter((block) => same[block])) {
            const blocks = data?.[row.section]
            if (!deepEqual(blocks?.[name], blocks?.[row.finalOf[name]])) same[name] = false
          }
      },
    )
    const t = useTranslator()
    const i18n = { translate: (...args) => t.value(...args) }
    // Identity-stable schemas and uischemas: a fresh object restarts a nested form.
    const kept = {}
    const keep = (key, build) => (kept[key] ??= build())
    const open = reactive({})
    const allOpen = reactive({})
    return { control, handleChange, rows, same, open, allOpen, i18n, keep }
  },
  methods: {
    section(name) {
      return this.control.schema.properties[name]
    },
    write(section, patch) {
      const data = this.control.data ?? {}
      const current = data[section] ?? {}
      if (Object.entries(patch).every(([key, value]) => deepEqual(value, current[key]))) return
      this.handleChange(this.control.path, { ...data, [section]: { ...current, ...patch } })
    },
    // Whole, so a key the form cleared leaves the data.
    replaceSection(name, section) {
      const data = this.control.data ?? {}
      if (deepEqual(section, data[name] ?? {})) return
      this.handleChange(this.control.path, { ...data, [name]: section })
    },
    writeBlock(row, name, block) {
      const mirrored = row.blocks.filter((other) => row.finalOf[other] === name && this.same[other])
      const copies = Object.fromEntries(mirrored.map((other) => [other, plainCopy(block)]))
      this.write(row.section, { [name]: block, ...copies })
    },
    toggleSame(row, name) {
      this.same[name] = !this.same[name]
      if (!this.same[name]) return
      const base = this.control.data?.[row.section]?.[row.finalOf[name]]
      this.write(row.section, { [name]: plainCopy(base) })
    },
    form(schema, uischema, data, onData) {
      const c = this.control
      return h(JsonForms, {
        schema,
        uischema,
        data,
        renderers: c.renderers,
        cells: c.cells,
        readonly: !c.enabled,
        ajv: blockAjv,
        i18n: this.i18n,
        onChange: (event) => onData(event.data),
      })
    },
    allEntries(id, count, body) {
      if (!count) return null
      const isOpen = !!this.allOpen[id]
      return h('div', { class: 'nf-field-all' }, [
        h(
          'button',
          {
            type: 'button',
            class: 'nf-all-toggle',
            'aria-expanded': String(isOpen),
            onClick: () => (this.allOpen[id] = !isOpen),
          },
          [chevron(isOpen), isOpen ? 'Hide all entries' : `All entries (${count} more)`],
        ),
        isOpen ? h('div', { class: 'nf-all' }, body()) : null,
      ])
    },
    optionField(row, name, block, option) {
      return h('div', { key: option.path.join('.') }, [
        h(NfOptionField, {
          schema: option.schema,
          value: valueAt(block, option.path),
          label: option.schema.title,
          disabled: !this.control.enabled,
          onUpdate: (value) => {
            const current = this.control.data?.[row.section]?.[name]
            this.writeBlock(row, name, withValueAt(current, option.path, value))
          },
        }),
        h('span', { class: 'nf-option-key' }, option.path[option.path.length - 1]),
      ])
    },
    optionGroups(row, name, groups, block) {
      return groups.map((group) =>
        h('div', { key: group.title }, [
          h('h4', { class: 'nf-all-heading' }, group.title),
          h(
            'div',
            { class: 'nf-all-grid' },
            group.controls.map((option) => this.optionField(row, name, block, option)),
          ),
        ]),
      )
    },
    // Keys no plugin models: kept in the data, so they must stay reachable.
    openKeysForm(row, name, block, group) {
      const holder = group.path.length ? valueAt(block, group.path) : block
      const rest = Object.fromEntries(group.keys.map((key) => [key, holder[key]]))
      const merge = (data) => {
        if (deepEqual(data ?? {}, rest)) return
        const current = this.control.data?.[row.section]?.[name]
        const inner = group.path.length ? valueAt(current, group.path) : current
        const declared = Object.fromEntries(
          Object.entries(inner).filter(([key]) => !group.keys.includes(key)),
        )
        const next = { ...declared, ...data }
        const updated = group.path.length ? withValueAt(current, group.path, next) : next
        this.writeBlock(row, name, updated)
      }
      return h('div', { key: 'open:' + group.title }, [
        h('h4', { class: 'nf-all-heading' }, group.title),
        this.form(OPEN_KEYS_SCHEMA, BLOCK_UISCHEMA, rest, merge),
      ])
    },
    finalHead(row, name) {
      return h('div', { class: 'nf-final-head' }, [
        h('span', ['Final corrector, ', h('code', name)]),
        h(VSwitch, {
          label: `Same as ${row.finalOf[name]}`,
          modelValue: !!this.same[name],
          color: 'primary',
          hideDetails: true,
          disabled: !this.control.enabled,
          'onUpdate:modelValue': () => this.toggleSame(row, name),
        }),
      ])
    },
    solverBlock(row, name) {
      const section = this.section(row.section)
      const schema = this.keep('block:' + name, () => withFieldChoices(section.properties[name]))
      const block = this.control.data?.[row.section]?.[name]
      const isFinal = row.finalOf[name] !== undefined
      const star = (section.required ?? []).includes(name) ? '*' : ''
      const groups = advancedGroups(schema, block)
      const openGroups = openKeyGroups(schema, block)
      const count =
        groups.reduce((total, group) => total + group.controls.length, 0) +
        openGroups.reduce((total, group) => total + group.keys.length, 0)
      return h('div', { class: ['nf-field-block', { 'nf-field-final': isFinal }], key: name }, [
        isFinal ? this.finalHead(row, name) : null,
        h('div', { style: isFinal && this.same[name] ? { display: 'none' } : undefined }, [
          h('div', { class: 'nf-compact-row nf-union-row' }, [
            h('span', { class: 'nf-compact-key' }, name + star),
            h('div', { class: 'nf-compact-controls' }, [
              this.form(schema, INLINE_BLOCK_UISCHEMA, block, (data) =>
                this.writeBlock(row, name, data),
              ),
            ]),
            this.allEntries(name, count || openGroups.length, () => [
              ...this.optionGroups(row, name, groups, block),
              ...openGroups.map((group) => this.openKeysForm(row, name, block, group)),
            ]),
          ]),
        ]),
      ])
    },
    schemeEntry({ section, key }) {
      const schema = this.section(section).properties[key]
      const uischema = this.keep('ui:' + key, () => ({
        ...BLOCK_UISCHEMA,
        options: { ...INLINE_OPTIONS, nfRowKey: key },
      }))
      return h('div', { class: 'nf-compact-row', key: `${section}:${key}` }, [
        h('span', { class: 'nf-compact-key' }, key),
        h('div', { class: 'nf-compact-controls' }, [
          this.form(schema, uischema, this.control.data?.[section]?.[key], (value) =>
            this.write(section, { [key]: value }),
          ),
        ]),
      ])
    },
    sectionForm(name, level) {
      const section = this.section(name)
      const schema = this.keep(`section:${name}:${level}`, () => ({
        ...section,
        properties: Object.fromEntries(
          Object.entries(section.properties ?? {}).filter(
            ([, property]) => isAdvanced(property) === (level === 'advanced'),
          ),
        ),
      }))
      if (!Object.keys(schema.properties).length) return null
      return this.form(schema, undefined, this.control.data?.[name] ?? {}, (data) =>
        this.replaceSection(name, data),
      )
    },
    rowBody(row) {
      if (row.kind === 'solver') return row.blocks.map((name) => this.solverBlock(row, name))
      if (row.kind === 'scheme')
        return [
          ...row.main.map(this.schemeEntry),
          this.allEntries('scheme:' + row.name, row.advanced.length, () => [
            h('h4', { class: 'nf-all-heading' }, `Further schemes for ${row.name}`),
            ...row.advanced.map(this.schemeEntry),
          ]),
        ]
      const properties = Object.values(this.section(row.section).properties ?? {})
      return [
        this.sectionForm(row.section, 'main'),
        this.allEntries('section:' + row.name, properties.filter(isAdvanced).length, () => [
          this.sectionForm(row.section, 'advanced'),
        ]),
      ]
    },
    rowSummary(row) {
      const data = this.control.data ?? {}
      if (row.kind === 'solver') {
        const schema = this.section(row.section).properties[row.name]
        return solverSummary(schema, data[row.section]?.[row.name])
      }
      if (row.kind === 'scheme')
        return row.main.map(({ section, key }) => schemeWords(data[section]?.[key])).join(' · ')
      return sectionSummary(this.section(row.section), data[row.section])
    },
    fieldRow(id, name, summary, body) {
      const isOpen = !!this.open[id]
      return h('div', { class: 'nf-field', key: id }, [
        h(
          'button',
          {
            type: 'button',
            class: 'nf-field-head',
            'aria-expanded': String(isOpen),
            onClick: () => (this.open[id] = !isOpen),
          },
          [
            h('span', { class: 'nf-field-name' }, name),
            h('span', { class: 'nf-field-summary' }, summary),
            chevron(isOpen),
          ],
        ),
        // Kept mounted: a closed row still mirrors its Final block.
        h('div', { class: 'nf-field-body', style: isOpen ? undefined : { display: 'none' } }, body),
      ])
    },
    backendNote() {
      const note = this.control.schema.nfBackend
      if (!note) return null
      const words = note.openfoam.flatMap((word, index) => [
        index ? ', ' : '',
        h('code', { key: word }, word),
      ])
      return h('div', { class: 'nf-backend' }, [
        h('span', ['Backend ', h('b', note.name), ', set by the application solver. ']),
        words.length
          ? h('span', [
              'OpenFOAM entries such as ',
              ...words,
              ` load as their ${note.name} equivalents.`,
            ])
          : null,
      ])
    },
  },
  render() {
    if (!this.control.visible) return null
    return h('div', { class: 'nf-field-form' }, [
      this.backendNote(),
      ...this.rows.map((row) =>
        this.fieldRow(`${row.kind}:${row.name}`, row.name, this.rowSummary(row), this.rowBody(row)),
      ),
    ])
  },
})
