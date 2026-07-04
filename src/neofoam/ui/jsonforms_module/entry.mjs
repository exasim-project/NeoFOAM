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

import { defineComponent, h, ref } from 'vue'
import { JsonForms, rendererProps, useJsonFormsControl } from '@jsonforms/vue'
import { vuetifyRenderers } from '@jsonforms/vue-vuetify'
import { rankWith, isNumberControl } from '@jsonforms/core'
import { VTextField } from 'vuetify/components'
import '@jsonforms/vue-vuetify/lib/jsonforms-vue-vuetify.css'

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
      'onUpdate:modelValue': this.onInput,
      onBlur: () => {
        this.editing = null
      },
    })
  },
})

const renderers = [
  ...vuetifyRenderers,
  { tester: rankWith(30, isNumberControl), renderer: NfNumberControl },
]

const NeoFoamJsonForms = defineComponent({
  name: 'json-forms',
  props: {
    schema: { type: Object, default: () => ({}) },
    uischema: { type: Object, default: null },
    data: { type: Object, default: () => ({}) },
  },
  emits: ['change'],
  setup(props, { emit }) {
    return () =>
      h(JsonForms, {
        schema: props.schema,
        uischema: props.uischema || undefined,
        data: props.data,
        renderers,
        onChange: (event) => emit('change', event),
      })
  },
})

export function install(app) {
  app.component('json-forms', NeoFoamJsonForms)
}

export default { install }
