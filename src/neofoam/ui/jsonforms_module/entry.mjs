// Trame client entry: registers a <json-forms> Vue component backed by JSONForms +
// the Vuetify renderers. Vue and Vuetify are externalized (globals provided by
// trame's client), so this bundle binds to trame's single Vue app / Vuetify plugin
// instance rather than a second copy.
//
// The wrapper injects `vuetifyRenderers` by default and re-emits JSONForms' `change`
// event, so trame templates only pass :schema / :uischema / :data and listen on
// @change — matching src/neofoam/ui/app.py.

import { defineComponent, h } from 'vue'
import { JsonForms } from '@jsonforms/vue'
import { vuetifyRenderers } from '@jsonforms/vue-vuetify'
import '@jsonforms/vue-vuetify/lib/jsonforms-vue-vuetify.css'

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
        renderers: vuetifyRenderers,
        onChange: (event) => emit('change', event),
      })
  },
})

export function install(app) {
  app.component('json-forms', NeoFoamJsonForms)
}

export default { install }
