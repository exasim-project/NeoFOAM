// Vite lib build: bundle the JSONForms Vuetify component into a single UMD that
// trame serves as a client module. Vue + Vuetify (all subpaths) are externalized
// to trame's client globals `Vue` / `Vuetify` so there is exactly ONE Vue app and
// ONE Vuetify plugin instance (a second copy would break Vue provide/inject via
// mismatched module-level Symbols). @mdi icon data is small and bundled in.
//
// Note: JSONForms imports `vuetify/labs/VStepperVertical`, which trame's Vuetify
// 3.11.2 UMD does NOT include — it maps to `Vuetify.VStepperVertical` (undefined)
// and only matters if a vertical-stepper uischema renders (ours don't).
//
// Regenerate: `bun install && bunx vite build` (outputs static/).

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// trame injects module scripts as ASYNC, so our bundle can evaluate before trame's
// vuetify3.js has attached the `Vuetify` global — a load-order race. If we mapped
// the vuetify externals straight to `Vuetify`/`Vuetify.components`, the UMD wrapper
// would dereference them at load and throw ("reading 'components' of undefined").
//
// Instead we map them to lazy Proxy globals (defined in `banner`, before the
// wrapper) that forward each property access to `window.Vuetify.*` at ACCESS time —
// which happens inside render/setup, after Vuetify is ready. Vue is fine as a direct
// global (`vue.global.js` is a non-async classic script, always loaded first).
const banner = `;(function(g){
  var P = function(get){ return new Proxy(function(){}, { get: function(_, k){ return get(k); } }); };
  g.__nfVuetify = g.__nfVuetify || P(function(k){ return g.Vuetify ? g.Vuetify[k] : undefined; });
  g.__nfVComp   = g.__nfVComp   || P(function(k){ return g.Vuetify && g.Vuetify.components ? g.Vuetify.components[k] : undefined; });
  g.__nfVDir    = g.__nfVDir    || P(function(k){ return g.Vuetify && g.Vuetify.directives ? g.Vuetify.directives[k] : undefined; });
})(typeof globalThis!=='undefined'?globalThis:window);`

const globalFor = (id) => {
  if (id === 'vue') return 'Vue'
  if (id === 'vuetify') return '__nfVuetify'
  if (id === 'vuetify/components') return '__nfVComp'
  if (id === 'vuetify/directives') return '__nfVDir'
  return '__nfVuetify' // vuetify/labs/* and any other vuetify subpath
}

export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: 'static',
    emptyOutDir: false,
    cssCodeSplit: false,
    lib: {
      entry: 'entry.mjs',
      name: 'neofoam_jsonforms',
      formats: ['umd'],
      fileName: () => 'neofoam_jsonforms.umd.js',
    },
    rollupOptions: {
      external: [/^vue$/, /^vuetify$/, /^vuetify\//],
      output: {
        exports: 'named',
        assetFileNames: 'neofoam_jsonforms.css',
        globals: globalFor,
        banner,
      },
    },
  },
})
