import { resolve } from 'node:path'
import { defineConfig } from 'vite'

const DIR = import.meta.dirname

/**
 * Content-script build: a single, self-contained classic script.
 * Declared via manifest content_scripts, which browsers run as classic
 * scripts (no ES module imports/exports allowed). A dedicated single-entry
 * IIFE build inlines all dependencies (including src/browser.ts) into
 * content.js so it has no top-level import/export and no external chunks.
 */
export default defineConfig({
  build: {
    outDir: resolve(DIR, 'dist/chrome'),
    // Keep the app build's output; only add the content script.
    emptyOutDir: false,
    rollupOptions: {
      input: {
        content: resolve(DIR, 'src/content.ts'),
      },
      output: {
        format: 'iife',
        entryFileNames: 'content.js',
      },
    },
  },
})
