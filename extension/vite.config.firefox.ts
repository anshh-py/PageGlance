import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'
import { defineConfig } from 'vite'

const DIR = import.meta.dirname

/**
 * Main app build: popup (index.html) + chat (chat.html).
 * Loaded via <script type="module">, so they stay ESM.
 */
const appConfig = defineConfig({
  plugins: [react()],
  build: {
    outDir: resolve(DIR, 'dist/firefox'),
    rollupOptions: {
      input: {
        popup: resolve(DIR, 'index.html'),
        chat: resolve(DIR, 'chat.html'),
      },
      output: {
        entryFileNames: 'assets/[name]-[hash].js',
      },
    },
  },
})

export default appConfig
