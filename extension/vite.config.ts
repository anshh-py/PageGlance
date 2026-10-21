import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        popup: resolve(import.meta.dirname, 'index.html'),
        chat: resolve(import.meta.dirname, 'chat.html'),
        content: resolve(import.meta.dirname, 'src/content.ts'),
      },
      output: {
        entryFileNames: (chunk) =>
          chunk.name === 'content' ? 'content.js' : 'assets/[name]-[hash].js',
      },
    },
  },
})
