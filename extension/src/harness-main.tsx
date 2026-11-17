/**
 * M8.4 browser-verification harness.
 *
 * Renders the real <App /> (no StrictMode, so restore/extraction effects run
 * once) with the mock chrome.* bridge and mock backend installed by
 * harness.html, and exposes driver buttons for the flows browser_check cannot
 * drive directly (file upload via a real File object).
 */
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'

declare global {
  interface Window {
    __mocks?: { failExtract: boolean; extractCount: number; uploadCount: number; importCount: number }
  }
}

document.documentElement.dataset.theme = 'dark'
document.documentElement.classList.add('theme-ready')

createRoot(document.getElementById('root')!).render(<App initialTheme="system" />)

function bind(id: string, fn: () => void) {
  const el = document.getElementById(id)
  if (el) el.addEventListener('click', fn)
}

function tick() {
  const el = document.getElementById('counts')
  if (el && window.__mocks) {
    const m = window.__mocks as { failExtract: boolean; extractCount: number; uploadCount: number; importCount: number }
    el.textContent = `extractions: ${m.extractCount}\nuploads: ${m.uploadCount}\nimports: ${m.importCount}`
  }
}

bind('btn-fail-extract', () => {
  const m = window.__mocks as { failExtract: boolean }
  m.failExtract = !m.failExtract
  const b = document.getElementById('btn-fail-extract')
  if (b) b.textContent = `failExtract: ${m.failExtract}`
})

bind('btn-auto-upload', () => {
  const input = document.querySelector<HTMLInputElement>('.document-file-input')
  if (!input) return
  const file = new File(['hello body'], 'demo.txt', { type: 'text/plain' })
  const dt = new DataTransfer()
  dt.items.add(file)
  input.files = dt.files
  input.dispatchEvent(new Event('change', { bubbles: true }))
})

bind('btn-bad-upload', () => {
  const input = document.querySelector<HTMLInputElement>('.document-file-input')
  if (!input) return
  const file = new File(['x'], 'bad.exe', { type: 'application/octet-stream' })
  const dt = new DataTransfer()
  dt.items.add(file)
  input.files = dt.files
  input.dispatchEvent(new Event('change', { bubbles: true }))
})

bind('btn-set-url', () => {
  const src = document.getElementById('driver-url') as HTMLInputElement | null
  const target = document.getElementById('ingest-url-input') as HTMLInputElement | null
  if (!src || !target) return
  // Use the native value setter so React's controlled input observes the change.
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
  setter?.call(target, src.value)
  target.dispatchEvent(new Event('input', { bubbles: true }))
  setTimeout(() => {
    const submit = target.form?.querySelector<HTMLButtonElement>('.url-import-submit')
    submit?.click()
  }, 120)
})

setInterval(tick, 200)
tick()
