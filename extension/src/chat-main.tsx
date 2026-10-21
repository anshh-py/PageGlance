import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import type { ThemePreference } from './main'

const THEME_STORAGE_KEY = 'pageGlanceTheme'

function isThemePreference(value: unknown): value is ThemePreference {
  return value === 'system' || value === 'light' || value === 'dark'
}

async function mountFullPage() {
  let theme: ThemePreference = 'system'
  try {
    const stored = await chrome.storage.local.get(THEME_STORAGE_KEY)
    if (isThemePreference(stored[THEME_STORAGE_KEY])) theme = stored[THEME_STORAGE_KEY]
  } catch {
    // Keep the system preference when extension storage is unavailable.
  }
  document.documentElement.dataset.theme = theme
  document.documentElement.classList.add('theme-ready')
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App initialTheme={theme} fullPage />
    </StrictMode>,
  )
}

void mountFullPage()
