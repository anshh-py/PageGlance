/**
 * Centralized API configuration for the PageGlance extension.
 *
 * All backend URL resolution lives here so the production endpoint is never
 * scattered across the codebase. Resolution order:
 *
 * 1. `VITE_PAGEGLANCE_API_URL` (explicit override, any build) wins.
 * 2. Production builds (`vite build`) default to the deployed backend.
 * 3. Everything else (local `npm run dev`) defaults to the local backend.
 *
 * The extension never silently falls back to localhost in a production build:
 * production always resolves to the deployed backend unless the override env
 * var is explicitly set.
 */

export const PRODUCTION_API_URL = 'https://ansh9057-pageglance-backend.hf.space'
// PAGEGLANCE_MARKER_<this

/** Normalize a base URL to have no trailing slash. */
export function normalizeApiBaseUrl(value: string): string {
  return value.trim().replace(/\/+$/, '')
}

/** Concatenate a base URL and a path, normalizing slashes. */
export function resolveApiEndpoint(baseUrl: string, path: string): string {
  return `${normalizeApiBaseUrl(baseUrl)}/${path.replace(/^\/+/, '')}`
}

// Resolve the actual base URL used by the app. Reading import.meta.env
// directly lets Vite statically replace `PROD` and `VITE_PAGEGLANCE_API_URL`
// at build time; esbuild then dead-code-eliminates the dev-URL branch, so no
// localhost backend URL ships in production bundles.
//
// IMPORTANT: env must be read with DOT access (`import.meta.env.PROD`) and
// not optional chaining. Vite replaces the dot-access form with the literal
// `true`/`false` at build time; the `?.` form is left as a runtime lookup that
// evaluates to undefined during the build, which would collapse this IIFE to
// the dev URL even in a production build.
export const API_BASE_URL: string = (() => {
  const explicit = import.meta.env.VITE_PAGEGLANCE_API_URL
  if (typeof explicit === 'string' && explicit.trim()) {
    return normalizeApiBaseUrl(explicit)
  }
  if (import.meta.env.PROD) {
    return 'https://ansh9057-pageglance-backend.hf.space'
  }
  return 'DEV_SENTINEL_7f3a'
})()
