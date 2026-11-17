/**
 * Test-only API configuration support for the PageGlance extension.
 *
 * This module exists solely so the app-facing `config.ts` stays clean and the
 * local dev backend URL never ships in a production bundle. It is imported
 * only by tests; the extension never imports it at runtime.
 *
 * Resolution order (same semantics as the app's `API_BASE_URL`):
 *   1. `VITE_PAGEGLANCE_API_URL` (explicit override, any build) wins.
 *   2. Production builds default to the deployed backend.
 *   3. Everything else defaults to the local backend.
 */

import { normalizeApiBaseUrl, PRODUCTION_API_URL } from './config'

export const DEV_API_URL = 'http://127.0.0.1:8000'

export interface ApiEnv {
  VITE_PAGEGLANCE_API_URL?: string
  PROD?: boolean
}

/** Resolve the API base URL from a Vite-like env object. Pure and testable. */
export function resolveApiBaseUrl(env: ApiEnv | undefined): string {
  const explicit = env?.VITE_PAGEGLANCE_API_URL
  if (typeof explicit === 'string' && explicit.trim()) {
    return normalizeApiBaseUrl(explicit)
  }
  if (env?.PROD) {
    return PRODUCTION_API_URL
  }
  return DEV_API_URL
}
