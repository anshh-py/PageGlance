import { browser } from './browser.ts'

/**
 * M9.2 installation authentication for the PageGlance extension.
 *
 * Each installation generates a cryptographically secure bearer token once,
 * stores it in `chrome.storage.local` (never hardcoded, never sent to logs),
 * and registers it with the backend's `/auth/register` endpoint. Every
 * protected API request then sends the token as a `Bearer` header. The backend
 * stores only a salted domain-separated hash of the token, so a leaked
 * database does not expose a usable credential.
 *
 * Registration is idempotent on the backend (keyed by token hash), so calling
 * it on every launch is safe and re-attaches an existing installation after a
 * backend database reset.
 */

export const INSTALLATION_TOKEN_KEY = 'pageGlanceInstallToken'

/** Number of random bytes in the bearer token (a 256-bit secret). */
export const TOKEN_BYTES = 32

export interface StorageLike {
  get: (keys: string | string[]) => Promise<Record<string, unknown>>
  set: (items: Record<string, unknown>) => Promise<void>
}

/** Encode raw bytes as a URL-safe base64 (no padding). */
export function toBase64Url(bytes: Uint8Array): string {
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

/** Generate a new cryptographically secure installation token. */
export function generateToken(): string {
  const cryptoObj = globalThis.crypto
  if (!cryptoObj || typeof cryptoObj.getRandomValues !== 'function') {
    throw new Error('A cryptographically secure random source is unavailable.')
  }
  const bytes = new Uint8Array(TOKEN_BYTES)
  cryptoObj.getRandomValues(bytes)
  return toBase64Url(bytes)
}

/** Read the stored installation token, if any. */
export async function loadStoredToken(storage: StorageLike = browser.storage.local): Promise<string | null> {
  const record = await storage.get(INSTALLATION_TOKEN_KEY)
  const value = record[INSTALLATION_TOKEN_KEY]
  return typeof value === 'string' && value.length > 0 ? value : null
}

/** Persist the installation token in `chrome.storage.local`. */
export async function saveToken(token: string, storage: StorageLike = browser.storage.local): Promise<void> {
  await storage.set({ [INSTALLATION_TOKEN_KEY]: token })
}

/** Resolve the backend's real registration endpoint from a base URL. */
function resolveRegistrationEndpoint(baseUrl: string): string {
  return `${baseUrl.replace(/\/+$/, '')}/auth/register`
}

/** Register the token with the backend's real `/auth/register` endpoint. */
export async function registerToken(
  token: string,
  apiUrl: string,
  fetcher: typeof fetch = fetch,
): Promise<string> {
  let response: Response
  try {
    response = await fetcher(resolveRegistrationEndpoint(apiUrl), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    })
  } catch {
    throw new Error('Could not register this installation with the PageGlance API.')
  }
  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    payload = null
  }
  if (!response.ok) {
    throw new Error(`Installation registration failed with status ${response.status}.`)
  }
  if (!payload || typeof payload !== 'object' || typeof (payload as { installation_id?: unknown }).installation_id !== 'string') {
    throw new Error('The backend returned an invalid registration response.')
  }
  return (payload as { installation_id: string }).installation_id
}

export interface EnsureInstallationOptions {
  storage?: StorageLike
  apiUrl?: string
  fetcher?: typeof fetch
}

let cachedPromise: Promise<string> | null = null

/** Reset the cached installation promise (for tests). */
export function resetInstallationCache(): void {
  cachedPromise = null
}

async function doEnsureInstallation(options: EnsureInstallationOptions = {}): Promise<string> {
  const storage = options.storage ?? browser.storage.local
  const apiUrl = options.apiUrl
  if (!apiUrl) {
    throw new Error('An API base URL is required to register the installation.')
  }
  let token = await loadStoredToken(storage)
  if (!token) {
    token = generateToken()
    await saveToken(token, storage)
  }
  await registerToken(token, apiUrl, options.fetcher)
  return token
}

/**
 * Ensure the installation token exists, is stored, and is registered with the
 * backend. Returns the token. The promise is cached so concurrent protected
 * calls share a single registration.
 */
export function ensureInstallation(options: EnsureInstallationOptions = {}): Promise<string> {
  if (!cachedPromise) {
    cachedPromise = doEnsureInstallation(options)
    cachedPromise.catch(() => {
      // Allow a later call to retry after a transient registration failure.
      cachedPromise = null
    })
  }
  return cachedPromise
}

/** Perform a fetch that authenticates with the installation bearer token. */
export async function authenticatedFetch(
  input: Parameters<typeof fetch>[0],
  init: RequestInit = {},
  options: EnsureInstallationOptions = {},
): Promise<Response> {
  const token = await ensureInstallation(options)
  const headers = new Headers(init.headers)
  headers.set('Authorization', `Bearer ${token}`)
  const fetcher = options.fetcher ?? fetch
  return fetcher(input, { ...init, headers })
}
