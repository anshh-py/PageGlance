import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  authenticatedFetch,
  ensureInstallation,
  generateToken,
  INSTALLATION_TOKEN_KEY,
  loadStoredToken,
  registerToken,
  resetInstallationCache,
  saveToken,
  toBase64Url,
} from './auth.ts'

function makeStorage(initial = {}) {
  const data = { ...initial }
  return {
    data,
    get: async (keys) => {
      const out = {}
      for (const key of [].concat(keys)) {
        if (key in data) out[key] = data[key]
      }
      return out
    },
    set: async (items) => Object.assign(data, items),
  }
}

function recordingFetcher(reply = { installation_id: 'inst-123' }, status = 200) {
  const calls = []
  const fetcher = async (url, options) => {
    calls.push({ url, options })
    return Response.json(reply, { status })
  }
  fetcher.calls = calls
  return fetcher
}

test('toBase64Url encodes bytes as URL-safe unpadded base64', () => {
  const ascii = (s) => new Uint8Array([...s].map((c) => c.charCodeAt(0)))
  assert.equal(toBase64Url(ascii('abc')), 'YWJj')
  assert.equal(toBase64Url(ascii('ab')), 'YWI')
  assert.equal(toBase64Url(ascii('abcd')), 'YWJjZA')
  // Bytes that yield '+' and '/' in standard base64 become '-' and '_'.
  assert.equal(toBase64Url(new Uint8Array([0xfb, 0xef, 0xff])), '--__')
})

test('generateToken uses the crypto random source and returns a URL-safe token', () => {
  const originalGetRandomValues = globalThis.crypto.getRandomValues
  try {
    let calls = 0
    globalThis.crypto.getRandomValues = (bytes) => {
      calls += 1
      bytes.fill(0)
      return bytes
    }
    const token = generateToken()
    assert.ok(calls >= 1, 'expected the cryptographically secure random source to be used')
    assert.equal(token.length, 43)
    assert.match(token, /^[A-Za-z0-9_-]+$/)
    assert.equal(token, 'A'.repeat(43))
  } finally {
    globalThis.crypto.getRandomValues = originalGetRandomValues
  }
})

test('generateToken produces fresh random tokens', () => {
  const a = generateToken()
  const b = generateToken()
  assert.notEqual(a, b)
})

test('loadStoredToken and saveToken round-trip through storage', async () => {
  const storage = makeStorage()
  assert.equal(await loadStoredToken(storage), null)
  await saveToken('the-token', storage)
  assert.equal(storage.data[INSTALLATION_TOKEN_KEY], 'the-token')
  assert.equal(await loadStoredToken(storage), 'the-token')
})

test('registerToken posts to the real registration endpoint and parses installation id', async () => {
  const fetcher = recordingFetcher()
  const installationId = await registerToken('tok', 'http://localhost:8000', fetcher)
  assert.equal(installationId, 'inst-123')
  assert.equal(fetcher.calls.length, 1)
  assert.equal(fetcher.calls[0].url, 'http://localhost:8000/auth/register')
  assert.equal(fetcher.calls[0].options.method, 'POST')
  assert.deepEqual(JSON.parse(String(fetcher.calls[0].options.body)), { token: 'tok' })
})

test('registerToken rejects error responses and invalid payloads', async () => {
  await assert.rejects(
    registerToken('tok', 'http://x', recordingFetcher({ detail: 'nope' }, 500)),
    /registration failed with status 500/,
  )
  await assert.rejects(
    registerToken('tok', 'http://x', recordingFetcher({ nope: true })),
    /invalid registration response/,
  )
  await assert.rejects(
    registerToken('tok', 'http://x', async () => {
      throw new TypeError('Failed to fetch')
    }),
    /Could not register this installation/,
  )
})

test('ensureInstallation generates, stores and registers a token on first run', async () => {
  resetInstallationCache()
  const storage = makeStorage()
  const fetcher = recordingFetcher()
  const token = await ensureInstallation({ storage, apiUrl: 'http://localhost:8000', fetcher })

  assert.ok(token.length >= 40)
  assert.equal(storage.data[INSTALLATION_TOKEN_KEY], token)
  assert.equal(fetcher.calls.length, 1)
  assert.equal(JSON.parse(String(fetcher.calls[0].options.body)).token, token)
})

test('ensureInstallation caches and reuses the registered token', async () => {
  resetInstallationCache()
  const storage = makeStorage()
  const fetcher = recordingFetcher()
  const first = await ensureInstallation({ storage, apiUrl: 'http://localhost:8000', fetcher })
  const second = await ensureInstallation({ storage, apiUrl: 'http://localhost:8000', fetcher })
  assert.equal(first, second)
  assert.equal(fetcher.calls.length, 1, 'only one registration should be performed')
})

test('ensureInstallation keeps a stored token without regenerating it', async () => {
  resetInstallationCache()
  const existing = 'existing-token-value'
  const storage = makeStorage({ [INSTALLATION_TOKEN_KEY]: existing })
  const fetcher = recordingFetcher()
  const token = await ensureInstallation({ storage, apiUrl: 'http://localhost:8000', fetcher })
  assert.equal(token, existing)
  assert.equal(storage.data[INSTALLATION_TOKEN_KEY], existing)
  assert.equal(JSON.parse(String(fetcher.calls[0].options.body)).token, existing)
})

test('ensureInstallation retries after a failed registration once the cache is reset', async () => {
  resetInstallationCache()
  const storage = makeStorage()
  let attempt = 0
  const flaky = async (url, options) => {
    attempt += 1
    if (attempt === 1) return Response.json({ detail: 'down' }, { status: 503 })
    return Response.json({ installation_id: 'inst-9' }, { status: 200 })
  }
  await assert.rejects(ensureInstallation({ storage, apiUrl: 'http://localhost:8000', fetcher: flaky }))
  resetInstallationCache()
  const token = await ensureInstallation({ storage, apiUrl: 'http://localhost:8000', fetcher: flaky })
  assert.ok(token.length >= 40)
  assert.equal(attempt, 2)
})

test('authenticatedFetch attaches the bearer token header', async () => {
  resetInstallationCache()
  const storage = makeStorage()
  const fetcher = recordingFetcher()
  const response = await authenticatedFetch('http://localhost:8000/documents/x/stats', { method: 'GET' }, {
    storage,
    apiUrl: 'http://localhost:8000',
    fetcher,
  })
  assert.equal(response.status, 200)
  const call = fetcher.calls.at(-1)
  assert.equal(call.url, 'http://localhost:8000/documents/x/stats')
  const token = storage.data[INSTALLATION_TOKEN_KEY]
  assert.equal(call.options.headers.get('Authorization'), `Bearer ${token}`)
})
