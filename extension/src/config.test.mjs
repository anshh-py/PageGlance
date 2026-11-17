import assert from 'node:assert/strict'
import { test } from 'node:test'
import { DEV_API_URL, resolveApiBaseUrl } from './config.test-support.ts'
import {
  PRODUCTION_API_URL,
  resolveApiEndpoint,
  normalizeApiBaseUrl,
} from './config.ts'

test('production builds resolve to the deployed backend by default', () => {
  assert.equal(resolveApiBaseUrl({ PROD: true }), PRODUCTION_API_URL)
  assert.match(PRODUCTION_API_URL, /^https:\/\/ansh9057-pageglance-backend\.hf\.space$/)
})

test('development builds resolve to the local backend when no override is set', () => {
  assert.equal(resolveApiBaseUrl({}), DEV_API_URL)
  assert.equal(resolveApiBaseUrl(undefined), DEV_API_URL)
})

test('explicit VITE_PAGEGLANCE_API_URL override wins in both modes', () => {
  const override = 'https://example.com/backend/'
  assert.equal(resolveApiBaseUrl({ PROD: true, VITE_PAGEGLANCE_API_URL: override }), 'https://example.com/backend')
  assert.equal(resolveApiBaseUrl({ VITE_PAGEGLANCE_API_URL: override }), 'https://example.com/backend')
})

test('production does not silently fall back to localhost', () => {
  const prod = resolveApiBaseUrl({ PROD: true })
  assert.notStrictEqual(prod, DEV_API_URL)
  assert.notEqual(prod, 'http://127.0.0.1:8000')
  assert.notEqual(prod, 'http://localhost:8000')
})

test('api endpoint builder joins base and path without duplicate slashes', () => {
  assert.equal(resolveApiEndpoint('https://api.example.com/', '/ask/stream'), 'https://api.example.com/ask/stream')
  assert.equal(resolveApiEndpoint('https://api.example.com', 'documents/upload'), 'https://api.example.com/documents/upload')
})

test('base URL normalization strips trailing slashes', () => {
  assert.equal(normalizeApiBaseUrl('http://localhost:8000///'), 'http://localhost:8000')
  assert.equal(normalizeApiBaseUrl('http://localhost:8000'), 'http://localhost:8000')
})
