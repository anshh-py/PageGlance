import assert from 'node:assert/strict'
import { test } from 'node:test'
import { assertImportableUrl, importUrlDocument } from './documents.ts'

const importReply = {
  document_id: 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
  chunk_count: 3,
  title: 'Example',
  source_url: 'https://example.com/article',
  indexed: true,
  fallback_code: null,
}

test('rejects unsupported URL schemes before any backend call', () => {
  for (const url of [
    'javascript:alert(1)',
    'file:///etc/passwd',
    'data:text/html,<script>alert(1)</script>',
    'about:blank',
    'chrome://extensions',
    'ftp://example.com/file',
  ]) {
    assert.throws(() => assertImportableUrl(url), /Only web addresses/)
  }
})

test('rejects blank and credential-bearing URLs', () => {
  assert.throws(() => assertImportableUrl('   '), /Enter a URL/)
  assert.throws(() => assertImportableUrl('https://user:secret@example.com/'), /must not contain embedded credentials/)
})

test('accepts http/https URLs and posts to the ingest endpoint', async () => {
  const url = 'https://example.com/article'
  const document = await importUrlDocument(url, 'http://localhost:8000', async (requestUrl, options) => {
    assert.equal(requestUrl, 'http://localhost:8000/ingest/url')
    assert.equal(options.method, 'POST')
    assert.deepEqual(JSON.parse(String(options.body)), { url: 'https://example.com/article' })
    return Response.json(importReply, { status: 200 })
  })
  assert.deepEqual(document, importReply)
})

test('does not report success when the backend has nothing to index', async () => {
  const document = await importUrlDocument(
    'https://example.com/empty',
    'http://localhost:8000',
    async () => Response.json(
      { ...importReply, document_id: 'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', indexed: false, fallback_code: 'empty_page_text' },
      { status: 200 },
    ),
  )
  assert.equal(document.indexed, false)
  assert.equal(document.fallback_code, 'empty_page_text')
})

test('surfaces backend validation errors without leaking credentials', async () => {
  await assert.rejects(
    importUrlDocument('https://example.com/', 'http://localhost:8000', async () =>
      Response.json({ detail: 'Private or local addresses are not allowed.' }, { status: 400 }),
    ),
    /Private or local addresses are not allowed./,
  )
})

test('rejects a malformed backend response and unreachable backend', async () => {
  await assert.rejects(
    importUrlDocument('https://example.com/', 'http://localhost:8000', async () =>
      Response.json({ nope: true }, { status: 200 }),
    ),
    /invalid import information/,
  )
  await assert.rejects(
    importUrlDocument('https://example.com/', 'http://localhost:8000', async () => {
      throw new TypeError('Failed to fetch')
    }),
    /Could not reach the PageGlance API/,
  )
})
