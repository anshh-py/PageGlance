import assert from 'node:assert/strict'
import { test } from 'node:test'
import { deleteDocument, importUrlDocument, uploadDocument } from './documents.ts'

function recordingFetcher(reply = {}, status = 200) {
  const calls = []
  const fetcher = async (url, options) => {
    calls.push({ url, options })
    if (status === 204) return new Response(null, { status })
    return Response.json(reply, { status })
  }
  fetcher.calls = calls
  return fetcher
}

test('deleteDocument attaches the bearer token when authorization is supplied', async () => {
  const fetcher = recordingFetcher({}, 204)
  await deleteDocument('d1', 'http://localhost:8000', fetcher, 'tok-1')
  const call = fetcher.calls[0]
  assert.equal(call.url, 'http://localhost:8000/documents/d1')
  assert.equal(call.options.method, 'DELETE')
  assert.equal(call.options.headers.Authorization, 'Bearer tok-1')
})

test('uploadDocument attaches the bearer token when authorization is supplied', async () => {
  const fetcher = recordingFetcher({
    document_id: 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    filename: 'a.txt',
    source_type: 'text',
    content_hash: 'a'.repeat(64),
    chunk_count: 1,
    duplicate: false,
    page_count: null,
  })
  const file = new File(['hello'], 'a.txt', { type: 'text/plain' })
  const result = await uploadDocument(file, 'http://localhost:8000', fetcher, 'tok-2')
  assert.equal(result.document_id, 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11')
  assert.equal(fetcher.calls[0].options.headers.Authorization, 'Bearer tok-2')
})

test('importUrlDocument attaches the bearer token when authorization is supplied', async () => {
  const fetcher = recordingFetcher({
    document_id: 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    chunk_count: 3,
    title: 'Example',
    source_url: 'https://example.com/article',
    indexed: true,
    fallback_code: null,
  })
  await importUrlDocument('https://example.com/article', 'http://localhost:8000', fetcher, 'tok-3')
  const call = fetcher.calls[0]
  assert.equal(call.options.headers.Authorization, 'Bearer tok-3')
  assert.equal(call.options.headers['Content-Type'], 'application/json')
})

test('document calls send no authorization header and no headers object when authorization is not supplied', async () => {
  const fetcher = recordingFetcher({}, 204)
  await deleteDocument('d1', 'http://localhost:8000', fetcher)
  assert.equal(fetcher.calls[0].options.headers, undefined)
})
