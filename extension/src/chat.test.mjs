import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  beginNewChat,
  buildConversationHistory,
  createMessage,
  normalizePageUrl,
  parseConversationStore,
  parseAnswerSources,
  restoreConversation,
  runSuggestion,
  SUGGESTIONS,
} from './chat.ts'
import { clearConversationStore, loadConversationStore, saveConversationStore } from './chatStorage.ts'
import { loadDocumentConversations, saveDocumentConversations } from './documentStorage.ts'
import {
  addUploadedDocument,
  citationItems,
  createScopedAskRequest,
  MAX_FILE_BYTES,
  parseUploadedDocuments,
  resolveCitationId,
  removeUploadedDocument,
  selectInitialDocument,
  uploadDocument,
  uploadStateReducer,
} from './documents.ts'
import { consumeAnswerStream } from './api.ts'

const memory = {}
Object.defineProperty(globalThis, 'chrome', {
  configurable: true,
  value: {
    storage: {
      local: {
        get: async (key) => ({ [key]: memory[key] }),
        set: async (items) => Object.assign(memory, items),
        remove: async (key) => { delete memory[key] },
      },
    },
  },
})

test('normalizes equivalent page URLs without merging distinct pages', () => {
  assert.equal(
    normalizePageUrl('https://Example.com/article?utm_source=mail#section'),
    normalizePageUrl('https://example.com/article'),
  )
  assert.notEqual(
    normalizePageUrl('https://example.com/article?id=1'),
    normalizePageUrl('https://example.com/article?id=2'),
  )
})

test('saves, restores, isolates, and clears conversations by page URL', async () => {
  const firstUrl = normalizePageUrl('https://example.com/one')
  const secondUrl = normalizePageUrl('https://example.com/two')
  const firstConversation = {
    pageUrl: firstUrl,
    pageTitle: 'First page',
    updatedAt: Date.now(),
    draft: 'Follow up',
    messages: [createMessage('user', 'Summarize'), createMessage('assistant', 'Summary', 'complete')],
  }
  const secondConversation = {
    pageUrl: secondUrl,
    pageTitle: 'Second page',
    updatedAt: Date.now(),
    draft: '',
    messages: [createMessage('user', 'Explain this')],
  }
  const store = {
    version: 1,
    conversations: { [firstUrl]: firstConversation, [secondUrl]: secondConversation },
  }
  await saveConversationStore(store)
  const restoredStore = await loadConversationStore()

  assert.equal(restoredStore.conversations[firstUrl].pageTitle, 'First page')
  assert.equal(restoredStore.conversations[firstUrl].draft, 'Follow up')
  assert.equal(restoredStore.conversations[secondUrl].messages[0].content, 'Explain this')
  await clearConversationStore()
  assert.deepEqual((await loadConversationStore()).conversations, {})
})

test('New Chat resets only the current page and preserves other saved pages', () => {
  const currentUrl = normalizePageUrl('https://example.com/current')
  const otherUrl = normalizePageUrl('https://example.com/other')
  const otherConversation = {
    pageUrl: otherUrl,
    pageTitle: 'Other page',
    updatedAt: 1,
    draft: '',
    messages: [createMessage('user', 'Keep this chat')],
  }
  const result = beginNewChat(
    { version: 1, conversations: { [otherUrl]: otherConversation } },
    currentUrl,
    'Current page',
  )

  assert.equal(result.conversation.messages.length, 0)
  assert.equal(result.store.conversations[currentUrl].pageTitle, 'Current page')
  assert.equal(result.store.conversations[otherUrl].messages[0].content, 'Keep this chat')
})

test('corrupt and old records are ignored; popup restoration marks a live stream interrupted', () => {
  assert.deepEqual(parseConversationStore({ version: 0, conversations: {} }).conversations, {})
  const pageUrl = normalizePageUrl('https://example.com/page')
  const store = parseConversationStore({
    version: 1,
    conversations: {
      [pageUrl]: {
        pageTitle: 'Example',
        messages: [
          createMessage('user', 'Question'),
          createMessage('assistant', 'partial answer', 'streaming'),
          { role: 'system', content: 'invalid' },
        ],
      },
    },
  })
  const restored = restoreConversation(store, pageUrl, 'Example')

  assert.equal(restored.interrupted, true)
  assert.equal(restored.conversation.messages.length, 2)
  assert.equal(restored.conversation.messages[1].content, 'partial answer')
  assert.equal(restored.conversation.messages[1].status, 'interrupted')
})

test('all welcome suggestions submit a concrete question', () => {
  assert.deepEqual(SUGGESTIONS.map((suggestion) => suggestion.title), [
    'Summarize this page',
    'Extract key takeaways',
    'Generate questions and answers',
    'Ask a question',
  ])
  for (const suggestion of SUGGESTIONS) {
    assert.ok(suggestion.title)
    assert.ok(suggestion.question.endsWith('?') || suggestion.question.endsWith('.'))
    const submitted = []
    runSuggestion(suggestion.id, (question) => submitted.push(question))
    assert.deepEqual(submitted, [suggestion.question])
  }
})

test('follow-up history stays bounded and marks interrupted assistant output', () => {
  const messages = [
    createMessage('user', 'Old unrelated detail'),
    createMessage('assistant', 'Partial result', 'interrupted'),
    createMessage('user', 'Explain the second point'),
  ]
  const history = buildConversationHistory(messages, 2, 150)

  assert.equal(history.length, 2)
  assert.match(history[0].content, /ended before completion and may be incomplete/)
  assert.equal(history[1].content, 'Explain the second point')
  assert.ok(history.reduce((count, message) => count + message.content.length, 0) <= 150)
})

const uploadReply = {
  document_id: '5f4a9974-8f53-4fc3-a71d-3369557d03d1',
  filename: 'guide.pdf',
  source_type: 'pdf',
  content_hash: 'a'.repeat(64),
  chunk_count: 4,
  duplicate: false,
  page_count: 3,
}

test('upload UI state reducer represents uploading, success, and readable errors', () => {
  let state = { status: 'idle' }
  state = uploadStateReducer(state, { type: 'start', filename: 'guide.pdf' })
  assert.deepEqual(state, { status: 'uploading', filename: 'guide.pdf' })
  state = uploadStateReducer(state, { type: 'success', message: 'Uploaded guide.pdf.' })
  assert.deepEqual(state, { status: 'success', message: 'Uploaded guide.pdf.' })
  state = uploadStateReducer(state, { type: 'error', message: 'PDF is malformed.' })
  assert.deepEqual(state, { status: 'error', message: 'PDF is malformed.' })
})

test('upload API posts multipart data and validates the backend response', async () => {
  let request
  const document = await uploadDocument(
    new File(['document contents'], 'guide.pdf', { type: 'application/pdf' }),
    'http://localhost:8000/',
    async (url, options) => {
      request = { url, options }
      return Response.json(uploadReply, { status: 200 })
    },
  )

  assert.equal(request.url, 'http://localhost:8000/documents/upload')
  assert.equal(request.options.method, 'POST')
  assert.equal(request.options.body.get('file').name, 'guide.pdf')
  assert.equal(request.options.headers, undefined)
  assert.deepEqual(document, uploadReply)
})

test('upload API surfaces server validation errors and rejects unsupported types locally', async () => {
  await assert.rejects(
    uploadDocument(new File(['x'], 'scan.png'), 'http://localhost:8000'),
    /Choose a \.txt or \.pdf file/,
  )
  await assert.rejects(
    uploadDocument(
      new File(['x'], 'bad.txt'),
      'http://localhost:8000',
      async () => Response.json({ detail: 'No readable text was found.' }, { status: 422 }),
    ),
    /No readable text was found/,
  )
  assert.equal(MAX_FILE_BYTES, 25 * 1024 * 1024)
  await assert.rejects(
    uploadDocument(
      { name: 'large.pdf', size: MAX_FILE_BYTES + 1 },
      'http://localhost:8000',
    ),
    /Files must be 25 MiB or smaller/,
  )
})

test('upload API accepts a file at the configured 25 MiB boundary', async () => {
  const file = new File(['small test fixture'], 'boundary.txt')
  Object.defineProperty(file, 'size', { value: MAX_FILE_BYTES })
  let sent = false
  await uploadDocument(file, 'http://localhost:8000', async () => {
    sent = true
    return Response.json({ ...uploadReply, source_type: 'text' }, { status: 200 })
  })
  assert.equal(sent, true)
})

test('uploaded documents can be selected and document asks omit webpage context', () => {
  const stored = addUploadedDocument([], { ...uploadReply, file_size_bytes: 1234 }, 10)
  assert.equal(parseUploadedDocuments(stored)[0].document_id, uploadReply.document_id)
  assert.equal(parseUploadedDocuments(stored)[0].file_size_bytes, 1234)
  assert.equal(selectInitialDocument(stored, null), stored[0])
  assert.equal(selectInitialDocument(stored, 'missing'), stored[0])
  assert.equal(selectInitialDocument(stored, uploadReply.document_id), stored[0])
  assert.equal(selectInitialDocument([], null), null)
  assert.deepEqual(
    createScopedAskRequest(
      'Question',
      { title: 'Unrelated page', url: 'https://example.com', text: 'page body' },
      stored[0],
      [],
    ),
    { question: 'Question', document_id: uploadReply.document_id, history: [] },
  )
})

test('removing an uploaded document removes only its local entry and later asks omit its ID', () => {
  const documents = addUploadedDocument([], uploadReply, 10)
  const other = {
    ...uploadReply,
    document_id: '8c04b574-4a09-4ad7-bc5a-4d75e19f30a1',
    filename: 'other.txt',
  }
  const library = addUploadedDocument(documents, other, 20)
  const remaining = removeUploadedDocument(library, uploadReply.document_id)
  const page = { title: 'Current page', url: 'https://example.com', text: 'Page context' }
  const selectedAfterRemoval = remaining.find((document) => document.document_id === uploadReply.document_id) ?? null

  assert.deepEqual(remaining.map((document) => document.document_id), [other.document_id])
  assert.equal(selectedAfterRemoval, null)
  assert.deepEqual(createScopedAskRequest('Question after removal', page, selectedAfterRemoval, []), {
    question: 'Question after removal', page, history: [],
  })
  assert.deepEqual(removeUploadedDocument(remaining, 'missing'), remaining)
})

test('webpage and uploaded-document requests stay mutually scoped', () => {
  const page = { title: 'Current page', url: 'https://example.com/article', text: 'Page-only evidence' }
  const document = { document_id: uploadReply.document_id }
  assert.deepEqual(createScopedAskRequest('Question about page', page, null, []), {
    question: 'Question about page', page, history: [],
  })
  assert.deepEqual(createScopedAskRequest('Question about file', page, document, []), {
    question: 'Question about file', document_id: uploadReply.document_id, history: [],
  })
})

test('switching between webpage and document contexts restores independent histories', async () => {
  const pageUrl = normalizePageUrl('https://example.com/context-switch')
  const webpageConversation = {
    pageUrl, pageTitle: 'Current page', updatedAt: 10, draft: 'Page draft',
    messages: [createMessage('user', 'Page question')],
  }
  const documentConversation = {
    updatedAt: 20, draft: 'Document draft', messages: [createMessage('user', 'Document question')],
  }
  await saveConversationStore({ version: 1, conversations: { [pageUrl]: webpageConversation } })
  await saveDocumentConversations({ [uploadReply.document_id]: documentConversation })

  const restoredPage = (await loadConversationStore()).conversations[pageUrl]
  const restoredDocument = (await loadDocumentConversations())[uploadReply.document_id]
  assert.equal(restoredPage.draft, 'Page draft')
  assert.equal(restoredPage.messages[0].content, 'Page question')
  assert.equal(restoredDocument.draft, 'Document draft')
  assert.equal(restoredDocument.messages[0].content, 'Document question')
})

test('citation presentation includes source labels and PDF page references', () => {
  assert.deepEqual(citationItems([{
    id: 'S2',
    document_id: uploadReply.document_id,
    title: 'Guide',
    source_type: 'pdf',
    filename: 'guide.pdf',
    source_url: null,
    page_start: 2,
    page_end: 3,
    page_numbers: [2, 3],
  }]), [{
    key: `${uploadReply.document_id}:S2`,
    id: 'S2',
    documentId: uploadReply.document_id,
    label: 'guide.pdf · pages 2, 3',
    href: null,
    snippet: null,
  }])
})

test('citation metadata keeps bounded snippets and rejects inline references without a matching source', () => {
  const sources = parseAnswerSources([{
    id: 'S1', document_id: uploadReply.document_id, title: 'Guide', source_type: 'pdf',
    filename: 'guide.pdf', snippet: 'Relevant passage', page_numbers: [4],
  }])
  assert.equal(sources[0].snippet, 'Relevant passage')
  assert.equal(parseAnswerSources([{
    id: 'S1', document_id: uploadReply.document_id, title: 'Guide', source_type: 'pdf',
    snippet: 'x'.repeat(900),
  }])[0].snippet.length, 700)
  assert.equal(resolveCitationId('S1', sources), 'S1')
  assert.equal(resolveCitationId('S2', sources), null)
})

test('SSE citations are delivered to chat state before streamed answer text', async () => {
  let sources
  let answer = ''
  const response = new Response([
    'event: sources\ndata: {"sources":[{"id":"S1","document_id":"5f4a9974-8f53-4fc3-a71d-3369557d03d1","title":"Guide","source_type":"pdf","filename":"guide.pdf","page_numbers":[4]}]}\n\n',
    'event: token\ndata: {"text":"Answer [S1]"}\n\n',
    'event: done\ndata: {"done":true}\n\n',
  ].join(''))
  await consumeAnswerStream(response, (text) => { answer += text }, (value) => { sources = value })

  assert.equal(answer, 'Answer [S1]')
  assert.equal(sources[0].filename, 'guide.pdf')
  assert.deepEqual(sources[0].page_numbers, [4])
})

test('SSE grounded-answer replacement safely overwrites partial tokens', async () => {
  const response = new Response([
    'event: token\ndata: {"text":"Uncited claim."}\n\n',
    'event: replace\ndata: {"answer":"I do not have enough evidence."}\n\n',
    'event: done\ndata: {"done":true}\n\n',
  ].join(''), { headers: { 'Content-Type': 'text/event-stream' } })
  let answer = ''
  await consumeAnswerStream(
    response,
    (text) => { answer += text },
    undefined,
    (replacement) => { answer = replacement },
  )
  assert.equal(answer, 'I do not have enough evidence.')
})
