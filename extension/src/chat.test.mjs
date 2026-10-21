import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  beginNewChat,
  buildConversationHistory,
  createMessage,
  normalizePageUrl,
  parseConversationStore,
  restoreConversation,
  runSuggestion,
  SUGGESTIONS,
} from './chat.ts'
import { clearConversationStore, loadConversationStore, saveConversationStore } from './chatStorage.ts'

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
