import type { AskHistoryMessage } from './messages'

export const CHAT_STORAGE_KEY = 'pageGlanceConversations'
export const CHAT_STORE_VERSION = 1
export const MAX_STORED_CONVERSATIONS = 20
export const MAX_STORED_MESSAGES = 30
export const MAX_STORED_MESSAGE_CHARS = 8_000

export type ChatRole = 'user' | 'assistant'
export type ChatMessageStatus = 'streaming' | 'complete' | 'interrupted' | 'error'

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  timestamp: number
  status?: ChatMessageStatus
}

export interface Conversation {
  pageUrl: string
  pageTitle: string
  updatedAt: number
  draft: string
  messages: ChatMessage[]
}

export interface ConversationStore {
  version: typeof CHAT_STORE_VERSION
  conversations: Record<string, Conversation>
}

export const SUGGESTIONS = [
  {
    id: 'summarize',
    title: 'Summarize this page',
    description: 'Get a concise overview of the main ideas.',
    question: 'Summarize this page.',
  },
  {
    id: 'takeaways',
    title: 'Extract key takeaways',
    description: 'Find the most useful points to remember.',
    question: 'Extract the key takeaways from this page.',
  },
  {
    id: 'questions',
    title: 'Generate questions and answers',
    description: 'Create a short study guide from this page.',
    question: 'Generate useful questions and concise answers based on this page.',
  },
  {
    id: 'ask',
    title: 'Ask a question',
    description: 'Start with a broad question about the page.',
    question: 'What should I know about this page?',
  },
] as const

export function runSuggestion(
  suggestionId: (typeof SUGGESTIONS)[number]['id'],
  submit: (question: string) => void,
): void {
  const suggestion = SUGGESTIONS.find((item) => item.id === suggestionId)
  if (suggestion) submit(suggestion.question)
}

export function normalizePageUrl(value: string): string {
  try {
    const url = new URL(value)
    url.hash = ''
    for (const key of [...url.searchParams.keys()]) {
      if (/^(utm_.+|fbclid|gclid)$/i.test(key)) url.searchParams.delete(key)
    }
    return url.href
  } catch {
    return ''
  }
}

export function emptyConversation(pageUrl: string, pageTitle: string): Conversation {
  return { pageUrl, pageTitle, updatedAt: Date.now(), draft: '', messages: [] }
}

export function beginNewChat(
  store: ConversationStore,
  pageUrl: string,
  pageTitle: string,
): { store: ConversationStore; conversation: Conversation } {
  const conversation = emptyConversation(pageUrl, pageTitle)
  const conversations = { ...store.conversations, [pageUrl]: conversation }
  return {
    conversation,
    store: {
      version: CHAT_STORE_VERSION,
      conversations: Object.fromEntries(
        Object.entries(conversations)
          .sort(([, left], [, right]) => right.updatedAt - left.updatedAt)
          .slice(0, MAX_STORED_CONVERSATIONS),
      ),
    },
  }
}

export function createMessage(
  role: ChatRole,
  content: string,
  status?: ChatMessageStatus,
): ChatMessage {
  return {
    id: globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`,
    role,
    content,
    timestamp: Date.now(),
    ...(role === 'assistant' && status ? { status } : {}),
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseMessage(value: unknown): ChatMessage | null {
  if (
    !isRecord(value) ||
    typeof value.id !== 'string' ||
    (value.role !== 'user' && value.role !== 'assistant') ||
    typeof value.content !== 'string' ||
    value.content.length > MAX_STORED_MESSAGE_CHARS ||
    typeof value.timestamp !== 'number' ||
    !Number.isFinite(value.timestamp)
  ) {
    return null
  }
  const statuses: ChatMessageStatus[] = ['streaming', 'complete', 'interrupted', 'error']
  const status = statuses.includes(value.status as ChatMessageStatus)
    ? (value.status as ChatMessageStatus)
    : value.role === 'assistant'
      ? 'complete'
      : undefined
  return {
    id: value.id.slice(0, 100),
    role: value.role,
    content: value.content,
    timestamp: value.timestamp,
    ...(value.role === 'assistant' ? { status } : {}),
  }
}

export function parseConversationStore(value: unknown): ConversationStore {
  if (!isRecord(value) || value.version !== CHAT_STORE_VERSION || !isRecord(value.conversations)) {
    return { version: CHAT_STORE_VERSION, conversations: {} }
  }
  const conversations: Record<string, Conversation> = {}
  for (const [rawUrl, candidate] of Object.entries(value.conversations)) {
    const pageUrl = normalizePageUrl(rawUrl)
    if (!pageUrl || !isRecord(candidate)) continue
    const pageTitle = typeof candidate.pageTitle === 'string' ? candidate.pageTitle.slice(0, 1000) : ''
    const updatedAt = typeof candidate.updatedAt === 'number' && Number.isFinite(candidate.updatedAt)
      ? candidate.updatedAt
      : Date.now()
    const draft = typeof candidate.draft === 'string' ? candidate.draft.slice(0, 4000) : ''
    const messages = Array.isArray(candidate.messages)
      ? candidate.messages.map(parseMessage).filter((message): message is ChatMessage => message !== null)
        .slice(-MAX_STORED_MESSAGES)
      : []
    conversations[pageUrl] = { pageUrl, pageTitle, updatedAt, draft, messages }
  }
  const recent = Object.entries(conversations)
    .sort(([, left], [, right]) => right.updatedAt - left.updatedAt)
    .slice(0, MAX_STORED_CONVERSATIONS)
  return { version: CHAT_STORE_VERSION, conversations: Object.fromEntries(recent) }
}

export function restoreConversation(store: ConversationStore, pageUrl: string, pageTitle: string) {
  const conversation = store.conversations[pageUrl]
  const restored = conversation
    ? { ...conversation, pageTitle: pageTitle || conversation.pageTitle, messages: [...conversation.messages] }
    : emptyConversation(pageUrl, pageTitle)
  let interrupted = false
  restored.messages = restored.messages.map((message) => {
    if (message.role === 'assistant' && message.status === 'streaming') {
      interrupted = true
      return { ...message, status: 'interrupted' }
    }
    return message
  })
  return { conversation: restored, interrupted }
}

export function buildConversationHistory(
  messages: ChatMessage[],
  maxMessages: number,
  maxChars: number,
): AskHistoryMessage[] {
  const selected: AskHistoryMessage[] = []
  let remaining = maxChars
  for (const message of [...messages].reverse()) {
    if (selected.length >= maxMessages) break
    const content = message.status === 'interrupted' || message.status === 'error'
      ? `[Earlier response ended before completion and may be incomplete.]\n${message.content}`
      : message.content
    if (content.length > remaining) continue
    selected.push({ role: message.role, content })
    remaining -= content.length
  }
  return selected.reverse()
}
