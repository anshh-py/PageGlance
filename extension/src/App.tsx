import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import type { ThemePreference } from './main'
import { consumeAnswerStream, parseAskResponse } from './api'
import {
  buildConversationHistory,
  beginNewChat,
  createMessage,
  emptyConversation,
  MAX_STORED_CONVERSATIONS,
  MAX_STORED_MESSAGES,
  normalizePageUrl,
  restoreConversation,
  runSuggestion,
  SUGGESTIONS,
  type ChatMessage,
  type Conversation,
  type ConversationStore,
} from './chat'
import { clearConversationStore, loadConversationStore, saveConversationStore } from './chatStorage'
import { MarkdownAnswer } from './MarkdownAnswer'
import {
  EXTRACT_PAGE_MESSAGE,
  type AskRequest,
  type ExtractPageResponse,
  type PageContext,
} from './messages'
import './App.css'

const API_URL = (import.meta.env.VITE_PAGEGLANCE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const THEME_ORDER: ThemePreference[] = ['system', 'light', 'dark']
const STORAGE_DEBOUNCE_MS = 250
const HISTORY_MAX_MESSAGES = 20
const HISTORY_MAX_CHARS = 32_000

interface AppProps {
  initialTheme: ThemePreference
  fullPage?: boolean
}

type PageStatus = 'loading' | 'ready' | 'unsupported'

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong. Please try again.'
}

async function extractPage(tabId: number): Promise<PageContext> {
  try {
    await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] })
    const result = await chrome.tabs.sendMessage<ExtractPageResponse>(tabId, {
      type: EXTRACT_PAGE_MESSAGE,
    })
    if (!result.ok) throw new Error(result.error)
    if (!result.page.text) throw new Error('No readable page text was found on this tab.')
    return result.page
  } catch (error) {
    const detail = getErrorMessage(error)
    throw new Error(
      `PageGlance cannot read this tab. Chrome restricts extensions on internal pages and the Chrome Web Store. (${detail})`,
      { cause: error },
    )
  }
}

function updateMessage(
  conversation: Conversation,
  messageId: string,
  update: (message: ChatMessage) => ChatMessage | null,
): Conversation {
  return {
    ...conversation,
    updatedAt: Date.now(),
    messages: conversation.messages
      .map((message) => (message.id === messageId ? update(message) : message))
      .filter((message): message is ChatMessage => message !== null),
  }
}

function App({ initialTheme, fullPage = false }: AppProps) {
  const [theme, setTheme] = useState(initialTheme)
  const [conversation, setConversation] = useState<Conversation | null>(null)
  const [question, setQuestion] = useState('')
  const [pageUrl, setPageUrl] = useState('')
  const [pageTitle, setPageTitle] = useState('')
  const [pageStatus, setPageStatus] = useState<PageStatus>('loading')
  const [restoreReady, setRestoreReady] = useState(false)
  const [pageError, setPageError] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [copiedMessageId, setCopiedMessageId] = useState('')
  const conversationRef = useRef<Conversation | null>(null)
  const storeRef = useRef<ConversationStore>({ version: 1, conversations: {} })
  const pageUrlRef = useRef('')
  const pageContextRef = useRef<PageContext | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const requestInFlightRef = useRef(false)
  const mountedRef = useRef(false)
  const persistTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const persistQueueRef = useRef<Promise<void>>(Promise.resolve())

  const queueSave = useCallback(() => {
    const snapshot = JSON.parse(JSON.stringify(storeRef.current)) as ConversationStore
    persistQueueRef.current = persistQueueRef.current
      .catch(() => undefined)
      .then(() => saveConversationStore(snapshot))
      .catch(() => {
        if (mountedRef.current) setError('Could not save this conversation to extension storage.')
      })
  }, [])

  const commitConversation = useCallback((next: Conversation, immediate = false) => {
    conversationRef.current = next
    if (mountedRef.current) setConversation(next)
    if (next.pageUrl) {
      storeRef.current = {
        ...storeRef.current,
        version: 1,
        conversations: Object.fromEntries(
          Object.entries({ ...storeRef.current.conversations, [next.pageUrl]: next })
            .sort(([, left], [, right]) => right.updatedAt - left.updatedAt)
            .slice(0, MAX_STORED_CONVERSATIONS),
        ),
      }
      if (persistTimerRef.current) clearTimeout(persistTimerRef.current)
      if (immediate) queueSave()
      else {
        persistTimerRef.current = setTimeout(() => {
          persistTimerRef.current = null
          queueSave()
        }, STORAGE_DEBOUNCE_MS)
      }
    }
  }, [queueSave])

  useEffect(() => {
    mountedRef.current = true
    let active = true

    async function restore() {
      try {
        const sourceTabId = fullPage ? Number(new URLSearchParams(location.search).get('sourceTabId')) : NaN
        const tab = Number.isInteger(sourceTabId) && sourceTabId > 0
          ? await chrome.tabs.get(sourceTabId)
          : (await chrome.tabs.query({ active: true, currentWindow: true }))[0]
        if (!active) return
        const initialUrl = normalizePageUrl(tab?.url ?? '')
        pageUrlRef.current = initialUrl
        setPageUrl(initialUrl)
        setPageTitle(tab?.title ?? '')

        const store = await loadConversationStore()
        if (!active) return
        storeRef.current = store
        if (initialUrl) {
          const restored = restoreConversation(store, initialUrl, tab?.title ?? '')
          conversationRef.current = restored.conversation
          setConversation(restored.conversation)
          setQuestion(restored.conversation.draft)
          if (restored.interrupted) {
            storeRef.current = {
              ...store,
              conversations: {
                ...store.conversations,
                [initialUrl]: restored.conversation,
              },
            }
            queueSave()
          }
        }
        if (active) setRestoreReady(true)
        if (active) setPageStatus('loading')

        if (tab?.id === undefined || !initialUrl) {
          throw new Error('PageGlance could not identify the active webpage.')
        }
        const extracted = await extractPage(tab.id)
        if (!active) return
        const extractedUrl = normalizePageUrl(extracted.url)
        pageContextRef.current = extracted
        setPageTitle(extracted.title)

        if (extractedUrl && extractedUrl !== initialUrl) {
          const updatedStore = await loadConversationStore()
          if (!active) return
          storeRef.current = updatedStore
          pageUrlRef.current = extractedUrl
          setPageUrl(extractedUrl)
          const restored = restoreConversation(updatedStore, extractedUrl, extracted.title)
          conversationRef.current = restored.conversation
          setConversation(restored.conversation)
          setQuestion(restored.conversation.draft)
          if (restored.interrupted) {
            storeRef.current = {
              ...updatedStore,
              conversations: { ...updatedStore.conversations, [extractedUrl]: restored.conversation },
            }
            queueSave()
          }
        } else if (conversationRef.current && conversationRef.current.pageTitle !== extracted.title) {
          commitConversation({ ...conversationRef.current, pageTitle: extracted.title })
        }
        setPageStatus('ready')
      } catch (caught) {
        if (active) {
          setPageStatus('unsupported')
          setPageError(getErrorMessage(caught))
          setRestoreReady(true)
        }
      }
    }

    void restore()
    return () => {
      active = false
      mountedRef.current = false
      controllerRef.current?.abort()
      if (persistTimerRef.current) clearTimeout(persistTimerRef.current)
      const current = conversationRef.current
      if (current) {
        const interrupted = {
          ...current,
          updatedAt: Date.now(),
          messages: current.messages.map((message) =>
            message.role === 'assistant' && message.status === 'streaming'
              ? { ...message, status: 'interrupted' as const }
              : message,
          ),
        }
        conversationRef.current = interrupted
        if (interrupted.pageUrl) {
          storeRef.current = {
            ...storeRef.current,
            conversations: { ...storeRef.current.conversations, [interrupted.pageUrl]: interrupted },
          }
          queueSave()
        }
      }
    }
  }, [commitConversation, fullPage, queueSave])

  function handleThemeChange() {
    const nextTheme = THEME_ORDER[(THEME_ORDER.indexOf(theme) + 1) % THEME_ORDER.length]
    setTheme(nextTheme)
    document.documentElement.dataset.theme = nextTheme
    void chrome.storage.local.set({ pageGlanceTheme: nextTheme }).catch(() => {
      setError('Could not save the theme preference.')
    })
  }

  function handleDraftChange(value: string) {
    setQuestion(value)
    const current = conversationRef.current
    if (current) commitConversation({ ...current, draft: value })
  }

  async function submitQuestion(rawQuestion: string) {
    const trimmedQuestion = rawQuestion.trim()
    if (!trimmedQuestion || requestInFlightRef.current) return
    if (pageStatus !== 'ready' || !pageUrlRef.current) {
      setError(pageError || 'Open a supported webpage before asking PageGlance.')
      return
    }

    requestInFlightRef.current = true
    const controller = new AbortController()
    controllerRef.current = controller
    setLoading(true)
    setError('')
    setCopiedMessageId('')
    let partialAnswer = ''

    try {
      let page = pageContextRef.current
      if (!page) {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
        if (!tab?.id) throw new Error('Could not find the active tab.')
        page = await extractPage(tab.id)
        pageContextRef.current = page
      }
      if (normalizePageUrl(page.url) !== pageUrlRef.current) {
        throw new Error('The active page changed. Close and reopen PageGlance on the new page.')
      }

      const current = conversationRef.current ?? emptyConversation(pageUrlRef.current, page.title)
      const history = buildConversationHistory(
        current.messages,
        HISTORY_MAX_MESSAGES,
        HISTORY_MAX_CHARS,
      )
      const userMessage = createMessage('user', trimmedQuestion)
      const assistantMessage = createMessage('assistant', '', 'streaming')
      const nextConversation: Conversation = {
        ...current,
        pageUrl: pageUrlRef.current,
        pageTitle: page.title,
        updatedAt: userMessage.timestamp,
        draft: '',
        messages: [...current.messages, userMessage, assistantMessage].slice(-MAX_STORED_MESSAGES),
      }
      setQuestion('')
      commitConversation(nextConversation, true)
      const payload: AskRequest = {
        question: trimmedQuestion,
        page,
        history,
      }
      const response = await fetch(`${API_URL}/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(
          response.status === 422
            ? 'The backend rejected the question, page content, or conversation history.'
            : `PageGlance API returned ${response.status}${detail ? `: ${detail}` : ''}`,
        )
      }

      await consumeAnswerStream(response, (text) => {
        partialAnswer += text
        const latest = conversationRef.current
        if (!latest) return
        commitConversation(
          updateMessage(latest, assistantMessage.id, (message) => ({
            ...message,
            content: partialAnswer,
          })),
        )
      })
      if (partialAnswer) partialAnswer = parseAskResponse({ answer: partialAnswer }).answer
      const latest = conversationRef.current
      if (latest) {
        commitConversation(
          updateMessage(latest, assistantMessage.id, (message) => ({
            ...message,
            content: partialAnswer,
            status: 'complete',
          })),
          true,
        )
      }
    } catch (caught) {
      if (caught instanceof Error && caught.name === 'AbortError') {
        const latest = conversationRef.current
        if (latest) {
          commitConversation(
            updateMessage(latest, latest.messages.at(-1)?.id ?? '', (message) => ({
              ...message,
              status: 'interrupted',
            })),
            true,
          )
        }
        if (mountedRef.current) setError('Generation was interrupted. The partial response is saved.')
      } else {
        const latest = conversationRef.current
        const assistant = latest?.messages.at(-1)
        if (latest && assistant?.role === 'assistant' && assistant.status === 'streaming') {
          commitConversation(
            assistant.content
              ? updateMessage(latest, assistant.id, (message) => ({ ...message, status: 'error' }))
              : updateMessage(latest, assistant.id, () => null),
            true,
          )
        }
        if (mountedRef.current) {
          const message = getErrorMessage(caught)
          setError(
            message.includes('Failed to fetch') || message.includes('NetworkError')
              ? `Could not reach the PageGlance API at ${API_URL}. Check the backend and VITE_PAGEGLANCE_API_URL setting.`
              : message,
          )
        }
      }
    } finally {
      requestInFlightRef.current = false
      controllerRef.current = null
      if (mountedRef.current) setLoading(false)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void submitQuestion(question)
  }

  function handleQuestionKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  function handleNewChat() {
    const current = conversationRef.current
    if (!pageUrlRef.current) return
    if (
      (current?.messages.length || current?.draft) &&
      !window.confirm("Start a new chat and clear this page's conversation?")
    ) {
      return
    }
    const started = beginNewChat(storeRef.current, pageUrlRef.current, pageTitle)
    storeRef.current = started.store
    setQuestion('')
    setError('')
    commitConversation(started.conversation, true)
  }

  async function handleClearHistory() {
    const hasHistory = Object.values(storeRef.current.conversations).some(
      (item) => item.messages.length > 0 || Boolean(item.draft),
    )
    if (hasHistory && !window.confirm('Clear all saved PageGlance conversations?')) return
    try {
      if (persistTimerRef.current) {
        clearTimeout(persistTimerRef.current)
        persistTimerRef.current = null
      }
      await persistQueueRef.current.catch(() => undefined)
      await clearConversationStore()
      const cleared: ConversationStore = { version: 1, conversations: {} }
      storeRef.current = cleared
      if (pageUrlRef.current) {
        const blank = emptyConversation(pageUrlRef.current, pageTitle)
        storeRef.current.conversations[pageUrlRef.current] = blank
        conversationRef.current = blank
        setConversation(blank)
      } else {
        conversationRef.current = null
        setConversation(null)
      }
      setQuestion('')
      setError('')
    } catch {
      setError('Could not clear saved conversations from extension storage.')
    }
  }

  async function handleCopy(message: ChatMessage) {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopiedMessageId(message.id)
      window.setTimeout(() => setCopiedMessageId(''), 1400)
    } catch {
      setError('Could not copy the answer. Check clipboard permissions and try again.')
    }
  }

  async function openFullPage() {
    try {
      if (persistTimerRef.current) {
        clearTimeout(persistTimerRef.current)
        persistTimerRef.current = null
        queueSave()
      }
      await persistQueueRef.current
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
      if (!tab?.id) throw new Error('Could not identify the current webpage.')
      const fullPageUrl = new URL(chrome.runtime.getURL('chat.html'))
      fullPageUrl.searchParams.set('sourceTabId', String(tab.id))
      await chrome.tabs.create({ url: fullPageUrl.toString(), active: true })
    } catch {
      setError('Could not open PageGlance in a new tab. Your saved conversation is unchanged.')
    }
  }

  const nextTheme = THEME_ORDER[(THEME_ORDER.indexOf(theme) + 1) % THEME_ORDER.length]
  const isDarkTheme =
    theme === 'dark' ||
    (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)
  const messages = conversation?.messages ?? []
  const canAsk = pageStatus === 'ready' && !loading

  return (
    <main className={`popup${fullPage ? ' full-page' : ''}`}>
      <header className="brand">
        <img className="brand-mark" src="/icons/pageglance-48.png" alt="" />
        <div className="brand-copy">
          <p className="eyebrow">PAGEGLANCE</p>
          <h1>Ask this page</h1>
        </div>
        <button
          className="theme-toggle expand-toggle"
          type="button"
          onClick={() => void openFullPage()}
          aria-label="Open PageGlance in a full page tab"
          title="Open in a new tab"
          hidden={fullPage}
        >
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path d="M11 3h6v6M17 3l-7 7M8 5H4a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-4" />
          </svg>
        </button>
        <button
          className="theme-toggle"
          type="button"
          onClick={handleThemeChange}
          aria-label={`Theme: ${theme}. Switch to ${nextTheme}.`}
          title={`Theme: ${theme}. Click to change`}
        >
          {isDarkTheme ? (
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M16.7 12.5A7.1 7.1 0 0 1 7.5 3.3a7.2 7.2 0 1 0 9.2 9.2Z" />
            </svg>
          ) : (
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <circle cx="10" cy="10" r="3.4" />
              <path d="M10 1.8v2M10 16.2v2M18.2 10h-2M3.8 10h-2m14-5.8-1.4 1.4m-8.8 8.8-1.4 1.4m11.6 0-1.4-1.4m-8.8-8.8L4.2 4.2" />
            </svg>
          )}
        </button>
      </header>

      <section className="page-context" aria-label="Current webpage">
        <span className="page-context-icon" aria-hidden="true">↗</span>
        <div>
          <p className="page-title">{pageTitle || (pageStatus === 'loading' ? 'Reading page…' : 'No page available')}</p>
          <p className="page-url">{pageUrl}</p>
        </div>
        <div className={`page-state page-state-${pageStatus}`} aria-label={`Page status: ${pageStatus}`} />
      </section>

      <div className="chat-actions">
        <button type="button" className="quiet-button" onClick={handleNewChat} disabled={!pageUrl || pageStatus !== 'ready' || loading}>
          <span aria-hidden="true">＋</span> New chat
        </button>
        <button type="button" className="quiet-button" onClick={() => void handleClearHistory()} disabled={!restoreReady || loading}>
          Clear history
        </button>
      </div>

      {!restoreReady && <div className="restore-status" role="status">Restoring your chat…</div>}
      {pageStatus === 'unsupported' && (
        <p className="message error" role="alert">{pageError || 'Open a supported webpage to use suggestions and ask questions.'}</p>
      )}

      {restoreReady && !messages.length && (
        <section className="welcome-state" aria-label="Suggested actions">
          <div className="suggestions">
            {SUGGESTIONS.map((suggestion) => (
              <button
                className="suggestion-card"
                type="button"
                key={suggestion.id}
                disabled={!canAsk}
                onClick={() => runSuggestion(suggestion.id, (prompt) => void submitQuestion(prompt))}
              >
                <span><strong>{suggestion.title}</strong><small>{suggestion.description}</small></span>
                <span className="suggestion-arrow" aria-hidden="true">→</span>
              </button>
            ))}
          </div>
          {pageStatus === 'loading' && <p className="page-loading-note" role="status">Checking whether this page is available…</p>}
          {pageStatus === 'unsupported' && <p className="page-loading-note">Suggestions are unavailable for this tab.</p>}
        </section>
      )}

      {messages.length > 0 && (
        <section className="conversation" aria-label="Conversation">
          <ol className="message-list" aria-live="polite">
            {messages.map((message) => (
              <li className={`chat-message chat-message-${message.role}`} key={message.id}>
                <span className="message-role">{message.role === 'user' ? 'You' : 'PAGEGLANCE'}</span>
                {message.role === 'assistant' && message.content ? (
                  <div className="assistant-bubble">
                    <MarkdownAnswer text={message.content} />
                    <div className="message-tools">
                      {message.status === 'complete' && (
                        <button type="button" className="copy-button" onClick={() => void handleCopy(message)}>
                          {copiedMessageId === message.id ? 'Copied' : 'Copy answer'}
                        </button>
                      )}
                    </div>
                  </div>
                ) : message.role === 'assistant' && message.status === 'streaming' ? (
                  <div className="assistant-bubble typing-bubble" role="status">
                    <span className="typing-dots" aria-hidden="true"><i /><i /><i /></span>
                    <span>Thinking…</span>
                  </div>
                ) : (
                  <div className={message.role === 'user' ? 'user-bubble' : 'assistant-bubble'}>
                    {message.role === 'assistant' ? 'No response text was returned.' : message.content}
                  </div>
                )}
                {message.role === 'assistant' && message.status === 'interrupted' && (
                  <p className="interrupted-note" role="status">Generation was interrupted. This saved response may be incomplete.</p>
                )}
                {message.role === 'assistant' && message.status === 'error' && (
                  <p className="interrupted-note">Generation ended before completion.</p>
                )}
                {message.role === 'assistant' && message.status === 'streaming' && loading && (
                  <button type="button" className="cancel-button" onClick={() => controllerRef.current?.abort()}>
                    Cancel generation
                  </button>
                )}
              </li>
            ))}
          </ol>
        </section>
      )}

      {error && <p className="message error" role="alert">{error}</p>}

      <form className="composer" onSubmit={handleSubmit}>
        <label htmlFor="question">YOUR QUESTION</label>
        <textarea
          id="question"
          value={question}
          onChange={(event) => handleDraftChange(event.target.value)}
          onKeyDown={handleQuestionKeyDown}
          placeholder={pageStatus === 'ready' ? 'Ask a follow-up about this page…' : 'Waiting for a supported webpage…'}
          maxLength={4000}
          rows={3}
          disabled={!canAsk}
        />
        <button className="ask-button" type="submit" disabled={!canAsk || !question.trim()}>
          {loading ? <><span className="spinner" aria-hidden="true" /> Generating answer…</> : <>Send question <span className="button-arrow" aria-hidden="true">→</span></>}
        </button>
      </form>

      <footer className="privacy-note">
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path d="M3 6.5h10v7H3zM5 6.5V4a3 3 0 0 1 6 0v2.5" />
        </svg>
        <span>Page content and this page’s chat go to your configured backend</span>
      </footer>
    </main>
  )
}

export default App
