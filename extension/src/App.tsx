import { useCallback, useEffect, useReducer, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import type { ThemePreference } from './main'
import { consumeAnswerStream, parseAskResponse } from './api'
import {
  addUploadedDocument,
  assertImportableUrl,
  createScopedAskRequest,
  deleteDocument,
  importUrlDocument,
  removeUploadedDocument,
  uploadStateReducer,
  uploadDocument,
  type DocumentConversation,
  type UploadedDocument,
} from './documents'
import {
  loadDocumentConversations,
  loadUploadedDocuments,
  saveDocumentConversations,
  saveUploadedDocuments,
} from './documentStorage'
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
  type ConversationStore,
  type ChatMessage,
  type Conversation,
} from './chat'
import { clearConversationStore, loadConversationStore, saveConversationStore } from './chatStorage'
import { AnswerWithSources } from './SourceCitations'
import { MarkdownAnswer } from './MarkdownAnswer'
import { EXTRACT_PAGE_MESSAGE, type AskRequest, type ExtractPageResponse, type PageContext } from './messages'
import { browser } from './browser'
import { API_BASE_URL } from './config'
import { ensureInstallation } from './auth'
import { isRestrictedPageUrl, RESTRICTED_PAGE_MESSAGE } from './pageSupport'
import {
  classifyExtractionFailure,
  decideFallbackAction,
  ExtractionFailureCategory,
  FallbackAction,
  type ExtractionAttemptDescriptor,
} from './extractionRecovery'
import './App.css'
const THEME_ORDER: ThemePreference[] = ['system', 'light', 'dark']
const STORAGE_DEBOUNCE_MS = 250
const HISTORY_MAX_MESSAGES = 20
const HISTORY_MAX_CHARS = 32_000
const SELECTED_DOCUMENT_KEY = 'pageGlanceSelectedDocument'
type PageStatus = 'loading' | 'ready' | 'unsupported'

function formatFileSize(size: number): string {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(0)} KB`
  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

interface AppProps {
  initialTheme: ThemePreference
  fullPage?: boolean
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong. Please try again.'
}

interface ExtractionFailureInfo {
  category: ExtractionFailureCategory
  action: FallbackAction
  message: string
}

type ExtractOutcome =
  | { ok: true; page: PageContext }
  | { ok: false; failure: ExtractionFailureInfo }

/**
 * One active-tab extraction attempt, returning a structured outcome instead of
 * throwing. Empty content is detected explicitly (it must not be misread as a
 * restricted page); every other failure is classified with the M7.1 classifier
 * and mapped to its permitted fallback action.
 */
async function performExtraction(tabId: number): Promise<ExtractOutcome> {
  try {
    await browser.scripting.executeScript({ target: { tabId }, files: ['content.js'] })
    const result = await browser.tabs.sendMessage<ExtractPageResponse>(tabId, {
      type: EXTRACT_PAGE_MESSAGE,
    })
    if (!result.ok) throw new Error(result.error)
    if (!result.page.text) {
      return {
        ok: false,
        failure: {
          category: ExtractionFailureCategory.EMPTY_CONTENT,
          action: decideFallbackAction(ExtractionFailureCategory.EMPTY_CONTENT),
          message: 'No readable page text was found on this tab.',
        },
      }
    }
    return { ok: true, page: result.page }
  } catch (caught) {
    const descriptor: ExtractionAttemptDescriptor = { error: caught }
    const category = classifyExtractionFailure(descriptor)
    const action = decideFallbackAction(category)
    const message =
      category === ExtractionFailureCategory.RESTRICTED_PAGE
        ? RESTRICTED_PAGE_MESSAGE
        : getErrorMessage(caught)
    return { ok: false, failure: { category, action, message } }
  }
}

/**
 * Extract from the active tab with at most one automatic retry for transient
 * failures (those mapped by the decision logic to RETRY_ACTIVE_TAB_EXTRACTION).
 * Before retrying, the active tab is rechecked against the original tab so we
 * never retry a tab that changed. Restricted/access-denied/empty/navigation/
 * invalid-response failures, and backend ingestion failures, are never retried
 * as extraction failures. Duplicate extraction is prevented by capping retries
 * at one.
 */
async function extractWithRecovery(tabId: number): Promise<ExtractOutcome> {
  const first = await performExtraction(tabId)
  if (first.ok || first.failure.action !== FallbackAction.RETRY_ACTIVE_TAB_EXTRACTION) {
    return first
  }
  try {
    const activeTab = (await browser.tabs.query({ active: true, currentWindow: true }))[0]
    if (activeTab?.id !== tabId) {
      return {
        ok: false,
        failure: {
          category: ExtractionFailureCategory.NAVIGATION_OR_TAB_CHANGED,
          action: FallbackAction.STOP,
          message: 'The active tab changed while extracting. Reopen PageGlance on the correct page.',
        },
      }
    }
  } catch {
    return first
  }
  return performExtraction(tabId)
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
  const [pageError, setPageError] = useState('')
  const [extractionFailure, setExtractionFailure] = useState<ExtractionFailureInfo | null>(null)
  const [uploadedDocuments, setUploadedDocuments] = useState<UploadedDocument[]>([])
  const [selectedDocument, setSelectedDocument] = useState<UploadedDocument | null>(null)
  const [sourceMenuOpen, setSourceMenuOpen] = useState(false)
  const [uploadState, dispatchUpload] = useReducer(uploadStateReducer, { status: 'idle' })
  const [urlImportState, dispatchUrlImport] = useReducer(uploadStateReducer, { status: 'idle' })
  const [urlImportDraft, setUrlImportDraft] = useState('')
  const [restoreReady, setRestoreReady] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [copiedMessageId, setCopiedMessageId] = useState('')
  const conversationRef = useRef<Conversation | null>(null)
  const uploadedDocumentsRef = useRef<UploadedDocument[]>([])
  const documentConversationsRef = useRef<Record<string, DocumentConversation>>({})
  const storeRef = useRef<ConversationStore>({ version: 1, conversations: {} })
  const pageUrlRef = useRef('')
  const pageTitleRef = useRef('')
  const pageContextRef = useRef<PageContext | null>(null)
  const selectedDocumentRef = useRef<UploadedDocument | null>(null)
  const importedDocumentRef = useRef<{ document_id: string; title: string } | null>(null)
  const sourceSelectorButtonRef = useRef<HTMLButtonElement>(null)
  const documentFileInputRef = useRef<HTMLInputElement>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const requestInFlightRef = useRef(false)
  const questionInputRef = useRef<HTMLTextAreaElement>(null)
  const wasLoadingRef = useRef(false)
  const mountedRef = useRef(false)
  const persistTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const persistQueueRef = useRef<Promise<void>>(Promise.resolve())

  // Restore keyboard focus to the composer after each answer finishes or fails.
  // This works for both webpage chat and PDF/document chat because they share this composer.
  useEffect(() => {
    if (loading) {
      wasLoadingRef.current = true
      return
    }
    if (wasLoadingRef.current) {
      wasLoadingRef.current = false
      if (restoreReady) requestAnimationFrame(() => questionInputRef.current?.focus())
    }
  }, [loading, restoreReady])

  const queueSave = useCallback(() => {
    const documentSnapshot = JSON.parse(JSON.stringify(documentConversationsRef.current)) as Record<string, DocumentConversation>
    persistQueueRef.current = persistQueueRef.current
      .catch(() => undefined)
      .then(() => saveDocumentConversations(documentSnapshot))
      .catch(() => {
        if (mountedRef.current) setError('Could not save this conversation to extension storage.')
      })
  }, [])

  const queuePageSave = useCallback(() => {
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
    const activeDocument = selectedDocumentRef.current
    if (next.pageUrl && activeDocument) {
      documentConversationsRef.current = {
        ...documentConversationsRef.current,
        [activeDocument.document_id]: {
          updatedAt: next.updatedAt,
          draft: next.draft,
          messages: next.messages,
        },
      }
    } else if (next.pageUrl) {
      storeRef.current = {
        ...storeRef.current,
        version: 1,
        conversations: Object.fromEntries(
          Object.entries({ ...storeRef.current.conversations, [next.pageUrl]: next })
            .sort(([, left], [, right]) => right.updatedAt - left.updatedAt)
            .slice(0, MAX_STORED_CONVERSATIONS),
        ),
      }
    }
    if (next.pageUrl) {
      if (persistTimerRef.current) clearTimeout(persistTimerRef.current)
      const save = activeDocument ? queueSave : queuePageSave
      if (immediate) save()
      else {
        persistTimerRef.current = setTimeout(() => {
          persistTimerRef.current = null
          save()
        }, STORAGE_DEBOUNCE_MS)
      }
    }
  }, [queuePageSave, queueSave])

  const activateDocument = useCallback((document: UploadedDocument | null) => {
    selectedDocumentRef.current = document
    setSelectedDocument(document)
    const next = document
      ? (() => {
          const saved = documentConversationsRef.current[document.document_id]
          return saved
            ? { pageUrl: `document:${document.document_id}`, pageTitle: document.filename, updatedAt: saved.updatedAt, draft: saved.draft, messages: [...saved.messages] }
            : emptyConversation(`document:${document.document_id}`, document.filename)
        })()
      : pageUrlRef.current
        ? restoreConversation(storeRef.current, pageUrlRef.current, pageTitleRef.current).conversation
        : null
    conversationRef.current = next
    setConversation(next)
    setQuestion(next?.draft ?? '')
    setError('')
  }, [])

  useEffect(() => {
    mountedRef.current = true
    let active = true

    async function restore() {
      try {
        const sourceTabId = fullPage ? Number(new URLSearchParams(location.search).get('sourceTabId')) : NaN
        const tab = Number.isInteger(sourceTabId) && sourceTabId > 0
          ? await browser.tabs.get(sourceTabId)
          : (await browser.tabs.query({ active: true, currentWindow: true }))[0]
        if (!active) return
        const initialUrl = normalizePageUrl(tab?.url ?? '')
        pageUrlRef.current = initialUrl
        setPageUrl(initialUrl)
        setPageTitle(tab?.title ?? '')
        pageTitleRef.current = tab?.title ?? ''
        const [docs, documentConversations, store] = await Promise.all([
          loadUploadedDocuments(), loadDocumentConversations(), loadConversationStore(),
        ])
        if (!active) return
        uploadedDocumentsRef.current = docs
        setUploadedDocuments(docs)
        documentConversationsRef.current = documentConversations
        storeRef.current = store
        let restored = initialUrl ? restoreConversation(store, initialUrl, tab?.title ?? '') : null
        if (restored) {
          conversationRef.current = restored.conversation
          setConversation(restored.conversation)
          setQuestion(restored.conversation.draft)
          if (restored.interrupted) queuePageSave()
        }
        if (active) setRestoreReady(true)

        const requestedDocumentId = fullPage ? new URLSearchParams(location.search).get('documentId') : null
        const requestedDocument = requestedDocumentId
          ? docs.find((item) => item.document_id === requestedDocumentId) ?? null
          : null
        if (requestedDocument) activateDocument(requestedDocument)
        if (tab?.id === undefined || !initialUrl) throw new Error('PageGlance could not identify the active webpage.')
        if (isRestrictedPageUrl(initialUrl)) {
          setPageStatus('unsupported')
          setPageError(RESTRICTED_PAGE_MESSAGE)
          setRestoreReady(true)
          return
        }
        setPageStatus('loading')
        const outcome = await extractWithRecovery(tab.id)
        if (!active) return
        if (!outcome.ok) {
          setPageStatus('unsupported')
          setPageError(outcome.failure.message)
          setExtractionFailure(outcome.failure)
          if (outcome.failure.action === FallbackAction.OFFER_URL_IMPORT) {
            setUrlImportDraft((current) => current || pageUrlRef.current || '')
          }
          setRestoreReady(true)
          return
        }
        const extracted = outcome.page
        setExtractionFailure(null)
        pageContextRef.current = extracted
        setPageTitle(extracted.title)
        pageTitleRef.current = extracted.title
        const extractedUrl = normalizePageUrl(extracted.url)
        if (extractedUrl && extractedUrl !== initialUrl) {
          pageUrlRef.current = extractedUrl
          setPageUrl(extractedUrl)
          restored = restoreConversation(storeRef.current, extractedUrl, extracted.title)
          if (!selectedDocumentRef.current) {
            conversationRef.current = restored.conversation
            setConversation(restored.conversation)
            setQuestion(restored.conversation.draft)
          }
        } else if (!selectedDocumentRef.current && conversationRef.current) {
          const updated = { ...conversationRef.current, pageTitle: extracted.title }
          conversationRef.current = updated
          setConversation(updated)
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
        const interrupted = { ...current, updatedAt: Date.now(), messages: current.messages.map((message) =>
          message.role === 'assistant' && message.status === 'streaming' ? { ...message, status: 'interrupted' as const } : message) }
        conversationRef.current = interrupted
        if (selectedDocumentRef.current && interrupted.pageUrl) {
          const doc = selectedDocumentRef.current
          documentConversationsRef.current[doc.document_id] = { updatedAt: interrupted.updatedAt, draft: interrupted.draft, messages: interrupted.messages }
          queueSave()
        } else if (interrupted.pageUrl) {
          storeRef.current.conversations[interrupted.pageUrl] = interrupted
          queuePageSave()
        }
      }
    }
  }, [activateDocument, fullPage, queuePageSave, queueSave])

  function handleThemeChange() {
    const nextTheme = THEME_ORDER[(THEME_ORDER.indexOf(theme) + 1) % THEME_ORDER.length]
    setTheme(nextTheme)
    document.documentElement.dataset.theme = nextTheme
    void browser.storage.local.set({ pageGlanceTheme: nextTheme }).catch(() => {
      setError('Could not save the theme preference.')
    })
  }

  function handleDraftChange(value: string) {
    setQuestion(value)
    const current = conversationRef.current
    if (current) commitConversation({ ...current, draft: value })
  }

  async function handleDocumentSelection(documentId: string) {
    if (loading || requestInFlightRef.current) return
    if (persistTimerRef.current) {
      clearTimeout(persistTimerRef.current)
      persistTimerRef.current = null
      if (selectedDocumentRef.current) queueSave()
      else queuePageSave()
    }
    await persistQueueRef.current.catch(() => undefined)
    const document = documentId
      ? uploadedDocumentsRef.current.find((item) => item.document_id === documentId) ?? null
      : null
    if (documentId && !document) return
    activateDocument(document)
    try {
      if (document) await browser.storage.local.set({ [SELECTED_DOCUMENT_KEY]: document.document_id })
      else await browser.storage.local.remove(SELECTED_DOCUMENT_KEY)
    } catch {
      setError('Could not save the selected chat context to extension storage.')
    }
  }

  async function handleRemoveDocument(documentId: string) {
    if (loading || requestInFlightRef.current) return
    const currentlySelected = selectedDocumentRef.current?.document_id === documentId
    try {
      const token = await ensureInstallation({ apiUrl: API_BASE_URL })
      await deleteDocument(documentId, API_BASE_URL, fetch, token)
    } catch (caught) {
      setError(getErrorMessage(caught) || 'Could not remove this document from the backend.')
      return
    }
    const nextDocuments = removeUploadedDocument(uploadedDocumentsRef.current, documentId)
    try {
      await saveUploadedDocuments(nextDocuments)
      if (currentlySelected) await browser.storage.local.remove(SELECTED_DOCUMENT_KEY)
      uploadedDocumentsRef.current = nextDocuments
      setUploadedDocuments(nextDocuments)
      if (currentlySelected) activateDocument(null)
      setSourceMenuOpen(false)
      setError('')
    } catch {
      setError('The document was deleted, but its entry could not be cleared from the extension library.')
    }
  }

  async function handleUploadDocument(file: File) {
    if (uploadState.status === 'uploading') return
    dispatchUpload({ type: 'start', filename: file.name })
    try {
      const token = await ensureInstallation({ apiUrl: API_BASE_URL })
      const response = await uploadDocument(file, API_BASE_URL, fetch, token)
      if (persistTimerRef.current) {
        clearTimeout(persistTimerRef.current)
        persistTimerRef.current = null
        if (selectedDocumentRef.current) queueSave()
        else queuePageSave()
      }
      await persistQueueRef.current.catch(() => undefined)
      const nextDocuments = addUploadedDocument(uploadedDocumentsRef.current, {
        ...response,
        file_size_bytes: file.size,
      })
      await saveUploadedDocuments(nextDocuments)
      const uploaded = nextDocuments.find((item) => item.document_id === response.document_id)
      if (!uploaded) throw new Error('The uploaded document could not be saved in this browser.')
      uploadedDocumentsRef.current = nextDocuments
      setUploadedDocuments(nextDocuments)
      activateDocument(uploaded)
      setExtractionFailure(null)
      setPageStatus('ready')
      await browser.storage.local.set({ [SELECTED_DOCUMENT_KEY]: uploaded.document_id })
      dispatchUpload({
        type: 'success',
        message: `${uploaded.filename} ${uploaded.duplicate ? 'is already available' : 'uploaded'} · ${uploaded.chunk_count} chunks${uploaded.page_count ? ` · ${uploaded.page_count} pages` : ''}.`,
      })
    } catch (caught) {
      dispatchUpload({ type: 'error', message: getErrorMessage(caught) })
    }
  }

  async function handleImportUrl() {
    if (urlImportState.status === 'uploading') return
    const url = urlImportDraft.trim()
    try {
      assertImportableUrl(url)
    } catch (caught) {
      dispatchUrlImport({ type: 'error', message: getErrorMessage(caught) })
      return
    }
    dispatchUrlImport({ type: 'start', filename: 'URL' })
    try {
      const token = await ensureInstallation({ apiUrl: API_BASE_URL })
      const result = await importUrlDocument(url, API_BASE_URL, fetch, token)
      if (!result.indexed) {
        dispatchUrlImport({
          type: 'error',
          message:
            result.fallback_code === 'empty_page_text'
              ? 'That page has no readable text, so nothing was indexed.'
              : 'That page could not be indexed by the backend.',
        })
        return
      }
      dispatchUrlImport({
        type: 'success',
        message: `Indexed “${result.title}” · ${result.chunk_count} chunks.`,
      })
      importedDocumentRef.current = { document_id: result.document_id, title: result.title }
      setExtractionFailure(null)
      setPageStatus('ready')
    } catch (caught) {
      dispatchUrlImport({ type: 'error', message: getErrorMessage(caught) })
    }
  }

  async function handleReextract() {
    if (loading || requestInFlightRef.current || pageStatus === 'loading' || uploadState.status === 'uploading' || urlImportState.status === 'uploading') return
    if (selectedDocument) return
    try {
      const sourceTabId = fullPage ? Number(new URLSearchParams(location.search).get('sourceTabId')) : NaN
      const tab = Number.isInteger(sourceTabId) && sourceTabId > 0
        ? await browser.tabs.get(sourceTabId)
        : (await browser.tabs.query({ active: true, currentWindow: true }))[0]
      if (tab?.id === undefined) throw new Error('Could not find the active tab.')
      setPageStatus('loading')
      const outcome = await extractWithRecovery(tab.id)
      if (!outcome.ok) {
        setPageStatus('unsupported')
        setPageError(outcome.failure.message)
        setExtractionFailure(outcome.failure)
        if (outcome.failure.action === FallbackAction.OFFER_URL_IMPORT) {
          setUrlImportDraft((current) => current || pageUrlRef.current || '')
        }
        return
      }
      const extracted = outcome.page
      setExtractionFailure(null)
      pageContextRef.current = extracted
      setPageTitle(extracted.title)
      pageTitleRef.current = extracted.title
      const extractedUrl = normalizePageUrl(extracted.url)
      if (extractedUrl) {
        pageUrlRef.current = extractedUrl
        setPageUrl(extractedUrl)
      }
      setPageStatus('ready')
    } catch (caught) {
      setPageStatus('unsupported')
      setPageError(getErrorMessage(caught))
    }
  }

  async function submitQuestion(rawQuestion: string) {
    const trimmedQuestion = rawQuestion.trim()
    if (!trimmedQuestion || requestInFlightRef.current) return
    const activeDocument = selectedDocumentRef.current
    const importedDoc = importedDocumentRef.current
    if (!activeDocument && !importedDoc && (pageStatus !== 'ready' || !pageUrlRef.current)) {
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
      const askDocument = activeDocument ?? importedDoc
      let page: PageContext | null = null
      if (!askDocument) {
        page = pageContextRef.current
        if (!page) {
          const sourceTabId = fullPage ? Number(new URLSearchParams(location.search).get('sourceTabId')) : NaN
          const tab = Number.isInteger(sourceTabId) && sourceTabId > 0
            ? await browser.tabs.get(sourceTabId)
            : (await browser.tabs.query({ active: true, currentWindow: true }))[0]
          if (tab?.id === undefined) throw new Error('Could not find the active tab.')
          const outcome = await extractWithRecovery(tab.id)
          if (!outcome.ok) {
            setExtractionFailure(outcome.failure)
            throw new Error(outcome.failure.message)
          }
          page = outcome.page
          pageContextRef.current = page
        }
        if (normalizePageUrl(page.url) !== pageUrlRef.current) {
          throw new Error('The active page changed. Close and reopen PageGlance on the new page.')
        }
      }
      const conversationKey = askDocument
        ? `document:${askDocument.document_id}`
        : pageUrlRef.current
      const conversationTitle = askDocument
        ? (activeDocument?.filename ?? importedDoc?.title ?? '')
        : page?.title ?? pageTitleRef.current
      const current = conversationRef.current ?? emptyConversation(conversationKey, conversationTitle)
      const history = buildConversationHistory(
        current.messages,
        HISTORY_MAX_MESSAGES,
        HISTORY_MAX_CHARS,
      )
      const userMessage = createMessage('user', trimmedQuestion)
      const assistantMessage = createMessage('assistant', '', 'streaming')
      const nextConversation: Conversation = {
        ...current,
        pageUrl: conversationKey,
        pageTitle: conversationTitle,
        updatedAt: userMessage.timestamp,
        draft: '',
        messages: [...current.messages, userMessage, assistantMessage].slice(-MAX_STORED_MESSAGES),
      }
      setQuestion('')
      commitConversation(nextConversation, true)
      const payload: AskRequest = createScopedAskRequest(
        trimmedQuestion,
        page,
        askDocument,
        history,
      )
      const token = await ensureInstallation({ apiUrl: API_BASE_URL })
      const response = await fetch(`${API_BASE_URL}/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
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
      }, (sources) => {
        const latest = conversationRef.current
        if (!latest) return
        commitConversation(updateMessage(latest, assistantMessage.id, (message) => ({
          ...message,
          sources,
        })))
      }, (answer) => {
        partialAnswer = answer
        const latest = conversationRef.current
        if (!latest) return
        commitConversation(updateMessage(latest, assistantMessage.id, (message) => ({
          ...message,
          content: answer,
        })))
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
              ? `Could not reach the PageGlance API at ${API_BASE_URL}. Check the backend and VITE_PAGEGLANCE_API_URL setting.`
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
    const activeDocument = selectedDocumentRef.current
    if (activeDocument) {
      if ((current?.messages.length || current?.draft) && !window.confirm(`Start a new chat about ${activeDocument.filename}?`)) return
      const next = emptyConversation(`document:${activeDocument.document_id}`, activeDocument.filename)
      setQuestion('')
      setError('')
      commitConversation(next, true)
      return
    }
    if (!pageUrlRef.current) return
    if (
      (current?.messages.length || current?.draft) &&
      !window.confirm("Start a new chat and clear this page's conversation?")
    ) {
      return
    }
    const started = beginNewChat(storeRef.current, pageUrlRef.current, pageTitleRef.current)
    storeRef.current = started.store
    setQuestion('')
    setError('')
    commitConversation(started.conversation, true)
  }

  async function handleClearHistory() {
    const activeDocument = selectedDocumentRef.current
    if (activeDocument) {
      const saved = documentConversationsRef.current[activeDocument.document_id]
      if ((saved?.messages.length || saved?.draft) &&
        !window.confirm(`Clear the saved chat for ${activeDocument.filename}?`)) return
    } else {
      const hasHistory = Object.values(storeRef.current.conversations).some(
        (item) => item.messages.length > 0 || Boolean(item.draft),
      )
      if (hasHistory && !window.confirm('Clear all saved PageGlance conversations?')) return
    }
    try {
      if (persistTimerRef.current) {
        clearTimeout(persistTimerRef.current)
        persistTimerRef.current = null
      }
      await persistQueueRef.current.catch(() => undefined)
      if (activeDocument) {
        const nextDocuments = { ...documentConversationsRef.current }
        delete nextDocuments[activeDocument.document_id]
        documentConversationsRef.current = nextDocuments
        const blank = emptyConversation(`document:${activeDocument.document_id}`, activeDocument.filename)
        conversationRef.current = blank
        setConversation(blank)
        await saveDocumentConversations(nextDocuments)
      } else {
        await clearConversationStore()
        storeRef.current = { version: 1, conversations: {} }
        if (pageUrlRef.current) {
          const blank = emptyConversation(pageUrlRef.current, pageTitleRef.current)
          storeRef.current.conversations[pageUrlRef.current] = blank
          conversationRef.current = blank
          setConversation(blank)
        } else {
          conversationRef.current = null
          setConversation(null)
        }
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
        if (selectedDocumentRef.current) queueSave()
        else queuePageSave()
      }
      await persistQueueRef.current
      const fullPageUrl = new URL(browser.runtime.getURL('chat.html'))
      const sourceTabId = fullPage ? Number(new URLSearchParams(location.search).get('sourceTabId')) : NaN
      const sourceTab = Number.isInteger(sourceTabId) && sourceTabId > 0
        ? await browser.tabs.get(sourceTabId)
        : (await browser.tabs.query({ active: true, currentWindow: true }))[0]
      if (!sourceTab?.id) throw new Error('Could not identify the current webpage.')
      fullPageUrl.searchParams.set('sourceTabId', String(sourceTab.id))
      if (selectedDocumentRef.current) {
        fullPageUrl.searchParams.set('documentId', selectedDocumentRef.current.document_id)
      }
      await browser.tabs.create({ url: fullPageUrl.toString(), active: true })
    } catch {
      setError('Could not open PageGlance in a new tab. Your saved conversation is unchanged.')
    }
  }

  const nextTheme = THEME_ORDER[(THEME_ORDER.indexOf(theme) + 1) % THEME_ORDER.length]
  const isDarkTheme =
    theme === 'dark' ||
    (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)
  const messages = conversation?.messages ?? []
  const canAsk = restoreReady && (selectedDocument ? true : pageStatus === 'ready') && !loading
  const showingUrlImportRecovery = extractionFailure?.action === FallbackAction.OFFER_URL_IMPORT
  const uploadControl = (
    <button
      type="button"
      className={`document-upload-button${uploadState.status === 'uploading' ? ' is-uploading' : ''}`}
      onClick={() => documentFileInputRef.current?.click()}
      disabled={uploadState.status === 'uploading' || loading || !restoreReady}
      aria-label={uploadState.status === 'uploading' ? 'Uploading document' : 'Upload TXT or PDF'}
      title="Upload TXT or PDF"
    >
      {uploadState.status === 'uploading' ? (
        <span className="document-upload-spinner" aria-hidden="true" />
      ) : (
        <svg viewBox="0 0 20 20" aria-hidden="true">
          <path d="M5.5 2.5h6l3 3v12h-9z" />
          <path d="M11.5 2.5v3h3M7.5 12h5M10 9.5v5" />
        </svg>
      )}
      <span className="upload-button-label">{uploadState.status === 'uploading' ? 'Uploading' : '📤 Upload'}</span>
    </button>
  )
  function handleSourceMenuKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') {
      event.preventDefault()
      setSourceMenuOpen(false)
      sourceSelectorButtonRef.current?.focus()
    }
  }

  return (
    <main className={`popup${fullPage ? ' full-page' : ''}`}>
      <header className="brand">
        <img className="brand-mark" src="/icons/pageglance-48.png" alt="" />
        <div className="brand-copy">
          <p className="eyebrow">PAGEGLANCE</p>
          <h1>{selectedDocument ? 'Document chat' : 'Ask this page'}</h1>
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

      <input
        ref={documentFileInputRef}
        className="document-file-input"
        type="file"
        accept=".txt,.pdf,text/plain,application/pdf"
        aria-label="Choose a TXT or PDF document"
        tabIndex={-1}
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) void handleUploadDocument(file)
          event.target.value = ''
        }}
        disabled={uploadState.status === 'uploading' || loading}
      />
      <div className="context-row">
      {uploadedDocuments.length > 0 && (
        <div className="source-selector">
          <button
            ref={sourceSelectorButtonRef}
            type="button"
            className="context-source-select"
            aria-label={`Chat source: ${selectedDocument?.filename ?? 'Current webpage'}`}
            aria-expanded={sourceMenuOpen}
            aria-controls="source-menu"
            onClick={() => setSourceMenuOpen((open) => !open)}
            disabled={!restoreReady || loading}
          >
            <span>{selectedDocument?.filename ?? 'Current webpage'}</span>
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" /></svg>
          </button>
          {sourceMenuOpen && (
            <div
              className="source-menu"
              id="source-menu"
              role="group"
              aria-label="Choose chat source"
              onKeyDown={handleSourceMenuKeyDown}
            >
              <button
                type="button"
                className="source-option"
                aria-pressed={!selectedDocument}
                onClick={() => {
                  setSourceMenuOpen(false)
                  void handleDocumentSelection('')
                }}
              >
                <span>Current webpage</span>
                {!selectedDocument && <span className="source-option-check" aria-hidden="true">✓</span>}
              </button>
              {uploadedDocuments.map((document) => (
                <div className="source-menu-item" key={document.document_id}>
                  <button
                    type="button"
                    className="source-option"
                    aria-pressed={selectedDocument?.document_id === document.document_id}
                    onClick={() => {
                      setSourceMenuOpen(false)
                      void handleDocumentSelection(document.document_id)
                    }}
                  >
                    <span className="source-option-type" aria-hidden="true">{document.source_type.toUpperCase()}</span>
                    <span className="source-option-name" title={document.filename}>{document.filename}</span>
                    {selectedDocument?.document_id === document.document_id && (
                      <span className="source-option-check" aria-hidden="true">✓</span>
                    )}
                  </button>
                  <button
                    type="button"
                    className="remove-document-button"
                    aria-label={`Remove ${document.filename} from uploaded documents`}
                    title={`Remove ${document.filename}`}
                    disabled={!restoreReady || loading}
                    onClick={() => void handleRemoveDocument(document.document_id)}
                  >
                    <svg viewBox="0 0 20 20" aria-hidden="true">
                      <path d="M4 6h12M8 6V4h4v2m-6 0 .7 10h6.6L14 6M8.5 9v4m3-4v4" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
        {selectedDocument ? (
          <section className="document-context" aria-label="Selected document details">
            <span className="document-context-icon" aria-hidden="true">&#9635;</span>
            <div className="document-context-copy">
              <p className="document-title">{selectedDocument.filename}</p>
              <p className="document-metadata">
                {selectedDocument.source_type.toUpperCase()} | Ready | {selectedDocument.chunk_count} indexed chunks
                {selectedDocument.page_count ? ` | ${selectedDocument.page_count} pages` : ''}
                {selectedDocument.file_size_bytes ? ` | ${formatFileSize(selectedDocument.file_size_bytes)}` : ''}
              </p>
            </div>
          </section>
        ) : (
          <section className={`page-context page-state-context-${pageStatus}`} aria-label="Current webpage">
            <span className="page-context-icon" aria-hidden="true">↗</span>
            <div>
              <p className="page-title">{pageTitle || (pageStatus === 'loading' ? 'Reading page…' : 'No page available')}</p>
              <p className="page-url" title={pageUrl}>{pageUrl || pageError}</p>
            </div>
            <div className={`page-state page-state-${pageStatus}`} aria-label={`Page status: ${pageStatus}`} />
          </section>
        )}
      </div>

      <div className="ingest-bar" aria-label="Add content">
        {!showingUrlImportRecovery && (
          <form
            className="url-import-form"
            onSubmit={(event) => {
              event.preventDefault()
              void handleImportUrl()
            }}
          >
            <label htmlFor="ingest-url-input">Import by URL</label>
            <div className="url-import-row">
              <input
                id="ingest-url-input"
                type="text"
                inputMode="url"
                value={urlImportDraft}
                onChange={(event) => setUrlImportDraft(event.target.value)}
                placeholder="https://example.com/article"
                disabled={urlImportState.status === 'uploading'}
              />
              <button
                type="submit"
                className="url-import-submit"
                disabled={urlImportState.status === 'uploading' || !urlImportDraft.trim()}
              >
                {urlImportState.status === 'uploading' ? 'Importing…' : 'Import URL'}
              </button>
            </div>
            {urlImportState.status === 'success' && (
              <p className="document-feedback" role="status">{urlImportState.message}</p>
            )}
            {urlImportState.status === 'error' && (
              <p className="message error document-feedback" role="alert">{urlImportState.message}</p>
            )}
          </form>
        )}
        <button
          type="button"
          className="quiet-button reextract-button"
          onClick={() => void handleReextract()}
          disabled={loading || !restoreReady || pageStatus === 'loading' || !!selectedDocument || uploadState.status === 'uploading' || urlImportState.status === 'uploading'}
          title={selectedDocument ? 'Switch to Current webpage to re-extract it' : 'Re-extract the current page'}
        >
          <span aria-hidden="true">↻</span> Re-extract page
        </button>
        {uploadControl}
      </div>
      {uploadState.status === 'uploading' && <p className="document-feedback" role="status">Uploading {uploadState.filename}...</p>}
      {uploadState.status === 'success' && <p className="document-feedback" role="status">{uploadState.message}</p>}
      {uploadState.status === 'error' && <p className="message error document-feedback" role="alert">{uploadState.message}</p>}

      <div className="chat-actions">
        <button type="button" className="quiet-button" onClick={handleNewChat} disabled={(!selectedDocument && (!pageUrl || pageStatus !== 'ready')) || loading}>
          <span aria-hidden="true">＋</span> New chat
        </button>
        <button type="button" className="quiet-button" onClick={() => void handleClearHistory()} disabled={!restoreReady || loading}>
          Clear history
        </button>
      </div>

      {!restoreReady && <div className="restore-status" role="status">Restoring your chat…</div>}
      {!selectedDocument && pageStatus === 'unsupported' && (
        extractionFailure ? (
          <section className="unsupported-state" aria-label="Page could not be read">
            <span className="unsupported-icon" aria-hidden="true">⚠️</span>
            <div>
              <h2 className="unsupported-title">Couldn’t read this page</h2>
              <p className="unsupported-copy">{extractionFailure.message}</p>
              {extractionFailure.action === FallbackAction.OFFER_URL_IMPORT && (
                <form
                  className="url-import-form"
                  onSubmit={(event) => {
                    event.preventDefault()
                    void handleImportUrl()
                  }}
                >
                  <label htmlFor="url-import-input">Import this page by URL</label>
                  <div className="url-import-row">
                    <input
                      id="url-import-input"
                      type="text"
                      inputMode="url"
                      value={urlImportDraft}
                      onChange={(event) => setUrlImportDraft(event.target.value)}
                      placeholder="https://example.com/article"
                      disabled={urlImportState.status === 'uploading'}
                    />
                    <button
                      type="submit"
                      className="url-import-submit"
                      disabled={urlImportState.status === 'uploading' || !urlImportDraft.trim()}
                    >
                      {urlImportState.status === 'uploading' ? 'Importing…' : 'Import URL'}
                    </button>
                  </div>
                  {urlImportState.status === 'success' && (
                    <p className="document-feedback" role="status">{urlImportState.message}</p>
                  )}
                  {urlImportState.status === 'error' && (
                    <p className="message error document-feedback" role="alert">{urlImportState.message}</p>
                  )}
                  <p className="unsupported-instruction">
                    Or use the <strong>Upload</strong> button to add a TXT/PDF transcript for this page.
                  </p>
                </form>
              )}
              {extractionFailure.action === FallbackAction.OFFER_MANUAL_UPLOAD && (
                <div className="manual-upload-recovery">
                  <p className="unsupported-copy">
                    Use the <strong>Upload</strong> control to add this page as a TXT or PDF document
                    and ask about it right away.
                  </p>
                  <button
                    type="button"
                    className="document-upload-button"
                    onClick={() => documentFileInputRef.current?.click()}
                    disabled={uploadState.status === 'uploading' || loading || !restoreReady}
                  >
                    {uploadState.status === 'uploading' ? (
                      <span className="document-upload-spinner" aria-hidden="true" />
                    ) : (
                      <svg viewBox="0 0 20 20" aria-hidden="true">
                        <path d="M5.5 2.5h6l3 3v12h-9z" />
                        <path d="M11.5 2.5v3h3M7.5 12h5M10 9.5v5" />
                      </svg>
                    )}
                    <span className="upload-button-label">
                      {uploadState.status === 'uploading' ? 'Uploading' : '📤 Upload'}
                    </span>
                  </button>
                </div>
              )}
              {extractionFailure.action === FallbackAction.RETRY_ACTIVE_TAB_EXTRACTION && (
                <p className="unsupported-instruction">
                  Automatic retry didn’t recover this page. Try <strong>reopening</strong> it, then ask again.
                </p>
              )}
              {extractionFailure.action === FallbackAction.STOP && (
                <p className="unsupported-instruction">
                  Reopen the page in a regular <strong>http</strong> or <strong>https</strong> tab and try again.
                </p>
              )}
              {extractionFailure.action === FallbackAction.USE_ALTERNATE_EXTRACTION && (
                <p className="unsupported-instruction">
                  This page can’t be read directly. Try opening it in a regular <strong>http</strong> or <strong>https</strong> tab, or use the <strong>Upload</strong> button to add a TXT/PDF transcript.
                </p>
              )}
              {extractionFailure.action === FallbackAction.RETRY_INGESTION && (
                <p className="unsupported-instruction">
                  The PageGlance backend couldn’t be reached. Make sure the backend is running, then reopen PageGlance and try again.
                </p>
              )}
            </div>
          </section>
        ) : (
          <section className="unsupported-state" aria-label="Unsupported page">
            <span className="unsupported-icon" aria-hidden="true">🔒</span>
            <div>
              <h2 className="unsupported-title">This page can’t be analyzed</h2>
              <p className="unsupported-copy">
                Browser security restrictions prevent PageGlance from reading this page.
              </p>
              <p className="unsupported-instruction">
                Open a regular <strong>http</strong> or <strong>https</strong> webpage, then try again.
              </p>
            </div>
          </section>
        )
      )}
      {selectedDocument && restoreReady && !messages.length && (
        <p className="document-welcome">Ask a question about {selectedDocument.filename}.</p>
      )}
      {!selectedDocument && restoreReady && !messages.length && (
        <section className="welcome-state" aria-label="Suggested actions">
          <div className="suggestions">
            {SUGGESTIONS.map((suggestion) => (
              <button className="suggestion-card" type="button" key={suggestion.id} disabled={!canAsk}
                onClick={() => runSuggestion(suggestion.id, (prompt) => void submitQuestion(prompt))}>
                <span><strong>{suggestion.title}</strong><small>{suggestion.description}</small></span>
                <span className="suggestion-arrow" aria-hidden="true">→</span>
              </button>
            ))}
          </div>
          {pageStatus === 'loading' && <p className="page-loading-note" role="status">Checking whether this page is available…</p>}
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
                    {message.sources?.length ? (
                      <AnswerWithSources text={message.content} sources={message.sources} />
                    ) : <MarkdownAnswer text={message.content} />}
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
        <label htmlFor="question">{selectedDocument ? 'QUESTION ABOUT DOCUMENT' : 'YOUR QUESTION'}</label>
        <textarea
          ref={questionInputRef}
          id="question"
          value={question}
          onChange={(event) => handleDraftChange(event.target.value)}
          onKeyDown={handleQuestionKeyDown}
          placeholder={selectedDocument ? 'Ask a question about this document...' : pageStatus === 'ready' ? 'Ask a follow-up about this page…' : 'Waiting for a supported webpage…'}
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
        <span>{selectedDocument ? 'Questions and selected document content go to your configured backend' : 'Page content and this page’s chat go to your configured backend'}</span>
      </footer>
    </main>
  )
}

export default App
