import type { AnswerSource, AskHistoryMessage, AskRequest, PageContext } from './messages'
import { parseChatMessage, type ChatMessage } from './chat.ts'

export const DOCUMENT_LIBRARY_KEY = 'pageGlanceUploadedDocuments'
export const DOCUMENT_CHAT_STORE_KEY = 'pageGlanceDocumentConversations'
export const MAX_SAVED_DOCUMENTS = 30
export const MAX_FILE_BYTES = 25 * 1024 * 1024
const DOCUMENT_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

export interface UploadedDocument {
  document_id: string
  filename: string
  source_type: 'text' | 'pdf'
  content_hash: string
  chunk_count: number
  duplicate: boolean
  page_count: number | null
  uploaded_at: number
  file_size_bytes?: number
}

export type UploadState =
  | { status: 'idle' }
  | { status: 'uploading'; filename: string }
  | { status: 'success'; message: string }
  | { status: 'error'; message: string }

export type UploadAction =
  | { type: 'start'; filename: string }
  | { type: 'success'; message: string }
  | { type: 'error'; message: string }

export function uploadStateReducer(_state: UploadState, action: UploadAction): UploadState {
  if (action.type === 'start') return { status: 'uploading', filename: action.filename }
  if (action.type === 'success') return { status: 'success', message: action.message }
  return { status: 'error', message: action.message }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function isUploadedDocument(value: unknown): value is UploadedDocument {
  return (
    isRecord(value) &&
    typeof value.document_id === 'string' && DOCUMENT_ID.test(value.document_id) &&
    typeof value.filename === 'string' && value.filename.length > 0 && value.filename.length <= 255 &&
    (value.source_type === 'pdf' || value.source_type === 'text') &&
    typeof value.content_hash === 'string' && /^[0-9a-f]{64}$/i.test(value.content_hash) &&
    Number.isInteger(value.chunk_count) && (value.chunk_count as number) >= 0 &&
    typeof value.duplicate === 'boolean' &&
    (value.page_count === null || (Number.isInteger(value.page_count) && (value.page_count as number) >= 0)) &&
    typeof value.uploaded_at === 'number' && Number.isFinite(value.uploaded_at) &&
    (!('file_size_bytes' in value) ||
      (Number.isInteger(value.file_size_bytes) && (value.file_size_bytes as number) >= 0))
  )
}

export function parseUploadedDocuments(value: unknown): UploadedDocument[] {
  if (!Array.isArray(value)) return []
  const valid = value.filter(isUploadedDocument)
  const unique = new Map<string, UploadedDocument>()
  for (const document of valid) unique.set(document.document_id, document)
  return [...unique.values()]
    .sort((left, right) => right.uploaded_at - left.uploaded_at)
    .slice(0, MAX_SAVED_DOCUMENTS)
}

export function selectInitialDocument(
  documents: UploadedDocument[],
  preferredDocumentId: string | null,
): UploadedDocument | null {
  return documents.find((document) => document.document_id === preferredDocumentId) ?? documents[0] ?? null
}

export function addUploadedDocument(
  documents: UploadedDocument[],
  response: Omit<UploadedDocument, 'uploaded_at'>,
  uploadedAt = Date.now(),
): UploadedDocument[] {
  return parseUploadedDocuments([
    { ...response, uploaded_at: uploadedAt },
    ...documents.filter((item) => item.document_id !== response.document_id),
  ])
}

export function removeUploadedDocument(
  documents: UploadedDocument[],
  documentId: string,
): UploadedDocument[] {
  return documents.filter((document) => document.document_id !== documentId)
}

export async function uploadDocument(
  file: File,
  apiUrl: string,
  fetcher: typeof fetch = fetch,
): Promise<Omit<UploadedDocument, 'uploaded_at'>> {
  const extension = file.name.split('.').at(-1)?.toLowerCase()
  if (extension !== 'txt' && extension !== 'pdf') {
    throw new Error('Choose a .txt or .pdf file.')
  }
  if (file.size === 0) throw new Error('The selected file is empty.')
  if (file.size > MAX_FILE_BYTES) throw new Error('Files must be 25 MiB or smaller.')

  const body = new FormData()
  body.append('file', file, file.name)
  let response: Response
  try {
    response = await fetcher(`${apiUrl.replace(/\/$/, '')}/documents/upload`, {
      method: 'POST',
      body,
    })
  } catch {
    throw new Error('Could not reach the PageGlance API. Check that the backend is running.')
  }

  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new Error(`The backend returned an unreadable response (${response.status}).`)
  }
  if (!response.ok) {
    const detail = isRecord(payload) && typeof payload.detail === 'string' ? payload.detail : ''
    throw new Error(detail || `Upload failed with status ${response.status}.`)
  }
  if (!isRecord(payload) || !isUploadedDocument({ ...payload, uploaded_at: Date.now() })) {
    throw new Error('The backend returned invalid document information.')
  }
  return {
    document_id: payload.document_id as string,
    filename: payload.filename as string,
    source_type: payload.source_type as 'text' | 'pdf',
    content_hash: payload.content_hash as string,
    chunk_count: payload.chunk_count as number,
    duplicate: payload.duplicate as boolean,
    page_count: payload.page_count as number | null,
  }
}

export function citationLabel(source: AnswerSource): string {
  const name = source.filename || source.title || 'Source document'
  const pages = source.page_numbers?.length
    ? source.page_numbers
    : source.page_start !== undefined && source.page_start !== null
      ? Array.from(
          { length: Math.max(1, (source.page_end ?? source.page_start) - source.page_start + 1) },
          (_, index) => source.page_start! + index,
        )
      : []
  const pageLabel = pages.length
    ? ` · page${pages.length > 1 ? 's' : ''} ${pages.join(', ')}`
    : ''
  return `${name}${pageLabel}`
}

export function citationItems(sources: AnswerSource[]) {
  return sources.map((source) => ({
    key: `${source.document_id}:${source.id}`,
    id: source.id,
    documentId: source.document_id,
    label: citationLabel(source),
    href: source.source_url || null,
    snippet: source.snippet || null,
  }))
}

export function resolveCitationId(referenceId: string, sources: AnswerSource[]): string | null {
  return sources.some((source) => source.id === referenceId) ? referenceId : null
}

export function createScopedAskRequest(
  question: string,
  page: PageContext | null,
  document: Pick<UploadedDocument, 'document_id'> | null,
  history: AskHistoryMessage[],
): AskRequest {
  if (document) return { question, document_id: document.document_id, history }
  if (!page) throw new Error('A webpage or uploaded document must be selected.')
  return { question, page, history }
}

export interface DocumentConversation {
  updatedAt: number
  draft: string
  messages: ChatMessage[]
}

export function parseDocumentConversations(value: unknown): Record<string, DocumentConversation> {
  if (!isRecord(value)) return {}
  const conversations: Record<string, DocumentConversation> = {}
  for (const [documentId, item] of Object.entries(value)) {
    if (!DOCUMENT_ID.test(documentId) || !isRecord(item)) continue
    conversations[documentId] = {
      updatedAt: typeof item.updatedAt === 'number' ? item.updatedAt : Date.now(),
      draft: typeof item.draft === 'string' ? item.draft.slice(0, 4000) : '',
      messages: Array.isArray(item.messages)
        ? item.messages.map(parseChatMessage).filter((message) => message !== null).slice(-30)
        : [],
    }
  }
  return conversations
}
