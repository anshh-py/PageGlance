export const EXTRACT_PAGE_MESSAGE = 'PAGEGLANCE_EXTRACT_PAGE' as const

export interface PageContext {
  title: string
  url: string
  text: string
}

export interface ExtractPageRequest {
  type: typeof EXTRACT_PAGE_MESSAGE
}

export type ExtractPageResponse =
  | { ok: true; page: PageContext }
  | { ok: false; error: string }

export interface AskRequest {
  question: string
  page?: PageContext
  document_id?: string
  document_ids?: string[]
  history?: AskHistoryMessage[]
}

export interface AnswerSource {
  id: string
  document_id: string
  title: string
  source_type: string
  filename?: string | null
  source_url?: string | null
  page_start?: number | null
  page_end?: number | null
  page_numbers?: number[]
  snippet?: string | null
}

export interface AskHistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface AskResponse {
  answer: string
  sources?: AnswerSource[]
}
