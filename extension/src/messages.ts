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
  page: PageContext
  history?: AskHistoryMessage[]
}

export interface AskHistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface AskResponse {
  answer: string
}
