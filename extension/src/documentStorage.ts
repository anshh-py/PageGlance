import {
  DOCUMENT_CHAT_STORE_KEY,
  DOCUMENT_LIBRARY_KEY,
  parseDocumentConversations,
  parseUploadedDocuments,
  type DocumentConversation,
  type UploadedDocument,
} from './documents.ts'
import { browser } from './browser.ts'

export async function loadUploadedDocuments(): Promise<UploadedDocument[]> {
  try {
    const stored = await browser.storage.local.get(DOCUMENT_LIBRARY_KEY)
    return parseUploadedDocuments(stored[DOCUMENT_LIBRARY_KEY])
  } catch {
    return []
  }
}

export async function saveUploadedDocuments(documents: UploadedDocument[]): Promise<void> {
  await browser.storage.local.set({ [DOCUMENT_LIBRARY_KEY]: documents })
}

export async function loadDocumentConversations(): Promise<Record<string, DocumentConversation>> {
  try {
    const stored = await browser.storage.local.get(DOCUMENT_CHAT_STORE_KEY)
    return parseDocumentConversations(stored[DOCUMENT_CHAT_STORE_KEY])
  } catch {
    return {}
  }
}

export async function saveDocumentConversations(
  conversations: Record<string, DocumentConversation>,
): Promise<void> {
  await browser.storage.local.set({ [DOCUMENT_CHAT_STORE_KEY]: conversations })
}
