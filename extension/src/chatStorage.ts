import {
  CHAT_STORAGE_KEY,
  CHAT_STORE_VERSION,
  parseConversationStore,
  type ConversationStore,
} from './chat.ts'
import { browser } from './browser.ts'

export async function loadConversationStore(): Promise<ConversationStore> {
  try {
    const stored = await browser.storage.local.get(CHAT_STORAGE_KEY)
    return parseConversationStore(stored[CHAT_STORAGE_KEY])
  } catch {
    return { version: CHAT_STORE_VERSION, conversations: {} }
  }
}

export async function saveConversationStore(store: ConversationStore): Promise<void> {
  await browser.storage.local.set({ [CHAT_STORAGE_KEY]: store })
}

export async function clearConversationStore(): Promise<void> {
  await browser.storage.local.remove(CHAT_STORAGE_KEY)
}
