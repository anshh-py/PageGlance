import type { ExtractPageRequest, ExtractPageResponse } from './messages'

const EXTRACT_PAGE_MESSAGE = 'PAGEGLANCE_EXTRACT_PAGE'
const MAX_PAGE_TEXT_LENGTH = 100_000
const initializedKey = '__pageGlanceContentScriptInitialized'

const scriptWindow = window as typeof window & { [initializedKey]?: boolean }

if (!scriptWindow[initializedKey]) {
  scriptWindow[initializedKey] = true

  chrome.runtime.onMessage.addListener(
    (message: unknown, _sender, sendResponse: (response: ExtractPageResponse) => void) => {
      if (!isExtractPageRequest(message)) return

      const text = document.body?.innerText?.trim() ?? ''
      sendResponse({
        ok: true,
        page: {
          title: document.title,
          url: window.location.href,
          text: text.slice(0, MAX_PAGE_TEXT_LENGTH),
        },
      })
    },
  )
}

function isExtractPageRequest(message: unknown): message is ExtractPageRequest {
  return (
    typeof message === 'object' &&
    message !== null &&
    'type' in message &&
    message.type === EXTRACT_PAGE_MESSAGE
  )
}
