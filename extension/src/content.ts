import type { ExtractPageRequest, ExtractPageResponse } from './messages'
import { browser } from './browser'
import { extractPageContent, loadLazyContent } from './extract'

const EXTRACT_PAGE_MESSAGE = 'PAGEGLANCE_EXTRACT_PAGE'
const MAX_PAGE_TEXT_LENGTH = 100_000
const initializedKey = '__pageGlanceContentScriptInitialized'

const scriptWindow = window as typeof window & { [initializedKey]?: boolean }

if (!scriptWindow[initializedKey]) {
  scriptWindow[initializedKey] = true

  browser.runtime.onMessage.addListener(
    (message: unknown, _sender, sendResponse: (response: ExtractPageResponse) => void) => {
      if (!isExtractPageRequest(message)) return

      // Bounded lazy-load capture before extraction, so dynamically added
      // content is included. The listener returns true to keep the message
      // channel open until the async work resolves.
      void (async () => {
        try {
          await loadLazyContent(window)
          const content = extractPageContent(document.body ?? document.documentElement)
          sendResponse({
            ok: true,
            page: {
              title: document.title,
              url: window.location.href,
              text: content.text.slice(0, MAX_PAGE_TEXT_LENGTH),
              blocks: content.blocks,
            },
          })
        } catch (error) {
          sendResponse({
            ok: false,
            error: error instanceof Error ? error.message : String(error),
          })
        }
      })()
      return true
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
