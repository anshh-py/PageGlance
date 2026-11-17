/**
 * Detection of browser-restricted pages that PageGlance cannot read.
 *
 * Chrome and Firefox forbid extensions from injecting content scripts into
 * internal pages (chrome://, edge://, about:, chrome-extension://, ...) and
 * into the Chrome Web Store. Instead of attempting a script injection that is
 * guaranteed to fail (and then surfacing a raw error), we detect these URLs
 * up front so the UI can show a clear, friendly unsupported-page state.
 */

export const RESTRICTED_PAGE_MESSAGE =
  'Browser security restrictions prevent PageGlance from reading this page. Open a regular http or https webpage, then try again.'

const RESTRICTED_PROTOCOLS = new Set([
  'chrome:',
  'chrome-extension:',
  'edge:',
  'moz-extension:',
  'about:',
])

// The Chrome Web Store is served over https but still blocks content scripts,
// so it must be detected by host rather than by protocol.
const WEBSTORE_HOSTS = new Set(['chromewebstore.google.com', 'chrome.google.com'])

/**
 * Returns true when the given URL is a browser-internal or Web Store page that
 * PageGlance cannot extract content from. Non-URL strings and ordinary web
 * addresses return false.
 */
export function isRestrictedPageUrl(value: string): boolean {
  let url: URL
  try {
    url = new URL(value)
  } catch {
    return false
  }
  if (RESTRICTED_PROTOCOLS.has(url.protocol)) return true
  return url.protocol === 'https:' && WEBSTORE_HOSTS.has(url.hostname)
}
