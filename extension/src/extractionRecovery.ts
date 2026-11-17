/**
 * M7.1 — Extraction failure classification & fallback decision logic.
 *
 * Pure, deterministic helpers that map a failed page-extraction attempt to a
 * stable failure category, then to the single next permitted fallback action.
 *
 * This is the foundation for the universal ingestion fallback system. It does
 * NOT itself run any fallback mechanism (retry, alternate extraction, URL
 * import, manual upload) — that work is deferred to later milestones. It also
 * never bypasses browser security, paywalls, CAPTCHAs or URL restrictions: the
 * decision function only *reports* a permitted action; the caller remains
 * responsible for honoring the same safety constraints.
 */

/** Stable, typed failure categories for page extraction. */
export const ExtractionFailureCategory = {
  EMPTY_CONTENT: 'empty_content',
  RESTRICTED_PAGE: 'restricted_page',
  ACCESS_DENIED: 'access_denied',
  CONTENT_SCRIPT_UNAVAILABLE: 'content_script_unavailable',
  EXTRACTION_TIMEOUT: 'extraction_timeout',
  NAVIGATION_OR_TAB_CHANGED: 'navigation_or_tab_changed',
  INVALID_EXTRACTION_RESPONSE: 'invalid_extraction_response',
  NETWORK_BACKEND_FAILURE: 'network_backend_failure',
  UNKNOWN: 'unknown',
} as const

export type ExtractionFailureCategory =
  (typeof ExtractionFailureCategory)[keyof typeof ExtractionFailureCategory]

/** The next permitted fallback action for a classified failure. */
export const FallbackAction = {
  RETRY_ACTIVE_TAB_EXTRACTION: 'retry_active_tab_extraction',
  RETRY_INGESTION: 'retry_ingestion',
  USE_ALTERNATE_EXTRACTION: 'use_alternate_extraction',
  OFFER_URL_IMPORT: 'offer_url_import',
  OFFER_MANUAL_UPLOAD: 'offer_manual_upload',
  STOP: 'stop',
} as const

export type FallbackAction = (typeof FallbackAction)[keyof typeof FallbackAction]

/** Input describing one failed extraction attempt. */
export interface ExtractionAttemptDescriptor {
  /** True when the target URL is a browser-restricted page. */
  restricted?: boolean
  /** Characters of text extracted before failure (0 = none). */
  textLength?: number
  /** Optional explicit machine-readable failure code, when known. */
  code?: string
  /** The thrown value (Error or otherwise), if extraction threw. */
  error?: unknown
}

function messageOf(value: unknown): string {
  if (value instanceof Error) return value.message ?? ''
  if (typeof value === 'string') return value
  if (value && typeof value === 'object' && 'message' in value) {
    return String((value as { message: unknown }).message)
  }
  return ''
}

/**
 * Map a normalized (lowercased) message to a category. Message heuristics are
 * a convenience for the current code, which surfaces plain text errors; callers
 * that can supply a structured `code` or `restricted` flag should prefer those
 * (they are checked first and are unambiguous).
 */
function categoryFromMessage(lower: string): ExtractionFailureCategory | null {
  if (!lower) return null
  if (lower.includes('no readable page text') || lower.includes('empty page')) {
    return ExtractionFailureCategory.EMPTY_CONTENT
  }
  if (
    lower.includes('security restrictions') ||
    lower.includes('cannot read this tab') ||
    lower.includes('internal pages') ||
    lower.includes('restricted')
  ) {
    return ExtractionFailureCategory.RESTRICTED_PAGE
  }
  if (lower.includes('access denied') || lower.includes('permission denied')) {
    return ExtractionFailureCategory.ACCESS_DENIED
  }
  if (lower.includes('timeout') || lower.includes('timed out')) {
    return ExtractionFailureCategory.EXTRACTION_TIMEOUT
  }
  if (lower.includes('page changed') || lower.includes('tab changed') || lower.includes('navigat')) {
    return ExtractionFailureCategory.NAVIGATION_OR_TAB_CHANGED
  }
  if (lower.includes('invalid response') || lower.includes('malformed response')) {
    return ExtractionFailureCategory.INVALID_EXTRACTION_RESPONSE
  }
  if (
    lower.includes('receiving end does not exist') ||
    lower.includes('message channel closed') ||
    lower.includes('could not establish connection') ||
    lower.includes('content script') ||
    lower.includes('not found')
  ) {
    return ExtractionFailureCategory.CONTENT_SCRIPT_UNAVAILABLE
  }
  if (
    lower.includes('network') ||
    lower.includes('failed to fetch') ||
    lower.includes('backend') ||
    lower.includes('connection')
  ) {
    return ExtractionFailureCategory.NETWORK_BACKEND_FAILURE
  }
  return null
}

const CODE_TO_CATEGORY: Record<string, ExtractionFailureCategory> = {
  EMPTY_CONTENT: ExtractionFailureCategory.EMPTY_CONTENT,
  RESTRICTED_PAGE: ExtractionFailureCategory.RESTRICTED_PAGE,
  ACCESS_DENIED: ExtractionFailureCategory.ACCESS_DENIED,
  CONTENT_SCRIPT_UNAVAILABLE: ExtractionFailureCategory.CONTENT_SCRIPT_UNAVAILABLE,
  EXTRACTION_TIMEOUT: ExtractionFailureCategory.EXTRACTION_TIMEOUT,
  NAVIGATION_OR_TAB_CHANGED: ExtractionFailureCategory.NAVIGATION_OR_TAB_CHANGED,
  INVALID_EXTRACTION_RESPONSE: ExtractionFailureCategory.INVALID_EXTRACTION_RESPONSE,
  NETWORK_BACKEND_FAILURE: ExtractionFailureCategory.NETWORK_BACKEND_FAILURE,
}

/**
 * Classify a failed extraction attempt into a stable category.
 *
 * Priority order: explicit `restricted` flag -> empty content (no error, zero
 * text) -> explicit `code` -> message heuristics -> unknown.
 */
export function classifyExtractionFailure(
  descriptor: ExtractionAttemptDescriptor,
): ExtractionFailureCategory {
  if (descriptor.restricted || descriptor.code === 'RESTRICTED_PAGE') {
    return ExtractionFailureCategory.RESTRICTED_PAGE
  }
  const textLength = typeof descriptor.textLength === 'number' ? descriptor.textLength : -1
  if (!descriptor.error && textLength === 0) {
    return ExtractionFailureCategory.EMPTY_CONTENT
  }
  if (descriptor.code) {
    const matched = CODE_TO_CATEGORY[descriptor.code]
    if (matched) return matched
  }
  const fromMessage = categoryFromMessage(messageOf(descriptor.error).toLowerCase())
  return fromMessage ?? ExtractionFailureCategory.UNKNOWN
}

const CATEGORY_TO_ACTION: Record<ExtractionFailureCategory, FallbackAction> = {
  // The page rendered nothing readable. URL import is the first thing to offer
  // where it is supported (it can re-read the URL server-side); manual upload
  // remains the fallback if URL import is unavailable. We do not claim URL
  // import will necessarily succeed.
  [ExtractionFailureCategory.EMPTY_CONTENT]: FallbackAction.OFFER_URL_IMPORT,
  // Browser-internal / Web Store pages, and access-denied pages, cannot be read
  // by the extension. The only permitted path is an alternate extraction method
  // that respects the same security model — never a bypass.
  [ExtractionFailureCategory.RESTRICTED_PAGE]: FallbackAction.USE_ALTERNATE_EXTRACTION,
  [ExtractionFailureCategory.ACCESS_DENIED]: FallbackAction.USE_ALTERNATE_EXTRACTION,
  // The content script could not be injected or reached; page extraction is
  // unavailable, so offering a URL import is the natural next step.
  [ExtractionFailureCategory.CONTENT_SCRIPT_UNAVAILABLE]: FallbackAction.OFFER_URL_IMPORT,
  // Transient — retrying the same active-tab extraction is reasonable.
  [ExtractionFailureCategory.EXTRACTION_TIMEOUT]: FallbackAction.RETRY_ACTIVE_TAB_EXTRACTION,
  // The page or tab changed under us; retrying would risk capturing the wrong
  // content, so stop and ask the user to act.
  [ExtractionFailureCategory.NAVIGATION_OR_TAB_CHANGED]: FallbackAction.STOP,
  // A malformed/unusable response is not a safe thing to silently repeat — it
  // may indicate the page or script is in an inconsistent state, so retrying
  // could confuse the user or capture misleading content. Stop and ask them to
  // act instead of auto-retrying.
  [ExtractionFailureCategory.INVALID_EXTRACTION_RESPONSE]: FallbackAction.STOP,
  // Backend/network failure already produced a partial result on the server, so
  // re-running page extraction would risk a duplicate ingestion. Retry only the
  // safe ingestion step, not active-tab extraction.
  [ExtractionFailureCategory.NETWORK_BACKEND_FAILURE]: FallbackAction.RETRY_INGESTION,
  // No safe fallback known — stop with a clear explanation.
  [ExtractionFailureCategory.UNKNOWN]: FallbackAction.STOP,
}

/** Determine the single next permitted fallback action for a category. */
export function decideFallbackAction(category: ExtractionFailureCategory): FallbackAction {
  return CATEGORY_TO_ACTION[category]
}

export interface RecoveryPlan {
  category: ExtractionFailureCategory
  action: FallbackAction
}

/** Convenience: classify a failed attempt and produce its recovery plan. */
export function planRecovery(descriptor: ExtractionAttemptDescriptor): RecoveryPlan {
  const category = classifyExtractionFailure(descriptor)
  return { category, action: decideFallbackAction(category) }
}
