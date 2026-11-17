import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  classifyExtractionFailure,
  decideFallbackAction,
  planRecovery,
  ExtractionFailureCategory,
  FallbackAction,
} from './extractionRecovery.ts'

test('classifies extraction failures from typed descriptors', () => {
  assert.equal(
    classifyExtractionFailure({ restricted: true }),
    ExtractionFailureCategory.RESTRICTED_PAGE,
  )
  assert.equal(
    classifyExtractionFailure({ code: 'ACCESS_DENIED' }),
    ExtractionFailureCategory.ACCESS_DENIED,
  )
  assert.equal(
    classifyExtractionFailure({ code: 'EXTRACTION_TIMEOUT' }),
    ExtractionFailureCategory.EXTRACTION_TIMEOUT,
  )
  assert.equal(
    classifyExtractionFailure({ code: 'CONTENT_SCRIPT_UNAVAILABLE' }),
    ExtractionFailureCategory.CONTENT_SCRIPT_UNAVAILABLE,
  )
  // Empty content: no error and zero text extracted.
  assert.equal(
    classifyExtractionFailure({ textLength: 0 }),
    ExtractionFailureCategory.EMPTY_CONTENT,
  )
  // Unknown when nothing else matches.
  assert.equal(
    classifyExtractionFailure({ error: new Error('something unexpected') }),
    ExtractionFailureCategory.UNKNOWN,
  )
})

test('classifies known message strings from the current extraction flow', () => {
  // The existing extractPage() wrapper and content-script error strings.
  assert.equal(
    classifyExtractionFailure({ error: new Error('PageGlance cannot read this tab. Chrome restricts extensions on internal pages and the Chrome Web Store.') }),
    ExtractionFailureCategory.RESTRICTED_PAGE,
  )
  assert.equal(
    classifyExtractionFailure({ error: new Error('No readable page text was found on this tab.') }),
    ExtractionFailureCategory.EMPTY_CONTENT,
  )
  assert.equal(
    classifyExtractionFailure({ error: new Error('The active page changed. Close and reopen PageGlance on the new page.') }),
    ExtractionFailureCategory.NAVIGATION_OR_TAB_CHANGED,
  )
  assert.equal(
    classifyExtractionFailure({ error: new Error('Could not establish connection. Receiving end does not exist.') }),
    ExtractionFailureCategory.CONTENT_SCRIPT_UNAVAILABLE,
  )
  assert.equal(
    classifyExtractionFailure({ error: new Error('Request timed out.') }),
    ExtractionFailureCategory.EXTRACTION_TIMEOUT,
  )
  assert.equal(
    classifyExtractionFailure({ error: new Error('Network error: failed to fetch') }),
    ExtractionFailureCategory.NETWORK_BACKEND_FAILURE,
  )
})

test('maps every failure category to the correct next fallback action', () => {
  assert.equal(
    decideFallbackAction(ExtractionFailureCategory.EMPTY_CONTENT),
    FallbackAction.OFFER_URL_IMPORT,
  )
  assert.equal(
    decideFallbackAction(ExtractionFailureCategory.RESTRICTED_PAGE),
    FallbackAction.USE_ALTERNATE_EXTRACTION,
  )
  assert.equal(
    decideFallbackAction(ExtractionFailureCategory.ACCESS_DENIED),
    FallbackAction.USE_ALTERNATE_EXTRACTION,
  )
  assert.equal(
    decideFallbackAction(ExtractionFailureCategory.CONTENT_SCRIPT_UNAVAILABLE),
    FallbackAction.OFFER_URL_IMPORT,
  )
  assert.equal(
    decideFallbackAction(ExtractionFailureCategory.EXTRACTION_TIMEOUT),
    FallbackAction.RETRY_ACTIVE_TAB_EXTRACTION,
  )
  assert.equal(
    decideFallbackAction(ExtractionFailureCategory.NAVIGATION_OR_TAB_CHANGED),
    FallbackAction.STOP,
  )
  assert.equal(
    decideFallbackAction(ExtractionFailureCategory.INVALID_EXTRACTION_RESPONSE),
    FallbackAction.STOP,
  )
  assert.equal(
    decideFallbackAction(ExtractionFailureCategory.NETWORK_BACKEND_FAILURE),
    FallbackAction.RETRY_INGESTION,
  )
  assert.equal(
    decideFallbackAction(ExtractionFailureCategory.UNKNOWN),
    FallbackAction.STOP,
  )
})

test('composes classification and decision into a recovery plan', () => {
  assert.deepEqual(planRecovery({ restricted: true }), {
    category: ExtractionFailureCategory.RESTRICTED_PAGE,
    action: FallbackAction.USE_ALTERNATE_EXTRACTION,
  })
  assert.deepEqual(planRecovery({ textLength: 0 }), {
    category: ExtractionFailureCategory.EMPTY_CONTENT,
    action: FallbackAction.OFFER_URL_IMPORT,
  })
  assert.deepEqual(planRecovery({ code: 'NETWORK_BACKEND_FAILURE' }), {
    category: ExtractionFailureCategory.NETWORK_BACKEND_FAILURE,
    action: FallbackAction.RETRY_INGESTION,
  })
  assert.deepEqual(planRecovery({ code: 'EXTRACTION_TIMEOUT' }), {
    category: ExtractionFailureCategory.EXTRACTION_TIMEOUT,
    action: FallbackAction.RETRY_ACTIVE_TAB_EXTRACTION,
  })
  // Invalid responses must never plan an automatic retry.
  assert.deepEqual(planRecovery({ code: 'INVALID_EXTRACTION_RESPONSE' }), {
    category: ExtractionFailureCategory.INVALID_EXTRACTION_RESPONSE,
    action: FallbackAction.STOP,
  })
})

test('recovery panel covers alternate-extraction and backend-ingestion actions', () => {
  // These two actions are surfaced by the recovery panel; their categories must
  // keep mapping to them so a classified failure always has a visible recovery
  // instruction (regression: these branches were previously unrendered).
  assert.equal(
    classifyExtractionFailure({ error: new Error('Access denied.') }),
    ExtractionFailureCategory.ACCESS_DENIED,
  )
  assert.equal(
    decideFallbackAction(ExtractionFailureCategory.ACCESS_DENIED),
    FallbackAction.USE_ALTERNATE_EXTRACTION,
  )
  assert.equal(
    classifyExtractionFailure({ error: new Error('Network error: failed to fetch') }),
    ExtractionFailureCategory.NETWORK_BACKEND_FAILURE,
  )
  assert.equal(
    decideFallbackAction(ExtractionFailureCategory.NETWORK_BACKEND_FAILURE),
    FallbackAction.RETRY_INGESTION,
  )
  assert.equal(
    decideFallbackAction(ExtractionFailureCategory.RESTRICTED_PAGE),
    FallbackAction.USE_ALTERNATE_EXTRACTION,
  )
})
