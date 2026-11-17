import assert from 'node:assert/strict'
import { test } from 'node:test'
import { isRestrictedPageUrl } from './pageSupport.ts'

test('detects browser-internal chrome:// pages as restricted', () => {
  assert.equal(isRestrictedPageUrl('chrome://settings'), true)
  assert.equal(isRestrictedPageUrl('chrome://newtab/'), true)
  assert.equal(isRestrictedPageUrl('chrome://extensions/'), true)
})

test('detects chrome-extension:// pages as restricted', () => {
  assert.equal(isRestrictedPageUrl('chrome-extension://abc123def456/options.html'), true)
})

test('detects edge:// pages as restricted', () => {
  assert.equal(isRestrictedPageUrl('edge://flags'), true)
  assert.equal(isRestrictedPageUrl('edge://settings/privacy'), true)
})

test('detects about: pages as restricted', () => {
  assert.equal(isRestrictedPageUrl('about:blank'), true)
  assert.equal(isRestrictedPageUrl('about:config'), true)
})

test('detects Firefox internal pages as restricted', () => {
  assert.equal(isRestrictedPageUrl('moz-extension://uuid-here/popup.html'), true)
})

test('detects Chrome Web Store pages as restricted', () => {
  assert.equal(
    isRestrictedPageUrl('https://chromewebstore.google.com/detail/some-extension/abc123'),
    true,
  )
  assert.equal(
    isRestrictedPageUrl('https://chrome.google.com/webstore/detail/another-extension/def456'),
    true,
  )
})

test('accepts ordinary HTTP/HTTPS websites as supported', () => {
  assert.equal(isRestrictedPageUrl('https://example.com/article?utm_source=sh#section'), false)
  assert.equal(isRestrictedPageUrl('http://localhost:8000/'), false)
  assert.equal(isRestrictedPageUrl('https://127.0.0.1:8000/health'), false)
})

test('treats other protocols and malformed input as not restricted', () => {
  assert.equal(isRestrictedPageUrl('ftp://example.com/file.txt'), false)
  assert.equal(isRestrictedPageUrl('file:///tmp/a.txt'), false)
  assert.equal(isRestrictedPageUrl('not-a-url'), false)
  assert.equal(isRestrictedPageUrl(''), false)
})
