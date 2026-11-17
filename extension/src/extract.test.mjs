import assert from 'node:assert/strict'
import { test } from 'node:test'
import { extractPageContent, loadLazyContent, MAX_EXTRACT_TEXT_CHARS } from './extract.ts'

// --- Minimal DOM fakes (subset of the DOM APIs the extractor uses) ---

const TEXT_NODE = 3
const ELEMENT_NODE = 1

function textNode(data) {
  return { nodeType: TEXT_NODE, data, textContent: data }
}

function element(tag, opts = {}, children = []) {
  const attrs = opts.attrs || {}
  const node = {
    nodeType: ELEMENT_NODE,
    tagName: tag.toUpperCase(),
    childNodes: children,
    getAttribute(name) {
      return Object.prototype.hasOwnProperty.call(attrs, name) ? attrs[name] : null
    },
    shadowRoot: opts.shadowRoot ?? null,
    contentDocument: opts.contentDocument ?? null,
    hidden: Boolean(opts.hidden),
  }
  Object.defineProperty(node, 'textContent', {
    get() {
      return children
        .map((child) => (child.nodeType === TEXT_NODE ? child.data : child.textContent ?? ''))
        .join('')
    },
  })
  return node
}

// --- 1. Dynamic DOM / SPA content extraction ---

test('extracts dynamically rendered SPA content and skips scripts/styles/hidden DOM', () => {
  const page = element('article', {}, [
    element('h1', {}, [textNode('Dynamic Title')]),
    element('script', {}, [textNode('window.secret = true')]),
    element('style', {}, [textNode('.a { color: red }')]),
    element('div', { attrs: { 'aria-hidden': 'true' } }, [textNode('should not appear')]),
    element('div', { attrs: { id: 'app' } }, [textNode('initial shell')]),
  ])

  // Simulate an SPA runtime replacing the mount point after some delay.
  const root = page.childNodes[4]
  root.childNodes = [textNode('content rendered by SPA at runtime')]

  const { text } = extractPageContent(page)
  assert.ok(text.includes('Dynamic Title'))
  assert.ok(text.includes('content rendered by SPA at runtime'))
  assert.ok(!text.includes('initial shell'))
  assert.ok(!text.includes('window.secret'))
  assert.ok(!text.includes('color: red'))
  assert.ok(!text.includes('should not appear'))
})

test('lazy-load scroll is bounded and restores the original scroll position', async () => {
  const calls = []
  const fakeWindow = {
    innerHeight: 500,
    scrollY: 0,
    document: { documentElement: { scrollHeight: 5000 } },
    scrollTo(x, y) {
      calls.push(y)
      fakeWindow.scrollY = y
    },
  }

  await loadLazyContent(fakeWindow, { maxSteps: 3, waitMs: 1 })
  assert.ok(calls.length <= 3 + 1, 'scroll steps must be bounded')
  assert.equal(fakeWindow.scrollY, 0, 'original scroll position must be restored')
})

// --- 2. Structured content: tables and code blocks ---

test('extracts tables with tab-separated cells and preserves code block whitespace', () => {
  const table = element('table', {}, [
    element('tr', {}, [element('th', {}, [textNode('Name')]), element('th', {}, [textNode('Value')])]),
    element('tr', {}, [element('td', {}, [textNode('alpha')]), element('td', {}, [textNode('1')])]),
    element('tr', {}, [element('td', {}, [textNode('beta')]), element('td', {}, [textNode('42')])]),
  ])
  const code = element('pre', {}, [
    element('code', {}, [textNode('const x = 1;\n  console.log(x);\n')]),
  ])
  const page = element('main', {}, [element('h2', {}, [textNode('Stats')]), table, code])

  const { text } = extractPageContent(page)
  assert.ok(text.includes('Stats'))
  assert.ok(text.includes('Name\tValue'), 'header cells should be tab-separated')
  assert.ok(text.includes('alpha\t1'))
  assert.ok(text.includes('beta\t42'))
  assert.ok(text.includes('const x = 1;'))
  assert.ok(text.includes('  console.log(x);'), 'code indentation should be preserved')
})

// --- 3. Extraction boundaries and ingestion-contract compatibility ---

test('bounds output size and returns the normalized ingestion text shape', () => {
  const huge = element('div', {}, [textNode('word '.repeat(100_000))])
  const { text } = extractPageContent(huge)
  assert.ok(text.length <= MAX_EXTRACT_TEXT_CHARS)
  assert.ok(typeof text === 'string')
  assert.ok(text.length > 0)
})

test('includes accessible same-origin iframes and open shadow DOM; skips inaccessible ones', () => {
  const shadowSpan = element('span', {}, [textNode('inside open shadow')])
  const withShadow = element('div', {}, [])
  withShadow.shadowRoot = { childNodes: [shadowSpan] }

  const iframeDoc = { body: element('p', {}, [textNode('same-origin iframe text')]) }
  const sameOriginFrame = element('iframe', {}, [])
  sameOriginFrame.contentDocument = iframeDoc
  const closedFrame = element('iframe', {}, [])
  closedFrame.contentDocument = null // cross-origin / inaccessible

  const page = element('article', {}, [withShadow, sameOriginFrame, closedFrame])
  const { text } = extractPageContent(page)
  assert.ok(text.includes('inside open shadow'))
  assert.ok(text.includes('same-origin iframe text'))
})

// --- M2.1.1: structured block references with valid offsets ---

test('structured blocks carry valid offsets against the normalized text', () => {
  const page = element('article', {}, [
    element('h2', {}, [textNode('Overview')]),
    element('p', {}, [textNode('First paragraph here.')]),
    element('ul', {}, [
      element('li', {}, [textNode('item one')]),
      element('li', {}, [textNode('item two')]),
    ]),
  ])
  const { text, blocks } = extractPageContent(page)
  assert.ok(blocks.length >= 3, 'expected heading, paragraph and list blocks')
  const types = new Set(blocks.map((b) => b.block_type))
  assert.ok(types.has('heading'))
  assert.ok(types.has('paragraph'))
  assert.ok(types.has('list_item'))

  for (const block of blocks) {
    assert.ok(block.start_char >= 0)
    assert.ok(block.end_char <= text.length, 'offset must not exceed text length')
    assert.ok(block.start_char <= block.end_char)
    const slice = text.slice(block.start_char, block.end_char)
    assert.ok(
      slice.includes(block.text.slice(0, 10)),
      `text slice should contain the block text for ${block.block_type}`,
    )
  }
})

test('existing webpage capture remains backward-compatible', () => {
  // Same fixed DOM as the pre-existing tests; the flattened text must be unchanged.
  const page = element('main', {}, [
    element('h1', {}, [textNode('Backward')]),
    element('p', {}, [textNode('hello world')]),
  ])
  const { text } = extractPageContent(page)
  assert.ok(text.includes('Backward'))
  assert.ok(text.includes('hello world'))
  assert.ok(typeof text === 'string')
  assert.ok(text.length > 0)
})

// --- M2.2: content-quality filtering & main-content prioritization ---

test('article extraction excludes navigation and advertising/cookie boilerplate', () => {
  const page = element('body', {}, [
    element('nav', {}, [element('a', {}, [textNode('Home')]), element('a', {}, [textNode('Pricing')])]),
    element('div', { attrs: { class: 'ad-banner' } }, [textNode('Buy our amazing product now')]),
    element('div', { attrs: { id: 'cookie-consent' } }, [textNode('Accept cookies')]),
    element('aside', { attrs: { class: 'social-share' } }, [textNode('Share on X')]),
    element('article', {}, [
      element('h1', {}, [textNode('Main Article Title')]),
      element('p', {}, [textNode('This is the real article body.')]),
    ]),
  ])
  const { text } = extractPageContent(page)
  // Meaningful article content is retained.
  assert.ok(text.includes('Main Article Title'))
  assert.ok(text.includes('real article body'))
  // Navigation and boilerplate are excluded.
  assert.ok(!text.includes('Home'))
  assert.ok(!text.includes('Pricing'))
  assert.ok(!text.includes('Buy our amazing product'))
  assert.ok(!text.includes('Accept cookies'))
  assert.ok(!text.includes('Share on X'))
})

test('documentation/application page retains content without an article element', () => {
  // No <article>/<main>. Header carries the page title (a heading) so it is
  // kept; navigation and a heading-less footer are treated as chrome.
  const page = element('body', {}, [
    element('header', { attrs: { class: 'site-header' } }, [
      element('h1', {}, [textNode('MyApp Documentation')]),
    ]),
    element('nav', {}, [element('a', {}, [textNode('Docs')])]),
    element('div', { attrs: { class: 'content' } }, [
      element('h2', {}, [textNode('Getting Started')]),
      element('p', {}, [textNode('Install the package with pip.')]),
    ]),
    element('footer', { attrs: { class: 'site-footer' } }, [
      element('p', {}, [textNode('© 2024 MyApp')]),
    ]),
  ])
  const { text } = extractPageContent(page)
  assert.ok(text.includes('MyApp Documentation'), 'title header should be preserved')
  assert.ok(text.includes('Getting Started'))
  assert.ok(text.includes('Install the package with pip.'))
  assert.ok(!text.includes('Docs'), 'navigation should be removed')
  assert.ok(!text.includes('© 2024'), 'heading-less footer should be removed')
  assert.ok(text.length > 0, 'fallback must never return empty content')
})

test('structured block offsets remain valid after boilerplate filtering', () => {
  const page = element('article', {}, [
    element('nav', {}, [element('a', {}, [textNode('Skip link')])]),
    element('h2', {}, [textNode('Sections')]),
    element('div', { attrs: { class: 'ad' } }, [textNode('sponsored')]),
    element('p', {}, [textNode('First section text.')]),
    element('ul', {}, [
      element('li', {}, [textNode('alpha')]),
      element('li', {}, [textNode('beta')]),
    ]),
  ])
  const { text, blocks } = extractPageContent(page)
  assert.ok(blocks.length > 0)
  for (const block of blocks) {
    assert.ok(block.start_char >= 0)
    assert.ok(block.end_char <= text.length, 'offset must not exceed text length')
    assert.ok(block.start_char <= block.end_char)
  }
  // Boilerplate must not appear inside any structured block.
  const joined = blocks.map((b) => b.text).join('\n')
  assert.ok(!joined.includes('sponsored'))
  assert.ok(!joined.includes('Skip link'))
  assert.ok(blocks.some((b) => b.block_type === 'heading'))
})

// --- M2.3: extraction edge-case handling ---

test('nested/overlapping containers do not duplicate the same DOM content, but keep legitimate repeated paragraphs', () => {
  const count = (text, sub) => text.split(sub).length - 1

  const shared = element('p', {}, [textNode('shared content')])
  const page = element('article', {}, [
    element('div', {}, [shared]),
    element('section', {}, [shared]), // same node reachable via overlapping containers
    element('p', {}, [textNode('Repeat')]),
    element('p', {}, [textNode('Repeat')]), // distinct node, identical text
  ])

  const { text } = extractPageContent(page)
  assert.equal(count(text, 'shared content'), 1, 'shared node must be extracted exactly once')
  assert.equal(count(text, 'Repeat'), 2, 'legitimate repeated paragraphs must both be kept')
})

test('hidden elements are excluded without dropping visible clipped or off-screen content', () => {
  const page = element('article', {}, [
    element('div', { attrs: { style: 'display: none' } }, [textNode('hidden display')]),
    element('div', { attrs: { style: 'visibility: hidden' } }, [textNode('hidden visibility')]),
    element('div', { attrs: { 'aria-hidden': 'true' } }, [textNode('hidden aria')]),
    element('div', { hidden: true }, [textNode('hidden attr')]),
    element('div', { attrs: { style: 'clip: rect(0,0,0,0); position: absolute' } }, [
      textNode('clipped content'),
    ]),
    element('div', { attrs: { style: 'position: absolute; left: -9999px' } }, [
      textNode('offscreen content'),
    ]),
    element('div', { attrs: { style: 'overflow: hidden' } }, [textNode('overflow content')]),
  ])

  const { text } = extractPageContent(page)
  assert.ok(!text.includes('hidden display'))
  assert.ok(!text.includes('hidden visibility'))
  assert.ok(!text.includes('hidden aria'))
  assert.ok(!text.includes('hidden attr'))
  assert.ok(text.includes('clipped content'), 'visually clipped content must be kept')
  assert.ok(text.includes('offscreen content'), 'off-screen content must be kept')
  assert.ok(text.includes('overflow content'), 'overflow-clipped content must be kept')
})

test('every structured block offset resolves to its exact text in the final flattened output', () => {
  const page = element('article', {}, [
    element('h1', {}, [textNode('Title')]),
    element('p', {}, [textNode('First para')]),
    element('ul', {}, [
      element('li', {}, [textNode('a')]),
      element('li', {}, [textNode('b')]),
    ]),
    element('blockquote', {}, [textNode('quoted')]),
    element('pre', {}, [element('code', {}, [textNode('code line')])]),
  ])

  const { text, blocks } = extractPageContent(page)
  assert.ok(blocks.length >= 5, 'expected heading, paragraph, list, quote and code blocks')
  for (const block of blocks) {
    assert.ok(Number.isInteger(block.start_char) && block.start_char >= 0)
    assert.ok(Number.isInteger(block.end_char) && block.end_char <= text.length)
    assert.ok(block.start_char <= block.end_char)
    const slice = text.slice(block.start_char, block.end_char)
    assert.equal(slice, block.text, `offset must resolve to exact text for ${block.block_type}`)
  }
})
