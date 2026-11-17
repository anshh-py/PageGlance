/**
 * Browser-side structured page extraction for PageGlance.
 *
 * Pure functions that walk a rendered DOM and produce normalized text that
 * preserves meaningful structure (headings, lists, tables, code blocks) and
 * includes only visible, accessible content. The module has no dependency on
 * browser globals, so it can be unit tested with minimal DOM fakes.
 *
 * The output is a single normalized string (same shape as the previous
 * `document.body.innerText` contract), so it flows through the existing
 * `PageContext` -> `/ingest/url` -> `ExtractedDocument` pipeline unchanged.
 */

export const MAX_EXTRACT_TEXT_CHARS = 100_000
export const MAX_LAZY_SCROLL_STEPS = 5
export const LAZY_SCROLL_WAIT_MS = 250

/**
 * One structured fragment of an extracted page, positioned within the returned
 * normalized text. Offsets are valid against the exact string returned by
 * ``extractPageContent`` (``start_char <= end_char <= text.length``).
 */
export interface ExtractedBlock {
  block_type: string
  text: string
  start_char: number
  end_char: number
}

const MAX_STRUCTURED_BLOCKS = 100
const MAX_BLOCK_TEXT_CHARS = 400

const TEXT_NODE = 3
const ELEMENT_NODE = 1

/** Elements whose rendered content is not meaningful page text. */
const SKIP_TAGS = new Set([
  'SCRIPT',
  'STYLE',
  'NOSCRIPT',
  'TEMPLATE',
  'HEAD',
  'TITLE',
  'META',
  'LINK',
])

/** Elements that separate passages with a line break. */
const BLOCK_TAGS = new Set([
  'P',
  'DIV',
  'BR',
  'LI',
  'TR',
  'SECTION',
  'ARTICLE',
  'BLOCKQUOTE',
  'H1',
  'H2',
  'H3',
  'H4',
  'H5',
  'H6',
  'UL',
  'OL',
  'TABLE',
  'HEADER',
  'FOOTER',
  'MAIN',
  'ASIDE',
  'NAV',
  'HR',
  'FIGURE',
  'FIGCAPTION',
])

const TABLE_CELL_TAGS = new Set(['TD', 'TH'])

/** True for elements the browser would not visually present as content. */
function isHiddenElement(el: Element): boolean {
  if ((el as HTMLElement).hidden) return true
  if (el.getAttribute('hidden') != null) return true
  if (el.getAttribute('aria-hidden') === 'true') return true
  const style = el.getAttribute('style') ?? ''
  if (/display\s*:\s*none/i.test(style)) return true
  if (/visibility\s*:\s*hidden/i.test(style)) return true
  const className = el.getAttribute('class') ?? ''
  return /(^|\s)(hidden|sr-only|visually-hidden)(\s|$)/i.test(className)
}

/** Boilerplate tokens matched against class/id names: advertisements, consent/
 * cookie banners, popups/modals, social-share controls, notifications and
 * promotions. Word-boundary anchored so e.g. "shared" does not match "share". */
const BOILERPLATE_PATTERN =
  /\b(ad|ads|adsense|advert|advertisement|sponsor|cookie|consent|popup|modal|overlay|toast|notification|social|share|sharing|newsletter|subscribe|promo|breadcrumb)\b/i

function classOrIdMatches(el: Element): boolean {
  const cls = el.getAttribute('class') ?? ''
  const id = el.getAttribute('id') ?? ''
  return BOILERPLATE_PATTERN.test(cls) || BOILERPLATE_PATTERN.test(id)
}

/** True if the subtree contains a heading (h1-h6). Used to keep header/footer
 * elements that carry meaningful titles on documentation/application pages. */
function hasSubheading(el: Element): boolean {
  const tag = el.tagName?.toUpperCase() ?? ''
  if (/^H[1-6]$/.test(tag)) return true
  for (const child of Array.from(el.childNodes ?? [])) {
    if (child.nodeType === ELEMENT_NODE && hasSubheading(child as Element)) return true
  }
  return false
}

interface Part {
  kind: 'text' | 'break' | 'code' | 'row'
  value: string
  block?: string | null
}

interface Ctx {
  parts: Part[]
  /** Stack of structured block types currently enclosing emitted text. */
  blockStack: Array<string | null>
  /** DOM nodes already emitted, so a node reachable via nested/overlapping
   * containers (e.g. slotted shadow content) is extracted only once. */
  visited: Set<Node>
}

function pushText(ctx: Ctx, value: string): void {
  if (!value) return
  ctx.parts.push({
    kind: 'text',
    value: value.replace(/\s+/g, ' '),
    block: ctx.blockStack[ctx.blockStack.length - 1] ?? null,
  })
}

function pushBreak(ctx: Ctx): void {
  ctx.parts.push({ kind: 'break', value: '' })
}

/** A table row whose cell separators must be preserved verbatim. */
function pushRow(ctx: Ctx, value: string): void {
  if (value) {
    ctx.parts.push({
      kind: 'row',
      value,
      block: ctx.blockStack[ctx.blockStack.length - 1] ?? null,
    })
  }
}

/** Collect the visible inline text of a subtree, joined into a single line. */
function inlineText(root: Element): string {
  const out: string[] = []
  const walk = (node: Node): void => {
    if (!node) return
    if (node.nodeType === TEXT_NODE) {
      out.push((node as Text).data)
      return
    }
    if (node.nodeType !== ELEMENT_NODE) return
    const el = node as Element
    if (isHiddenElement(el)) return
    const tag = el.tagName.toUpperCase()
    if (SKIP_TAGS.has(tag)) return
    if (tag === 'PRE' || tag === 'CODE') {
      out.push(el.textContent ?? '')
      return
    }
    if (el.childNodes) {
      for (const child of Array.from(el.childNodes)) walk(child)
    }
  }
  walk(root)
  return out.join('').replace(/\s+/g, ' ').trim()
}

function visitTable(table: Element, ctx: Ctx): void {
  pushBreak(ctx)
  ctx.blockStack.push('table')
  pushText(ctx, '<table>')
  pushBreak(ctx)
  for (const row of Array.from(table.childNodes ?? [])) {
    if (row.nodeType !== ELEMENT_NODE) continue
    const rowTag = (row as Element).tagName?.toUpperCase()
    if (rowTag !== 'TR') continue
    const cells: string[] = []
    for (const cell of Array.from(row.childNodes ?? [])) {
      if (cell.nodeType !== ELEMENT_NODE) continue
      const cellTag = (cell as Element).tagName?.toUpperCase()
      if (!cellTag || !TABLE_CELL_TAGS.has(cellTag)) continue
      cells.push(inlineText(cell as Element))
    }
    pushRow(ctx, cells.join('\t'))
    pushBreak(ctx)
  }
  pushText(ctx, '</table>')
  pushBreak(ctx)
  ctx.blockStack.pop()
}

function visit(node: Node | null, ctx: Ctx): void {
  if (!node) return
  if (ctx.visited.has(node)) return
  ctx.visited.add(node)
  const type = node.nodeType
  if (type === TEXT_NODE) {
    pushText(ctx, (node as Text).data)
    return
  }
  if (type !== ELEMENT_NODE) return
  const el = node as Element
  if (isHiddenElement(el)) return
  const tag = el.tagName.toUpperCase()
  if (SKIP_TAGS.has(tag)) return

  // Boilerplate filtering: navigation is never page content; class/id-based
  // ads, consent/cookie banners, popups/modals and social-share controls are
  // dropped; header/footer chrome is dropped unless it carries a heading (e.g.
  // a documentation page title). This is conservative: nothing else is removed,
  // so no clear main-content region still yields the full remaining text.
  if (tag === 'NAV') return
  if (classOrIdMatches(el)) return
  if ((tag === 'HEADER' || tag === 'FOOTER') && !hasSubheading(el)) return

  // Accessible same-origin iframes: contentDocument is readable only when the
  // frame is same-origin. Cross-origin frames throw on access and are skipped
  // (browser security is not bypassed).
  if (tag === 'IFRAME') {
    let doc: Document | null = null
    try {
      doc = (el as HTMLIFrameElement).contentDocument
    } catch {
      doc = null
    }
    if (doc && doc.body) {
      pushBreak(ctx)
      pushText(ctx, '[iframe]')
      pushBreak(ctx)
      visit(doc.body, ctx)
      pushBreak(ctx)
      pushText(ctx, '[/iframe]')
      pushBreak(ctx)
    }
    return
  }

  if (tag === 'TABLE') {
    visitTable(el, ctx)
    return
  }

  // Code blocks keep their whitespace so indentation and line breaks survive.
  if (tag === 'PRE' || tag === 'CODE') {
    const code = (el.textContent ?? '').replace(/^\n+/, '').replace(/\n+\s*$/, '')
    if (code) {
      pushBreak(ctx)
      ctx.parts.push({ kind: 'code', value: code, block: 'code' })
      pushBreak(ctx)
    }
    return
  }

  // Open Shadow DOM is traversable; closed shadow roots are not exposed and are
  // skipped (no attempt to bypass encapsulation).
  if (el.shadowRoot && el.shadowRoot.childNodes) {
    for (const child of Array.from(el.shadowRoot.childNodes)) visit(child, ctx)
  }

  // Structured block boundaries (headings, paragraphs, list items, blockquotes)
  // are recorded so they survive as block references with valid offsets.
  let blockType: string | null = null
  if (/^H[1-6]$/.test(tag)) blockType = 'heading'
  else if (tag === 'P') blockType = 'paragraph'
  else if (tag === 'LI') blockType = 'list_item'
  else if (tag === 'BLOCKQUOTE') blockType = 'quote'
  if (blockType) ctx.blockStack.push(blockType)

  if (el.childNodes) {
    // Prioritize semantic main/content regions so meaningful text leads the
    // output; all other non-boilerplate children are still visited, so nothing
    // meaningful is lost when no clear main-content region exists (fallback).
    const children = Array.from(el.childNodes)
    const isContentRegion = (n: Node): boolean =>
      n.nodeType === ELEMENT_NODE &&
      ((n as Element).tagName?.toUpperCase() === 'MAIN' ||
        (n as Element).tagName?.toUpperCase() === 'ARTICLE')
    const regions = children.filter(isContentRegion)
    const rest = children.filter((n) => !isContentRegion(n))
    for (const child of [...regions, ...rest]) visit(child, ctx)
  }

  if (blockType) ctx.blockStack.pop()

  if (BLOCK_TAGS.has(tag)) pushBreak(ctx)
}

/**
 * Extract normalized visible text from a rendered DOM subtree.
 *
 * Returns a single bounded string with meaningful structural separation plus the
 * list of structured block references positioned within that exact string. The
 * ``text`` shape mirrors the previous ``innerText`` contract so the callers and
 * the ingestion pipeline are unchanged; ``blocks`` is an additive extension.
 */
export function extractPageContent(root: Element): {
  text: string
  blocks: ExtractedBlock[]
} {
  const ctx: Ctx = { parts: [], blockStack: [], visited: new Set() }
  visit(root, ctx)

  // Each emitted line records the block type it belongs to (null = unstructured).
  const lines: Array<{ value: string; block: string | null }> = []
  let current = ''
  let currentBlock: string | null = null
  const flushCurrent = (): void => {
    if (current.trim()) {
      lines.push({ value: current.trim(), block: currentBlock })
    }
    current = ''
    currentBlock = null
  }

  for (const part of ctx.parts) {
    if (part.kind === 'break') {
      flushCurrent()
    } else if (part.kind === 'code') {
      flushCurrent()
      for (const codeLine of part.value.split(/\r?\n/)) {
        // Preserve code indentation (leading whitespace); only trim the tail.
        lines.push({ value: codeLine.trimEnd(), block: 'code' })
      }
    } else {
      current += part.value
      if (part.block != null) currentBlock = part.block
    }
  }
  flushCurrent()

  // Assemble the normalized text while recording each block's character span
  // against the exact returned string (capped at MAX_EXTRACT_TEXT_CHARS).
  const emittedLines: Array<{ value: string; block: string | null; start: number }> = []
  let pos = 0
  for (const line of lines) {
    if (!line.value.trim()) continue
    emittedLines.push({ value: line.value, block: line.block, start: pos })
    pos += line.value.length + 1 // account for the trailing newline
  }
  const full = emittedLines.map((l) => l.value).join('\n')
  const text = full.slice(0, MAX_EXTRACT_TEXT_CHARS)

  const blocks: ExtractedBlock[] = []
  let i = 0
  while (i < emittedLines.length && blocks.length < MAX_STRUCTURED_BLOCKS) {
    const line = emittedLines[i]
    if (!line.block) {
      i += 1
      continue
    }
    const type = line.block
    const start_char = line.start
    const textLines: string[] = [line.value]
    let end_char = start_char + line.value.length
    let j = i + 1
    while (j < emittedLines.length && emittedLines[j].block === type) {
      textLines.push(emittedLines[j].value)
      end_char = emittedLines[j].start + emittedLines[j].value.length
      j += 1
    }
    if (start_char < text.length) {
      blocks.push({
        block_type: type,
        text: textLines.join('\n').slice(0, MAX_BLOCK_TEXT_CHARS),
        start_char,
        end_char: Math.min(end_char, text.length),
      })
    }
    i = j
  }

  return { text, blocks }
}

/**
 * Bounded lazy-load capture. Scrolls down a limited number of steps to trigger
 * lazy-loaded content, then restores the original scroll position. Never
 * scrolls infinitely and does not permanently disturb the page.
 */
export async function loadLazyContent(
  windowObj: Window,
  opts?: { maxSteps?: number; waitMs?: number },
): Promise<void> {
  const w = windowObj
  if (typeof w.scrollTo !== 'function') return
  const maxSteps = opts?.maxSteps ?? MAX_LAZY_SCROLL_STEPS
  const waitMs = opts?.waitMs ?? LAZY_SCROLL_WAIT_MS
  const delay = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms))

  const originalY = typeof w.scrollY === 'number' ? w.scrollY : 0
  let steps = 0
  try {
    while (steps < maxSteps) {
      const viewport = typeof w.innerHeight === 'number' ? w.innerHeight : 0
      const scrollHeight = w.document?.documentElement?.scrollHeight ?? 0
      if (viewport <= 0 || scrollHeight <= 0) break
      const y = typeof w.scrollY === 'number' ? w.scrollY : 0
      const maxY = Math.max(0, scrollHeight - viewport)
      if (y >= maxY) break
      w.scrollTo(0, Math.min(y + Math.max(viewport, 256), maxY))
      steps += 1
      await delay(waitMs)
    }
    await delay(waitMs)
  } finally {
    w.scrollTo(0, originalY)
  }
}
