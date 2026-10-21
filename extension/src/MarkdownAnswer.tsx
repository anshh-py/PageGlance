import type { ReactNode } from 'react'

type Block =
  | { type: 'heading'; depth: number; text: string }
  | { type: 'paragraph'; text: string }
  | { type: 'list'; ordered: boolean; items: string[] }
  | { type: 'quote'; text: string }
  | { type: 'code'; text: string }

const INLINE_MARKUP = /(\*\*.+?\*\*|\*[^*]+\*|`[^`]+`)/g
const HEADING = /^(#{1,3})\s+(.+)$/
const UNORDERED_ITEM = /^\s*[-*+]\s+(.+)$/
const ORDERED_ITEM = /^\s*\d+[.)]\s+(.+)$/

function parseBlocks(markdown: string): Block[] {
  const lines = markdown.replace(/\r\n?/g, '\n').split('\n')
  const blocks: Block[] = []
  let index = 0

  while (index < lines.length) {
    const line = lines[index].trim()
    if (!line) {
      index += 1
      continue
    }

    if (line.startsWith('```')) {
      const code: string[] = []
      index += 1
      while (index < lines.length && !lines[index].trim().startsWith('```')) {
        code.push(lines[index])
        index += 1
      }
      if (index < lines.length) index += 1
      blocks.push({ type: 'code', text: code.join('\n') })
      continue
    }

    const heading = line.match(HEADING)
    if (heading) {
      blocks.push({ type: 'heading', depth: heading[1].length, text: heading[2] })
      index += 1
      continue
    }

    if (line.startsWith('> ')) {
      const quote: string[] = []
      while (index < lines.length && lines[index].trim().startsWith('> ')) {
        quote.push(lines[index].trim().slice(2))
        index += 1
      }
      blocks.push({ type: 'quote', text: quote.join(' ') })
      continue
    }

    const unordered = line.match(UNORDERED_ITEM)
    const ordered = line.match(ORDERED_ITEM)
    if (unordered || ordered) {
      const orderedList = Boolean(ordered)
      const items: string[] = []
      while (index < lines.length) {
        const match = lines[index].trim().match(orderedList ? ORDERED_ITEM : UNORDERED_ITEM)
        if (!match) break
        items.push(match[1])
        index += 1
      }
      blocks.push({ type: 'list', ordered: orderedList, items })
      continue
    }

    const paragraph = [line]
    index += 1
    while (index < lines.length) {
      const next = lines[index].trim()
      if (!next || next.startsWith('```') || next.startsWith('> ') || HEADING.test(next)) break
      if (UNORDERED_ITEM.test(next) || ORDERED_ITEM.test(next)) break
      paragraph.push(next)
      index += 1
    }
    blocks.push({ type: 'paragraph', text: paragraph.join(' ') })
  }

  return blocks
}

function renderInline(text: string): ReactNode[] {
  return text.split(INLINE_MARKUP).filter(Boolean).map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index}>{part.slice(2, -2)}</strong>
    }
    if (part.startsWith('*') && part.endsWith('*')) {
      return <em key={index}>{part.slice(1, -1)}</em>
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={index}>{part.slice(1, -1)}</code>
    }
    return <span key={index}>{part}</span>
  })
}

export function MarkdownAnswer({ text }: { text: string }) {
  return (
    <div className="answer-content">
      {parseBlocks(text).map((block, index) => {
        if (block.type === 'heading') {
          const Tag = block.depth === 1 ? 'h3' : block.depth === 2 ? 'h4' : 'h5'
          return <Tag key={index}>{renderInline(block.text)}</Tag>
        }
        if (block.type === 'list') {
          const Tag = block.ordered ? 'ol' : 'ul'
          return (
            <Tag key={index}>
              {block.items.map((item, itemIndex) => <li key={itemIndex}>{renderInline(item)}</li>)}
            </Tag>
          )
        }
        if (block.type === 'quote') {
          return <blockquote key={index}>{renderInline(block.text)}</blockquote>
        }
        if (block.type === 'code') {
          return <pre key={index}><code>{block.text}</code></pre>
        }
        return <p key={index}>{renderInline(block.text)}</p>
      })}
    </div>
  )
}
