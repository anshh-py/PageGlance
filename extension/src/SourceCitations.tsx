import { useEffect, useRef, useState } from 'react'
import type { AnswerSource } from './messages'
import { citationItems, resolveCitationId } from './documents'
import { MarkdownAnswer } from './MarkdownAnswer'

export function AnswerWithSources({ text, sources }: { text: string; sources: AnswerSource[] }) {
  const [expanded, setExpanded] = useState(false)
  const [highlightedId, setHighlightedId] = useState<string | null>(null)
  const sourceRefs = useRef<Record<string, HTMLLIElement | null>>({})
  const items = citationItems(sources)
  const citationIds = sources.map((source) => source.id)

  useEffect(() => {
    if (expanded && highlightedId) sourceRefs.current[highlightedId]?.scrollIntoView({ block: 'nearest' })
  }, [expanded, highlightedId])

  if (!sources.length) return null

  function showCitation(referenceId: string) {
    const sourceId = resolveCitationId(referenceId, sources)
    if (!sourceId) return
    setHighlightedId(sourceId)
    setExpanded(true)
  }

  return (
    <>
      <MarkdownAnswer text={text} citationIds={citationIds} onCitationClick={showCitation} />
      <section className="source-citations" aria-label="Answer sources">
        <button
          className="source-toggle"
          type="button"
          aria-expanded={expanded}
          onClick={() => setExpanded((value) => !value)}
        >
          <span>Sources · {items.length} {items.length === 1 ? 'reference' : 'references'}</span>
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" /></svg>
        </button>
        {expanded && (
          <ol>
            {items.map((item) => (
              <li
                key={item.key}
                ref={(element) => { sourceRefs.current[item.id] = element }}
                className={highlightedId === item.id ? 'source-highlighted' : undefined}
              >
                <strong>{item.id}</strong>
                <span className="source-reference">
                  {item.href ? (
                    <a href={item.href} target="_blank" rel="noreferrer">{item.label}</a>
                  ) : item.label}
                </span>
                <span className="source-document-id">Document {item.documentId}</span>
                {item.snippet && <p className="source-snippet">{item.snippet}</p>}
              </li>
            ))}
          </ol>
        )}
      </section>
    </>
  )
}
