import type { AskResponse } from './messages'
import { parseAnswerSources } from './chat.ts'

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

/** Parse the FastAPI response and unwrap a model answer serialized as an answer envelope. */
export function parseAskResponse(payload: unknown): AskResponse {
  if (!isObject(payload) || typeof payload.answer !== 'string') {
    throw new Error('The backend returned an invalid answer.')
  }

  const answer = payload.answer
  try {
    const nested: unknown = JSON.parse(answer)
    if (isObject(nested) && typeof nested.answer === 'string') {
      return { answer: nested.answer }
    }
  } catch {
    // Ordinary answer text is the expected case; preserve it unchanged.
  }

  const sources = parseAnswerSources(payload.sources)
  return { answer, ...(sources ? { sources } : {}) }
}

function isStringRecord(value: unknown): value is Record<string, string> {
  return (
    typeof value === 'object' &&
    value !== null &&
    Object.values(value).every((field) => typeof field === 'string')
  )
}

/** Consume token and completion events from the backend's POST SSE response. */
export async function consumeAnswerStream(
  response: Response,
  onText: (text: string) => void,
  onSources?: (sources: NonNullable<AskResponse['sources']>) => void,
  onReplace?: (answer: string) => void,
): Promise<void> {
  if (!response.body) throw new Error('The backend returned an empty answer stream.')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let completed = false

  function processFrame(frame: string) {
    let event = 'message'
    const data: string[] = []
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith(':')) continue
      if (line.startsWith('event:')) event = line.slice(6).trim()
      if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
    }
    if (!data.length) return

    const payloadText = data.join('\n')
    if (event === 'token') {
      let payload: unknown
      try {
        payload = JSON.parse(payloadText)
      } catch {
        throw new Error('The backend sent an invalid answer stream.')
      }
      if (!isStringRecord(payload) || typeof payload.text !== 'string') {
        throw new Error('The backend sent an invalid answer stream.')
      }
      onText(payload.text)
    } else if (event === 'error') {
      let payload: unknown
      try {
        payload = JSON.parse(payloadText)
      } catch {
        throw new Error('The language model stream failed.')
      }
      throw new Error(
        isStringRecord(payload) && payload.detail
          ? payload.detail
          : 'The language model stream failed.',
      )
    } else if (event === 'sources') {
      let payload: unknown
      try {
        payload = JSON.parse(payloadText)
      } catch {
        throw new Error('The backend sent invalid source citations.')
      }
      if (!isObject(payload) || !Array.isArray(payload.sources)) {
        throw new Error('The backend sent invalid source citations.')
      }
      onSources?.(parseAnswerSources(payload.sources) ?? [])
    } else if (event === 'replace') {
      let payload: unknown
      try {
        payload = JSON.parse(payloadText)
      } catch {
        throw new Error('The backend sent an invalid answer replacement.')
      }
      if (!isObject(payload) || typeof payload.answer !== 'string') {
        throw new Error('The backend sent an invalid answer replacement.')
      }
      onReplace?.(payload.answer)
    } else if (event === 'done') {
      completed = true
    }
  }

  try {
    while (!completed) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      const frames = buffer.split(/\r?\n\r?\n/)
      buffer = frames.pop() ?? ''
      for (const frame of frames) {
        processFrame(frame)
        if (completed) break
      }
      if (done) {
        if (buffer.trim()) processFrame(buffer)
        break
      }
    }
  } finally {
    try {
      await reader.cancel()
    } catch {
      // The stream may already be closed or aborted.
    }
    reader.releaseLock()
  }

  if (!completed) throw new Error('The answer stream ended before completion.')
}
