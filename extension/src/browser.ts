/**
 * Cross-browser API abstraction for PageGlance extension.
 *
 * Provides a unified `browser` API that works in both Chrome and Firefox.
 * Firefox natively supports `browser.*` Promise-based APIs. Chrome uses
 * `chrome.*` (mostly callback-based, with some Promise support in MV3).
 *
 * Usage: import { browser } from './browser'
 */

type StorageLocal = {
  get: (keys: string | string[]) => Promise<Record<string, unknown>>
  set: (items: Record<string, unknown>) => Promise<void>
  remove: (keys: string | string[]) => Promise<void>
}

type RuntimeOnMessage = {
  addListener: (
    callback: (message: unknown, sender: unknown, sendResponse: (response: unknown) => void) => void,
  ) => void
}

type Runtime = {
  getURL: (path: string) => string
  onMessage: RuntimeOnMessage
}

type Tab = { id?: number; url?: string; title?: string }

type Tabs = {
  query: (queryInfo: { active: boolean; currentWindow: boolean }) => Promise<Tab[]>
  create: (createProperties: { url: string; active?: boolean }) => Promise<Tab>
  get: (tabId: number) => Promise<Tab>
  sendMessage: <T>(tabId: number, message: unknown) => Promise<T>
}

type Scripting = {
  executeScript: (injection: { target: { tabId: number }; files: string[] }) => Promise<unknown[]>
}

type ChromeLike = {
  storage: { local: StorageLocal }
  runtime: Runtime
  tabs: Tabs
  scripting: Scripting
}

function getBrowserGlobal(name: string): ChromeLike | undefined {
  const value = (globalThis as Record<string, unknown>)[name]
  return typeof value !== 'undefined' ? (value as ChromeLike) : undefined
}

/**
 * Resolve the browser API object at each call site rather than snapshotting it
 * at module load. In tests the `chrome`/`browser` global is installed after
 * the (hoisted) ESM imports run, so eager capture would see `undefined`; lazy
 * resolution lets mocks installed before any call take effect.
 */
function getBackend(): ChromeLike {
  const firefoxLike = getBrowserGlobal('browser')
  const chromeLike = getBrowserGlobal('chrome')
  const backend = firefoxLike ?? chromeLike
  if (!backend) {
    throw new Error('Neither the browser nor chrome API is available in this environment.')
  }
  return backend
}

export const browser = {
  storage: {
    local: {
      get: (keys: string | string[]) => getBackend().storage.local.get(keys),
      set: (items: Record<string, unknown>) => getBackend().storage.local.set(items),
      remove: (keys: string | string[]) => getBackend().storage.local.remove(keys),
    },
  },
  runtime: {
    getURL: (path: string) => getBackend().runtime.getURL(path),
    onMessage: {
      addListener: (
        callback: (message: unknown, sender: unknown, sendResponse: (response: unknown) => void) => void,
      ) => {
        getBackend().runtime.onMessage.addListener(callback)
      },
    },
  },
  tabs: {
    query: (queryInfo: { active: boolean; currentWindow: boolean }) => getBackend().tabs.query(queryInfo),
    create: (createProperties: { url: string; active?: boolean }) =>
      getBackend().tabs.create(createProperties),
    get: (tabId: number) => getBackend().tabs.get(tabId),
    sendMessage: <T>(tabId: number, message: unknown) => getBackend().tabs.sendMessage<T>(tabId, message),
  },
  scripting: {
    executeScript: (injection: { target: { tabId: number }; files: string[] }) =>
      getBackend().scripting.executeScript(injection),
  },
}

export const isFirefox = typeof browser !== 'undefined'
