interface ChromeTab {
  id?: number
  url?: string
  title?: string
}

interface ChromeApi {
  storage: {
    local: {
      get(keys: string | string[]): Promise<Record<string, unknown>>
      set(items: Record<string, unknown>): Promise<void>
      remove(keys: string | string[]): Promise<void>
    }
  }
  runtime: {
    getURL(path: string): string
    onMessage: {
      addListener(
        callback: (
          message: unknown,
          sender: unknown,
          sendResponse: (response: unknown) => void,
        ) => void,
      ): void
    }
  }
  tabs: {
    query(queryInfo: { active: boolean; currentWindow: boolean }): Promise<ChromeTab[]>
    create(createProperties: { url: string; active?: boolean }): Promise<ChromeTab>
    get(tabId: number): Promise<ChromeTab>
    sendMessage<T>(tabId: number, message: unknown): Promise<T>
  }
  scripting: {
    executeScript(injection: { target: { tabId: number }; files: string[] }): Promise<unknown[]>
  }
}

declare const chrome: ChromeApi

