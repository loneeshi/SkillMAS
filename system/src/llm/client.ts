/**
 * Lightweight OpenAI-compatible LLM client.
 *
 * Uses native fetch with optional proxy support via undici ProxyAgent.
 * Supports tool calling, streaming is intentionally omitted to keep
 * each agent's runtime footprint minimal.
 *
 * Retry strategy (three tiers):
 *   - Network errors (fetch failed, ECONNRESET, timeout): retry indefinitely
 *     with exponential backoff capped at 30s. Gives up after networkTimeoutMs
 *     (default 10 min) to prevent true deadlocks.
 *   - Rate-limit errors (429): retry up to maxRateLimitRetries (default 8),
 *     honouring the Retry-After header when present.
 *   - Server errors (5xx): retry up to maxRetries (default 5).
 *   - Client errors (4xx except 429), parse errors: NOT retried.
 */

import type { ChatMessage, ChatResponse, ChatOptions } from "./types.js"

export interface LLMClientOptions {
  apiKey: string
  baseURL?: string
  defaultModel?: string
  timeout?: number
  proxy?: string
  /** Max retry attempts for server (5xx) errors (default: 5) */
  maxRetries?: number
  /** Max retry attempts for rate-limit (429) errors (default: 8) */
  maxRateLimitRetries?: number
  /** Base delay in ms for exponential backoff (default: 1000) */
  retryBaseDelayMs?: number
  /** Max total wait time in ms for network errors before giving up (default: 600000 = 10 min) */
  networkTimeoutMs?: number
}

let _dispatcher: unknown | undefined

async function getProxyDispatcher(proxyUrl: string): Promise<unknown> {
  if (_dispatcher) return _dispatcher
  try {
    const { ProxyAgent } = await import("undici")
    _dispatcher = new ProxyAgent(proxyUrl)
    return _dispatcher
  } catch {
    return undefined
  }
}

function detectProxy(): string | undefined {
  return (
    process.env.https_proxy ||
    process.env.HTTPS_PROXY ||
    process.env.http_proxy ||
    process.env.HTTP_PROXY ||
    undefined
  )
}

function isNetworkError(err: unknown): boolean {
  if (err instanceof Error) {
    const msg = err.message.toLowerCase()
    if (msg.includes("fetch failed")) return true
    if (msg.includes("econnreset")) return true
    if (msg.includes("econnrefused")) return true
    if (msg.includes("etimedout")) return true
    if (msg.includes("connect timeout")) return true
    if (msg.includes("socket hang up")) return true
    if (msg.includes("network")) return true
    if (msg.includes("aborted") && !msg.includes("user")) return true
    const cause = (err as any).cause
    if (cause) return isNetworkError(cause)
  }
  return false
}

function isServerError(err: unknown): boolean {
  if (err instanceof Error) {
    if (/llm api 5\d\d/i.test(err.message)) return true
  }
  return false
}

function isRateLimitError(err: unknown): boolean {
  if (err instanceof Error) {
    const msg = err.message.toLowerCase()
    if (msg.includes("tool_calls") || msg.includes("tool messages responding")) return false
    if (/llm api 429/i.test(err.message)) return true
  }
  return false
}

function parseRetryAfterMs(err: unknown): number | undefined {
  if (err instanceof Error) {
    const match = /retry-after:\s*(\d+)/i.exec(err.message)
    if (match) return parseInt(match[1], 10) * 1000
  }
  return undefined
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms))
}

export class LLMClient {
  private apiKey: string
  private baseURL: string
  private defaultModel: string
  private timeout: number
  private proxy: string | undefined
  private maxRetries: number
  private maxRateLimitRetries: number
  private retryBaseDelayMs: number
  private networkTimeoutMs: number

  constructor(options: LLMClientOptions) {
    this.apiKey = options.apiKey
    this.baseURL = (options.baseURL ?? "https://api.openai.com/v1").replace(/\/+$/, "")
    this.defaultModel = options.defaultModel ?? "gpt-4o-mini"
    this.timeout = options.timeout ?? 120_000
    this.proxy = options.proxy ?? detectProxy()
    this.maxRetries = options.maxRetries ?? 5
    this.maxRateLimitRetries = options.maxRateLimitRetries ?? 8
    this.retryBaseDelayMs = options.retryBaseDelayMs ?? 1000
    this.networkTimeoutMs = options.networkTimeoutMs ?? 600_000
  }

  async chat(messages: ChatMessage[], options?: ChatOptions): Promise<ChatResponse> {
    let lastError: unknown
    let serverAttempts = 0
    let rateLimitAttempts = 0
    const networkStart = Date.now()
    let networkWarnedAt = 0

    for (let attempt = 0; ; attempt++) {
      try {
        return await this.chatOnce(messages, options)
      } catch (err) {
        lastError = err

        if (options?.signal?.aborted) {
          throw err
        }

        if (isNetworkError(err)) {
          const elapsed = Date.now() - networkStart
          if (elapsed >= this.networkTimeoutMs) {
            console.error(`[llm] Network down for ${Math.round(elapsed / 1000)}s, giving up.`)
            throw err
          }

          const baseDelay = Math.min(this.retryBaseDelayMs * Math.pow(2, attempt), 30_000)
          const jitter = Math.random() * baseDelay * 0.3
          const delay = Math.round(baseDelay + jitter)

          const now = Date.now()
          if (now - networkWarnedAt >= 60_000) {
            console.warn(`[llm] Network down (${Math.round(elapsed / 1000)}s), retrying every ${Math.round(baseDelay / 1000)}s — will wait up to ${Math.round((this.networkTimeoutMs - elapsed) / 1000)}s more`)
            networkWarnedAt = now
          }

          await sleep(delay)
          continue
        }

        if (isRateLimitError(err) && rateLimitAttempts < this.maxRateLimitRetries) {
          rateLimitAttempts++
          const retryAfterMs = parseRetryAfterMs(err)
          const baseDelay = retryAfterMs ?? Math.min(this.retryBaseDelayMs * Math.pow(2, rateLimitAttempts - 1), 60_000)
          const jitter = Math.random() * Math.min(baseDelay * 0.2, 5_000)
          const delay = Math.round(baseDelay + jitter)
          const errMsg = err instanceof Error ? err.message.slice(0, 120) : String(err)
          console.warn(`[llm] Rate limited, retry ${rateLimitAttempts}/${this.maxRateLimitRetries} after ${delay}ms — ${errMsg}`)
          await sleep(delay)
          continue
        }

        if (isServerError(err) && serverAttempts < this.maxRetries) {
          serverAttempts++
          const baseDelay = this.retryBaseDelayMs * Math.pow(2, serverAttempts - 1)
          const jitter = Math.random() * baseDelay * 0.5
          const delay = Math.round(baseDelay + jitter)
          const errMsg = err instanceof Error ? err.message.slice(0, 120) : String(err)
          console.warn(`[llm] Server error retry ${serverAttempts}/${this.maxRetries} after ${delay}ms — ${errMsg}`)
          await sleep(delay)
          continue
        }

        throw err
      }
    }
  }

  private async chatOnce(messages: ChatMessage[], options?: ChatOptions): Promise<ChatResponse> {
    const model = options?.model ?? this.defaultModel

    const body: Record<string, unknown> = {
      model,
      messages,
      temperature: options?.temperature ?? 0.7,
    }

    if (options?.maxTokens) body.max_tokens = options.maxTokens
    if (options?.tools?.length) {
      body.tools = options.tools
      body.tool_choice = options.toolChoice ?? "auto"
    }

    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.timeout)

    try {
      const fetchOptions: Record<string, unknown> = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify(body),
        signal: options?.signal ?? controller.signal,
      }

      if (this.proxy) {
        const dispatcher = await getProxyDispatcher(this.proxy)
        if (dispatcher) {
          fetchOptions.dispatcher = dispatcher
        }
      }

      const res = await fetch(`${this.baseURL}/chat/completions`, fetchOptions as RequestInit)

      if (!res.ok) {
        const text = await res.text().catch(() => "")
        if (res.status === 429) {
          const retryAfterHeader = res.headers.get("retry-after")
          const retryAfterSuffix = retryAfterHeader ? ` retry-after: ${retryAfterHeader}` : ""
          throw new Error(`LLM API ${res.status}: ${text.slice(0, 400)}${retryAfterSuffix}`)
        }
        throw new Error(`LLM API ${res.status}: ${text.slice(0, 500)}`)
      }

      const data = (await res.json()) as any
      const choice = data.choices?.[0]
      const message = choice?.message

      if (message?.refusal) {
        return {
          content: "",
          toolCalls: [],
          finishReason: choice?.finish_reason ?? "stop",
          refusal: message.refusal,
          usage: data.usage
            ? {
                promptTokens: data.usage.prompt_tokens ?? 0,
                completionTokens: data.usage.completion_tokens ?? 0,
                totalTokens: data.usage.total_tokens ?? 0,
                reasoningTokens: data.usage.completion_tokens_details?.reasoning_tokens,
              }
            : undefined,
        }
      }

      return {
        content: message?.content ?? "",
        toolCalls: message?.tool_calls ?? [],
        finishReason: choice?.finish_reason ?? "stop",
        refusal: null,
        usage: data.usage
          ? {
              promptTokens: data.usage.prompt_tokens ?? 0,
              completionTokens: data.usage.completion_tokens ?? 0,
              totalTokens: data.usage.total_tokens ?? 0,
              reasoningTokens: data.usage.completion_tokens_details?.reasoning_tokens,
            }
          : undefined,
      }
    } finally {
      clearTimeout(timer)
    }
  }
}
