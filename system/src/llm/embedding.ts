import type { SkillCatalogEntry } from "../skill/utilization"

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

export interface ProviderEnvConfig {
  apiKey?: string
  baseURL?: string
}

export function resolveProviderEnv(): ProviderEnvConfig {
  return {
    apiKey: process.env.HL_API_KEY || process.env.AZ_API_KEY || process.env.OPENAI_API_KEY || undefined,
    baseURL: process.env.HL_BASE_URL || process.env.AZ_BASE_URL || process.env.OPENAI_BASE_URL || process.env.OPENAI_API_BASE || undefined,
  }
}

export interface EmbeddingClientOptions {
  apiKey: string
  baseURL?: string
  model?: string
  timeout?: number
  proxy?: string
  maxBatchSize?: number
}

export interface TextSimilarityScorer {
  scoreMany(query: string, candidates: Record<string, string>): Promise<Record<string, number>>
}

export class EmbeddingClient {
  private apiKey: string
  private baseURL: string
  private model: string
  private timeout: number
  private proxy: string | undefined
  private maxBatchSize: number
  private cache = new Map<string, number[]>()

  constructor(options: EmbeddingClientOptions) {
    this.apiKey = options.apiKey
    this.baseURL = (options.baseURL ?? "https://api.openai.com/v1").replace(/\/+$/, "")
    this.model = options.model ?? "text-embedding-3-large"
    this.timeout = options.timeout ?? 120_000
    this.proxy = options.proxy ?? detectProxy()
    this.maxBatchSize = options.maxBatchSize ?? 64
  }

  async embedMany(inputs: string[]): Promise<number[][]> {
    const normalizedInputs = inputs.map((input) => input.trim())
    const missing = [...new Set(normalizedInputs.filter((input) => input.length > 0 && !this.cache.has(input)))]

    for (let i = 0; i < missing.length; i += this.maxBatchSize) {
      const batch = missing.slice(i, i + this.maxBatchSize)
      const vectors = await this.embedBatch(batch)
      for (let j = 0; j < batch.length; j++) {
        this.cache.set(batch[j], vectors[j] ?? [])
      }
    }

    return normalizedInputs.map((input) => this.cache.get(input) ?? [])
  }

  private async embedBatch(inputs: string[]): Promise<number[][]> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.timeout)

    try {
      const fetchOptions: Record<string, unknown> = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model: this.model,
          input: inputs,
        }),
        signal: controller.signal,
      }

      if (this.proxy) {
        const dispatcher = await getProxyDispatcher(this.proxy)
        if (dispatcher) fetchOptions.dispatcher = dispatcher
      }

      const res = await fetch(`${this.baseURL}/embeddings`, fetchOptions as RequestInit)
      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(`Embedding API ${res.status}: ${text.slice(0, 500)}`)
      }

      const data = await res.json() as {
        data?: Array<{ embedding?: number[]; index?: number }>
      }
      const rows = [...(data.data ?? [])].sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
      return rows.map((row) => row.embedding ?? [])
    } finally {
      clearTimeout(timer)
    }
  }
}

export class LexicalSimilarityScorer implements TextSimilarityScorer {
  async scoreMany(query: string, candidates: Record<string, string>): Promise<Record<string, number>> {
    const queryTokens = tokenize(query)
    const result: Record<string, number> = {}
    for (const [id, text] of Object.entries(candidates)) {
      const candidateTokens = tokenize(text)
      result[id] = jaccard(queryTokens, candidateTokens)
    }
    return result
  }
}

export class EmbeddingSimilarityScorer implements TextSimilarityScorer {
  private client: EmbeddingClient
  private fallback: TextSimilarityScorer
  private warnedFallback = false

  constructor(client: EmbeddingClient, fallback: TextSimilarityScorer = new LexicalSimilarityScorer()) {
    this.client = client
    this.fallback = fallback
  }

  async scoreMany(query: string, candidates: Record<string, string>): Promise<Record<string, number>> {
    const entries = Object.entries(candidates)
    if (entries.length === 0) return {}

    try {
      const vectors = await this.client.embedMany([query, ...entries.map(([, text]) => text)])
      const queryVector = vectors[0] ?? []
      const result: Record<string, number> = {}
      for (let i = 0; i < entries.length; i++) {
        result[entries[i][0]] = normalizeCosine(cosineSimilarity(queryVector, vectors[i + 1] ?? []))
      }
      return result
    } catch (error) {
      if (!this.warnedFallback) {
        this.warnedFallback = true
        console.warn(`[embedding] Falling back to lexical similarity: ${(error as Error).message}`)
      }
      return this.fallback.scoreMany(query, candidates)
    }
  }
}

let _defaultScorer: TextSimilarityScorer | null = null

export function getDefaultSimilarityScorer(): TextSimilarityScorer {
  if (_defaultScorer) return _defaultScorer

  const env = resolveProviderEnv()
  if (env.apiKey) {
    _defaultScorer = new EmbeddingSimilarityScorer(new EmbeddingClient({
      apiKey: env.apiKey,
      baseURL: env.baseURL,
      model: "text-embedding-3-large",
    }))
    return _defaultScorer
  }

  _defaultScorer = new LexicalSimilarityScorer()
  return _defaultScorer
}

export function buildSkillSemanticText(skill: SkillCatalogEntry): string {
  const family = "family" in skill && typeof (skill as { family?: unknown }).family === "string"
    ? (skill as { family?: string }).family
    : ""
  const agent = "agent" in skill && typeof (skill as { agent?: unknown }).agent === "string"
    ? (skill as { agent?: string }).agent
    : ""
  return [
    skill.description,
    skill.whenToUse,
    ...(skill.tags ?? []),
    ...(skill.allowedTools ?? []),
    ...(skill.toolHints ?? []),
    family,
    agent,
  ].filter(Boolean).join(" ")
}

function normalizeCosine(value: number): number {
  const shifted = (value + 1) / 2
  return Math.max(0, Math.min(1, shifted))
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  if (normA === 0 || normB === 0) return 0
  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

const STOPWORDS = new Set([
  "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
  "have", "has", "had", "do", "does", "did", "will", "would", "could",
  "should", "may", "might", "shall", "can", "need", "dare", "ought",
  "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
  "into", "through", "during", "before", "after", "and", "but", "or",
  "nor", "not", "so", "yet", "both", "either", "neither", "each",
  "this", "that", "these", "those", "it", "its", "when", "where",
  "how", "what", "which", "who", "whom", "if", "then", "than",
])

function tokenize(text: string): Set<string> {
  const words = text.toLowerCase().replace(/[^a-z0-9\s_-]/g, " ").split(/\s+/)
  const tokens = new Set<string>()
  for (const word of words) {
    if (word.length > 1 && !STOPWORDS.has(word)) tokens.add(word)
  }
  return tokens
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1
  if (a.size === 0 || b.size === 0) return 0
  let intersection = 0
  for (const token of a) {
    if (b.has(token)) intersection++
  }
  const union = new Set([...a, ...b]).size
  return union === 0 ? 0 : intersection / union
}
