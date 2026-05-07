export { LLMClient } from "./client.js"
export type { LLMClientOptions } from "./client.js"
export {
  EmbeddingClient,
  EmbeddingSimilarityScorer,
  LexicalSimilarityScorer,
  getDefaultSimilarityScorer,
  resolveProviderEnv,
  buildSkillSemanticText,
} from "./embedding.js"
export type {
  EmbeddingClientOptions,
  TextSimilarityScorer,
  ProviderEnvConfig,
} from "./embedding.js"
export type {
  ChatMessage,
  ChatResponse,
  ChatOptions,
  ToolCallRequest,
  ToolSchema,
  ToolMode,
  ToolChoice,
} from "./types.js"
