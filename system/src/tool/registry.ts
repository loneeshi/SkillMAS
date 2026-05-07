import { z } from "zod"
import { readdir, readFile, writeFile, unlink } from "node:fs/promises"
import { join } from "node:path"
import type { ToolPolicy } from "../spec/agent.js"

export const ToolParamSchema = z.object({
  type: z.string().min(1),
  description: z.string(),
  required: z.boolean().default(false),
  items: z.record(z.unknown()).optional(),
})

export type ToolParam = z.infer<typeof ToolParamSchema>

export const ToolDefinitionSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  description: z.string(),
  category: z.string().min(1),
  parameters: z.record(z.string(), ToolParamSchema).default({}),
})

export type ToolDefinition = z.infer<typeof ToolDefinitionSchema>

export class ToolRegistry {
  constructor(private toolsDir: string) {}

  async loadAll(): Promise<Map<string, ToolDefinition>> {
    const tools = new Map<string, ToolDefinition>()
    let entries: string[]

    try {
      entries = await readdir(this.toolsDir)
    } catch {
      return tools
    }

    const jsonFiles = entries.filter((f) => f.endsWith(".json"))

    for (const file of jsonFiles) {
      try {
        const raw = await readFile(join(this.toolsDir, file), "utf-8")
        const json = JSON.parse(raw)
        const normalized = normalizeToolJSON(json, file)
        const tool = ToolDefinitionSchema.parse(normalized)
        tools.set(tool.id, tool)
      } catch {
        // skip malformed tool definitions
      }
    }

    return tools
  }

  async get(toolId: string): Promise<ToolDefinition | undefined> {
    const filePath = join(this.toolsDir, `${toolId}.json`)

    try {
      const raw = await readFile(filePath, "utf-8")
      return ToolDefinitionSchema.parse(JSON.parse(raw))
    } catch {
      return undefined
    }
  }

  async register(tool: ToolDefinition): Promise<void> {
    const validated = ToolDefinitionSchema.parse(tool)
    const filePath = join(this.toolsDir, `${validated.id}.json`)
    await writeFile(filePath, JSON.stringify(validated, null, 2) + "\n", "utf-8")
  }

  async unregister(toolId: string): Promise<boolean> {
    const filePath = join(this.toolsDir, `${toolId}.json`)

    try {
      await unlink(filePath)
      return true
    } catch {
      return false
    }
  }

  resolve(policy: ToolPolicy, available: Map<string, ToolDefinition>): ToolDefinition[] {
    let tools: ToolDefinition[]

    if (policy.allow.length === 0 || (policy.allow.length === 1 && policy.allow[0] === "*")) {
      // Empty allow list or wildcard "*" → include all available tools
      tools = [...available.values()]
    } else {
      tools = policy.allow
        .map((id) => available.get(id))
        .filter((t): t is ToolDefinition => t !== undefined)
    }

    if (policy.deny.length > 0) {
      const denySet = new Set(policy.deny)
      tools = tools.filter((t) => !denySet.has(t.id))
    }

    return tools
  }
}

export function getDefaultRegistry(baseDir?: string): ToolRegistry {
  const dir = baseDir ? join(baseDir, "tools") : join(process.cwd(), "tools")
  return new ToolRegistry(dir)
}

/**
 * Normalize a raw JSON tool file into the shape expected by ToolDefinitionSchema.
 *
 * Handles two formats:
 *   1. Native format: { id, name, category, description, parameters: { key: { type, description, required } } }
 *   2. JSON Schema format: { id, description, parameters: { type:"object", properties:{...}, required:[...] } }
 *
 * For format 2 we derive `name` from `id`, `category` from the prefix, and flatten `properties`.
 */
function normalizeToolJSON(
  json: Record<string, unknown>,
  filename: string,
): Record<string, unknown> {
  const id = (json.id as string) ?? filename.replace(/\.json$/, "")

  if (json.name && json.category) {
    return json
  }

  const name =
    (json.name as string) ??
    id
      .split(".")
      .pop()!
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase())

  const category =
    (json.category as string) ?? id.split(".")[0] ?? "general"

  const rawParams = json.parameters as Record<string, unknown> | undefined

  if (
    rawParams &&
    rawParams.type === "object" &&
    typeof rawParams.properties === "object"
  ) {
    const props = rawParams.properties as Record<
      string,
      Record<string, unknown>
    >
    const reqArr = Array.isArray(rawParams.required)
      ? (rawParams.required as string[])
      : []
    const reqSet = new Set(reqArr)

    const flat: Record<string, { type: string; description: string; required: boolean; items?: Record<string, unknown> }> = {}
    for (const [key, val] of Object.entries(props)) {
      const param: { type: string; description: string; required: boolean; items?: Record<string, unknown> } = {
        type: (val.type as string) ?? "string",
        description: (val.description as string) ?? "",
        required: reqSet.has(key),
      }
      // Preserve array items schema
      if (val.type === "array" && val.items) {
        param.items = val.items as Record<string, unknown>
      }
      flat[key] = param
    }

    return { id, name, category, description: json.description ?? "", parameters: flat }
  }

  return { ...json, id, name, category }
}
