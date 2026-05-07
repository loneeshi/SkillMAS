import fs from "fs/promises"
import path from "path"
import { type AgentSpec, AgentSpecSchema } from "../spec/agent.js"
import { parseAgentFile, stringifyAgentFile, type ParsedAgent } from "../parser/index.js"

export class AgentStore {
  constructor(private agentsDir: string) {}

  async create(spec: AgentSpec, prompt: string): Promise<string> {
    const filePath = this.resolvePath(spec.id)
    await fs.mkdir(path.dirname(filePath), { recursive: true })

    try {
      await fs.access(filePath)
      throw new Error(`Agent "${spec.id}" already exists at ${filePath}`)
    } catch (err: any) {
      if (err.code !== "ENOENT") throw err
    }

    const validated = AgentSpecSchema.parse(spec)
    const content = stringifyAgentFile(validated, prompt)
    await fs.writeFile(filePath, content, "utf-8")
    return filePath
  }

  async get(agentId: string): Promise<ParsedAgent | undefined> {
    const filePath = this.resolvePath(agentId)
    try {
      const raw = await fs.readFile(filePath, "utf-8")
      return parseAgentFile(raw, filePath)
    } catch (err: any) {
      if (err.code === "ENOENT") return undefined
      throw err
    }
  }

  async update(
    agentId: string,
    specUpdates?: Partial<AgentSpec>,
    newPrompt?: string,
  ): Promise<boolean> {
    const existing = await this.get(agentId)
    if (!existing) return false

    const mergedSpec = AgentSpecSchema.parse({
      ...existing.spec,
      ...specUpdates,
    })

    const prompt = newPrompt ?? existing.prompt
    const content = stringifyAgentFile(mergedSpec, prompt)

    await fs.writeFile(this.resolvePath(agentId), content, "utf-8")
    return true
  }

  async remove(agentId: string): Promise<boolean> {
    const filePath = this.resolvePath(agentId)
    try {
      await fs.unlink(filePath)
    } catch (err: any) {
      if (err.code === "ENOENT") return false
      throw err
    }

    const dir = path.dirname(filePath)
    if (dir !== this.agentsDir) {
      try {
        const entries = await fs.readdir(dir)
        if (entries.length === 0) await fs.rmdir(dir)
      } catch {}
    }

    return true
  }

  async list(): Promise<ParsedAgent[]> {
    const results: ParsedAgent[] = []
    await this.scanDir(this.agentsDir, results)
    return results
  }

  async exists(agentId: string): Promise<boolean> {
    try {
      await fs.access(this.resolvePath(agentId))
      return true
    } catch {
      return false
    }
  }

  private resolvePath(agentId: string): string {
    return path.join(this.agentsDir, `${agentId}.md`)
  }

  private async scanDir(dir: string, results: ParsedAgent[]): Promise<void> {
    let names: string[]
    try {
      names = await fs.readdir(dir)
    } catch {
      return
    }

    for (const name of names) {
      const fullPath = path.join(dir, name)
      const stat = await fs.stat(fullPath).catch(() => null)
      if (!stat) continue
      if (stat.isDirectory()) {
        await this.scanDir(fullPath, results)
      } else if (name.endsWith(".md")) {
        try {
          const raw = await fs.readFile(fullPath, "utf-8")
          results.push(parseAgentFile(raw, fullPath))
        } catch (err) {
          console.warn(`Failed to parse agent file ${fullPath}:`, err)
        }
      }
    }
  }
}
