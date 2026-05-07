/**
 * WorkspaceMerger — merges isolated worker baseDirs back into the main workspace.
 *
 * After parallel training runs, each worker has its own:
 *   - memory/*.md      (agent memories; legacy .jsonl can still appear)
 *   - agents/*.md      (potentially new agents from extension)
 *   - skills/**\/*.md   (potentially new skills from evolution)
 *
 * Merge strategy:
 *   - Memory:  Markdown episodic merge with exact-line dedupe, preserving main structural section
 *              (legacy .jsonl append fallback for agents without .md)
 *   - Agents:  Copy new files; on conflict the worker version WINS (it has injected skills etc.)
 *   - Skills:  Copy new files; on conflict the worker version WINS (it was evolved during training)
 */

import {
  readdir,
  readFile,
  appendFile,
  copyFile,
  stat,
  mkdir,
  writeFile,
} from "node:fs/promises"
import { join } from "node:path"
import { createHash } from "node:crypto"
interface MarkdownMemoryDocument {
  structural: string
  episodic: {
    lessons: string[]
    errors: string[]
    actions: string[]
  }
}

export interface MergeResult {
  memoryFiles: number
  memoryEntries: number
  agentsCopied: string[]
  agentsRenamed: string[]
  skillsCopied: string[]
  skillsDeduped: string[]
}

export class WorkspaceMerger {
  constructor(private mainDir: string) {}

  async mergeFrom(
    workerDir: string,
    workerId: string,
  ): Promise<MergeResult> {
    const result: MergeResult = {
      memoryFiles: 0,
      memoryEntries: 0,
      agentsCopied: [],
      agentsRenamed: [],
      skillsCopied: [],
      skillsDeduped: [],
    }

    await this.mergeMemory(workerDir, result)
    await this.mergeAgents(workerDir, workerId, result)
    await this.mergeSkills(workerDir, workerId, result)

    return result
  }

  private async mergeMemory(
    workerDir: string,
    result: MergeResult,
  ): Promise<void> {
    const workerMemDir = join(workerDir, "memory")
    const mainMemDir = join(this.mainDir, "memory")
    await mkdir(mainMemDir, { recursive: true })

    let files: string[]
    try {
      files = await readdir(workerMemDir)
    } catch {
      return
    }

    const workerMarkdownAgents = new Set<string>()
    for (const file of files) {
      if (!file.endsWith(".md")) continue
      workerMarkdownAgents.add(file.replace(/\.md$/, ""))
      await this.mergeMarkdownMemory(workerMemDir, mainMemDir, file, result)
    }

    for (const file of files) {
      if (!file.endsWith(".jsonl")) continue
      const agentId = file.replace(/\.jsonl$/, "")
      if (workerMarkdownAgents.has(agentId)) continue
      if (await this.fileExists(join(mainMemDir, `${agentId}.md`))) continue

      const workerPath = join(workerMemDir, file)
      const mainPath = join(mainMemDir, file)

      const content = await readFile(workerPath, "utf-8")
      const lines = content.trim().split("\n").filter(Boolean)
      if (lines.length === 0) continue

      await appendFile(mainPath, "\n" + lines.join("\n") + "\n", "utf-8")
      result.memoryFiles++
      result.memoryEntries += lines.length
    }
  }

  private async mergeMarkdownMemory(
    workerMemDir: string,
    mainMemDir: string,
    file: string,
    result: MergeResult,
  ): Promise<void> {
    const workerPath = join(workerMemDir, file)
    const mainPath = join(mainMemDir, file)

    const workerRaw = await readFile(workerPath, "utf-8")
    const workerDoc = parseMarkdownMemoryDocument(workerRaw)
    const workerEntries = countMarkdownEpisodicEntries(workerRaw)

    let mainRaw: string | null = null
    try {
      mainRaw = await readFile(mainPath, "utf-8")
    } catch {
      // main file does not exist yet
    }

    if (mainRaw === null) {
      await copyFile(workerPath, mainPath)
      result.memoryFiles++
      result.memoryEntries += workerEntries
      return
    }

    const mainDoc = parseMarkdownMemoryDocument(mainRaw)
    const mergedLessons = this.mergeUniqueLines(mainDoc.episodic.lessons, workerDoc.episodic.lessons)
    const mergedErrors = this.mergeUniqueLines(mainDoc.episodic.errors, workerDoc.episodic.errors)
    const mergedActions = this.mergeUniqueLines(mainDoc.episodic.actions, workerDoc.episodic.actions)

    const addedCount =
      (mergedLessons.length - mainDoc.episodic.lessons.length) +
      (mergedErrors.length - mainDoc.episodic.errors.length) +
      (mergedActions.length - mainDoc.episodic.actions.length)

    if (addedCount <= 0) return

    const mergedDoc = {
      structural: mainDoc.structural,
      episodic: {
        lessons: mergedLessons,
        errors: mergedErrors,
        actions: mergedActions,
      },
    }

    await writeFile(mainPath, formatMarkdownMemoryDocument(mergedDoc), "utf-8")
    result.memoryFiles++
    result.memoryEntries += addedCount
  }

  private async mergeAgents(
    workerDir: string,
    workerId: string,
    result: MergeResult,
  ): Promise<void> {
    const workerAgentsDir = join(workerDir, "agents")
    const mainAgentsDir = join(this.mainDir, "agents")
    await mkdir(mainAgentsDir, { recursive: true })

    let files: string[]
    try {
      files = await readdir(workerAgentsDir)
    } catch {
      return
    }

    const mainFiles = new Set<string>()
    try {
      const existing = await readdir(mainAgentsDir)
      for (const f of existing) mainFiles.add(f)
    } catch {
      // empty
    }

    for (const file of files) {
      if (!file.endsWith(".md")) continue
      const srcPath = join(workerAgentsDir, file)

      if (!mainFiles.has(file)) {
        await copyFile(srcPath, join(mainAgentsDir, file))
        result.agentsCopied.push(file)
        continue
      }

      const srcContent = await readFile(srcPath, "utf-8")
      const mainContent = await readFile(
        join(mainAgentsDir, file),
        "utf-8",
      )

      if (this.hash(srcContent) === this.hash(mainContent)) {
        continue
      }

      await copyFile(srcPath, join(mainAgentsDir, file))
      result.agentsCopied.push(`${file} (updated by ${workerId})`)
    }
  }

  private async mergeSkills(
    workerDir: string,
    workerId: string,
    result: MergeResult,
  ): Promise<void> {
    const workerSkillsDir = join(workerDir, "skills")
    const mainSkillsDir = join(this.mainDir, "skills")
    await mkdir(mainSkillsDir, { recursive: true })

    const workerFiles = await this.walkDir(workerSkillsDir)

    for (const relPath of workerFiles) {
      if (!relPath.endsWith(".md")) continue
      const srcPath = join(workerSkillsDir, relPath)
      const destPath = join(mainSkillsDir, relPath)

      let destExists = false
      try {
        await stat(destPath)
        destExists = true
      } catch {
        // doesn't exist
      }

      if (!destExists) {
        await mkdir(join(mainSkillsDir, join(relPath, "..")), {
          recursive: true,
        })
        await copyFile(srcPath, destPath)
        result.skillsCopied.push(relPath)
        continue
      }

      const srcContent = await readFile(srcPath, "utf-8")
      const destContent = await readFile(destPath, "utf-8")

      if (this.hash(srcContent) === this.hash(destContent)) {
        result.skillsDeduped.push(relPath)
        continue
      }

      await copyFile(srcPath, destPath)
      result.skillsCopied.push(`${relPath} (updated by ${workerId})`)
    }
  }

  private async walkDir(dir: string): Promise<string[]> {
    const results: string[] = []

    let entries: string[]
    try {
      entries = await readdir(dir)
    } catch {
      return results
    }

    for (const entry of entries) {
      const fullPath = join(dir, entry)
      const info = await stat(fullPath)

      if (info.isDirectory()) {
        const sub = await this.walkDir(fullPath)
        for (const s of sub) {
          results.push(join(entry, s))
        }
      } else {
        results.push(entry)
      }
    }

    return results
  }

  private mergeUniqueLines(base: string[], incoming: string[]): string[] {
    const seen = new Set(base.map((line) => line.trim()))
    const merged = [...base]

    for (const line of incoming) {
      const normalized = line.trim()
      if (!normalized) continue
      if (seen.has(normalized)) continue
      seen.add(normalized)
      merged.push(normalized)
    }

    return merged
  }

  private async fileExists(path: string): Promise<boolean> {
    try {
      await stat(path)
      return true
    } catch {
      return false
    }
  }

  private hash(content: string): string {
    return createHash("md5").update(content).digest("hex")
  }
}

function parseMarkdownMemoryDocument(raw: string): MarkdownMemoryDocument {
  const section = (heading: string): string[] => {
    const match = raw.match(new RegExp(`^## ${heading}\\n([\\s\\S]*?)(?=^## |\\s*$)`, "m"))
    if (!match) return []
    return match[1]
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.startsWith("- "))
      .map((line) => line.slice(2).trim())
      .filter(Boolean)
  }

  return {
    structural: raw.split(/^## /m)[0].trim(),
    episodic: {
      lessons: section("Lessons"),
      errors: section("Errors"),
      actions: section("Actions"),
    },
  }
}

function formatMarkdownMemoryDocument(doc: MarkdownMemoryDocument): string {
  const list = (items: string[]): string => items.map((item) => `- ${item}`).join("\n")
  return [
    doc.structural,
    "## Lessons",
    list(doc.episodic.lessons),
    "## Errors",
    list(doc.episodic.errors),
    "## Actions",
    list(doc.episodic.actions),
    "",
  ].filter((part, index) => index === 0 || part.length > 0).join("\n\n")
}

function countMarkdownEpisodicEntries(raw: string): number {
  const doc = parseMarkdownMemoryDocument(raw)
  return doc.episodic.lessons.length + doc.episodic.errors.length + doc.episodic.actions.length
}
