/**
 * SkillManager v2 — manages skills in both legacy (.md) and directory (SKILL.md) formats.
 *
 * Supports:
 *   - Legacy: skills/<domain>/<name>.md (single file, frontmatter + content)
 *   - Directory: skills/<domain>/<name>/SKILL.md + supporting files
 *   - Dynamic context injection: !`command` replaced with shell output
 *   - Argument substitution: $ARGUMENTS, $ARGUMENTS[N], $N
 *   - Supporting file resolution for directory-based skills
 */

import { readFile, writeFile, unlink, readdir, mkdir, rmdir, stat } from "node:fs/promises"
import { join, dirname, relative, resolve } from "node:path"
import { execSync } from "node:child_process"
import matter from "gray-matter"
import { SkillSpecSchema } from "../spec/skill.js"
import type { SkillSpec, SkillSpecInput } from "../spec/skill.js"
import { validateSkillContent } from "./content-validator.js"
import type { SkillValidationResult } from "./content-validator.js"
import { isSeedSkill } from "./seed-skill.js"

export interface ResolvedSkill {
  spec: SkillSpec
  content: string
  sourcePath?: string
  dir?: string
  supportingFiles?: string[]
}

export interface ResolveFamilyAliasOptions {
  agentId?: string
  activeOnly?: boolean
  includeShadow?: boolean
}

export class SkillManager {
  constructor(private skillsDir: string) {}

  async create(spec: SkillSpecInput, content: string): Promise<void> {
    const validated = SkillSpecSchema.parse(spec)
    const filePath = this.resolveLegacyPath(validated.id)

    // Protect manually-authored seed skills from auto-generated overwrites
    const existing = await this.get(validated.id)
    if (existing) {
      const newGen = validated.generatedBy
      if (isSeedSkill(existing.spec) && newGen && newGen !== "manual") {
        console.warn(
          `[SkillManager] Refusing to overwrite seed skill "${validated.id}" (generatedBy: ${existing.spec.generatedBy ?? "manual"}) with auto-generated content (generatedBy: ${newGen})`,
        )
        return
      }
    }

    const validation = validateSkillContent(validated, content)
    let finalContent = content

    if (validation.errors.length > 0) {
      // Reject auto-generated skills that fail validation — don't pollute the skill bank
      if (validated.generatedBy && validated.generatedBy !== "manual") {
        throw new Error(
          `[SkillManager] Rejecting "${validated.id}" — failed validation: ${validation.errors.join("; ")}`,
        )
      }
      console.warn(
        `[SkillManager] Validation errors for "${validated.id}" (saving anyway):`,
        validation.errors,
      )
      const issueLines = [
        ...validation.errors.map((e) => `  ERROR: ${e}`),
        ...validation.warnings.map((w) => `  WARNING: ${w}`),
      ]
      finalContent = `<!--\nVALIDATION ISSUES:\n${issueLines.join("\n")}\n-->\n\n${content}`
    } else if (validation.warnings.length > 0) {
      console.debug(
        `[SkillManager] Validation warnings for "${validated.id}":`,
        validation.warnings,
      )
    }

    await mkdir(dirname(filePath), { recursive: true })

    const { id: _, ...frontmatterFields } = validated
    const cleaned = this.cleanFrontmatter(frontmatterFields)
    const md = matter.stringify(finalContent, cleaned)
    await writeFile(filePath, md, "utf-8")
  }

  async createDirectory(spec: SkillSpecInput, content: string, supportingFiles?: Record<string, string>): Promise<void> {
    const validated = SkillSpecSchema.parse(spec)
    const skillDir = join(this.skillsDir, validated.id)
    await mkdir(skillDir, { recursive: true })

    const { id: _, ...frontmatterFields } = validated
    const cleaned = this.cleanFrontmatter(frontmatterFields)
    const md = matter.stringify(content, cleaned)
    await writeFile(join(skillDir, "SKILL.md"), md, "utf-8")

    if (supportingFiles) {
      for (const [filename, fileContent] of Object.entries(supportingFiles)) {
        const filePath = join(skillDir, filename)
        await mkdir(dirname(filePath), { recursive: true })
        await writeFile(filePath, fileContent, "utf-8")
      }
    }
  }

  async get(skillId: string): Promise<ResolvedSkill | undefined> {
    const dirSkill = await this.getFromDirectory(skillId)
    if (dirSkill) return dirSkill

    return this.getFromLegacy(skillId)
  }

  private async getFromDirectory(skillId: string): Promise<ResolvedSkill | undefined> {
    const skillDir = join(this.skillsDir, skillId)
    const skillMd = join(skillDir, "SKILL.md")

    let raw: string
    try {
      raw = await readFile(skillMd, "utf-8")
    } catch {
      return undefined
    }

    const parsed = matter(raw)
    const spec = SkillSpecSchema.parse({ ...parsed.data, id: skillId })

    const supportingFiles: string[] = []
    await this.listSupportingFiles(skillDir, skillDir, supportingFiles)

    return { spec, content: parsed.content.trim(), sourcePath: skillMd, dir: skillDir, supportingFiles }
  }

  private async getFromLegacy(skillId: string): Promise<ResolvedSkill | undefined> {
    const filePath = this.resolveLegacyPath(skillId)

    let raw: string
    try {
      raw = await readFile(filePath, "utf-8")
    } catch {
      return undefined
    }

    const parsed = matter(raw)
    const spec = SkillSpecSchema.parse({ ...parsed.data, id: skillId })
    return { spec, content: parsed.content.trim(), sourcePath: filePath }
  }

  async update(
    skillId: string,
    updates: Partial<SkillSpec>,
    newContent?: string,
  ): Promise<boolean> {
    const existing = await this.get(skillId)
    if (!existing) return false

    const merged: SkillSpec = { ...existing.spec, ...updates, id: skillId }
    const content = newContent ?? existing.content
    const validated = SkillSpecSchema.parse(merged)

    const isDir = existing.dir !== undefined
    const filePath = isDir
      ? join(existing.dir!, "SKILL.md")
      : this.resolveLegacyPath(skillId)

    const { id: _, ...frontmatterFields } = validated
    const cleaned = this.cleanFrontmatter(frontmatterFields)
    const md = matter.stringify(content, cleaned)
    await writeFile(filePath, md, "utf-8")

    return true
  }

  async remove(skillId: string): Promise<boolean> {
    const existing = await this.get(skillId)
    if (!existing) return false

    if (existing.dir) {
      const { rm } = await import("node:fs/promises")
      await rm(existing.dir, { recursive: true, force: true })
      return true
    }

    const filePath = this.resolveLegacyPath(skillId)
    try {
      await unlink(filePath)
    } catch {
      return false
    }

    let dir = dirname(filePath)
    while (dir !== this.skillsDir && dir.startsWith(this.skillsDir)) {
      try {
        const entries = await readdir(dir)
        if (entries.length > 0) break
        await rmdir(dir)
        dir = dirname(dir)
      } catch {
        break
      }
    }

    return true
  }

  async list(filter?: { tags?: string[]; type?: string }): Promise<SkillSpec[]> {
    const specs: SkillSpec[] = []
    await this.scanDir(this.skillsDir, specs)

    let result = specs

    if (filter?.tags && filter.tags.length > 0) {
      const tagSet = new Set(filter.tags)
      result = result.filter((s) => s.tags.some((t) => tagSet.has(t)))
    }

    if (filter?.type) {
      result = result.filter((s) => s.type === filter.type)
    }

    return result
  }

  async listByFamily(
    family: string,
    options?: ResolveFamilyAliasOptions,
  ): Promise<SkillSpec[]> {
    const skills = await this.list()
    return skills.filter((skill) => {
      if (skill.family !== family) return false
      if (options?.activeOnly && skill.status !== "active") return false
      if (!options?.includeShadow && skill.status === "shadow") return false
      if (options?.agentId && skill.agent && skill.agent !== options.agentId) return false
      return true
    })
  }

  async resolveFamilyAlias(
    family: string,
    options?: ResolveFamilyAliasOptions,
  ): Promise<ResolvedSkill | undefined> {
    const skills = await this.listByFamily(family, options)
    const sorted = sortFamilyCandidates(skills, options?.agentId)
    for (const spec of sorted) {
      const resolved = await this.get(spec.id)
      if (resolved) return resolved
    }
    return undefined
  }

  async resolveFamilyAliases(
    families: string[],
    options?: ResolveFamilyAliasOptions,
  ): Promise<ResolvedSkill[]> {
    const resolved: ResolvedSkill[] = []
    for (const family of families) {
      const skill = await this.resolveFamilyAlias(family, options)
      if (skill) resolved.push(skill)
    }
    return resolved
  }

  async resolve(skillIds: string[]): Promise<ResolvedSkill[]> {
    const results: ResolvedSkill[] = []

    for (const id of skillIds) {
      const skill = await this.get(id)
      if (skill) {
        results.push(skill)
      } else {
        console.warn(`Skill not found: ${id}`)
      }
    }

    return results
  }

  async buildSkillPrompt(
    skillIds: string[],
    args?: { arguments?: string; argumentList?: string[] },
  ): Promise<string> {
    const skills = await this.resolve(skillIds)

    return skills
      .map(({ spec, content, dir, supportingFiles }) => {
        let processed = content

        processed = this.processArgumentSubstitution(processed, args)
        processed = this.processDynamicInjection(processed, dir)

        const header = `## ${spec.id}`
        const meta: string[] = []
        if (spec.type !== "reference") meta.push(`Type: ${spec.type}`)
        if (spec.whenToUse) meta.push(`When to use: ${spec.whenToUse}`)
        if (supportingFiles && supportingFiles.length > 0) {
          meta.push(`Supporting files: ${supportingFiles.join(", ")}`)
        }
        const metaBlock = meta.length > 0 ? meta.join("\n") + "\n\n" : ""

        const whenToUseLine = spec.when_to_use
          ? `**When to use**: ${spec.when_to_use}\n\n`
          : ""

        return `${header}\n${whenToUseLine}${metaBlock}${processed}`
      })
      .join("\n\n---\n\n")
  }

  async readSupportingFile(skillId: string, filename: string): Promise<string | undefined> {
    const skill = await this.get(skillId)
    if (!skill?.dir) return undefined

    try {
      return await readFile(join(skill.dir, filename), "utf-8")
    } catch {
      return undefined
    }
  }

  processArgumentSubstitution(
    content: string,
    args?: { arguments?: string; argumentList?: string[] },
  ): string {
    if (!args) return content

    let result = content

    if (args.arguments !== undefined) {
      result = result.replace(/\$ARGUMENTS/g, args.arguments)
    }

    if (args.argumentList) {
      for (let i = 0; i < args.argumentList.length; i++) {
        result = result.replace(new RegExp(`\\$ARGUMENTS\\[${i}\\]`, "g"), args.argumentList[i])
        result = result.replace(new RegExp(`\\$${i}(?![0-9])`, "g"), args.argumentList[i])
      }
    }

    if (args.arguments !== undefined && !content.includes("$ARGUMENTS")) {
      result += `\n\nARGUMENTS: ${args.arguments}`
    }

    return result
  }

  processDynamicInjection(content: string, skillDir?: string): string {
    return content.replace(/!\`([^`]+)\`/g, (_match, command: string) => {
      try {
        const cwd = skillDir ?? this.skillsDir
        const output = execSync(command, {
          cwd,
          encoding: "utf-8",
          timeout: 10_000,
          stdio: ["pipe", "pipe", "pipe"],
        })
        return output.trim()
      } catch (err) {
        return `[Command failed: ${command}]`
      }
    })
  }

  async validate(skillId: string): Promise<{ valid: boolean; errors: string[]; warnings?: string[] }> {
    const errors: string[] = []
    const warnings: string[] = []

    const skill = await this.get(skillId)
    if (!skill) {
      return { valid: false, errors: [`Skill not found: ${skillId}`] }
    }

    const result = SkillSpecSchema.safeParse(skill.spec)
    if (!result.success) {
      for (const issue of result.error.issues) {
        errors.push(`${issue.path.join(".")}: ${issue.message}`)
      }
      return { valid: false, errors }
    }

    if (skill.spec.context === "fork" && !skill.spec.agent) {
      errors.push("Skills with context:fork should specify an agent type")
    }

    if (skill.spec.arguments && skill.content.includes("$ARGUMENTS")) {
      const hasAll = skill.spec.arguments
        .filter((a) => a.required)
        .every((_, i) => skill.content.includes(`$${i}`) || skill.content.includes(`$ARGUMENTS[${i}]`))
      if (!hasAll) {
        errors.push("Required arguments declared but not all referenced in content")
      }
    }

    const contentValidation = validateSkillContent(skill.spec, skill.content)
    errors.push(...contentValidation.errors)
    warnings.push(...contentValidation.warnings)

    return { valid: errors.length === 0, errors, warnings }
  }

  async validateContent(skillId: string): Promise<SkillValidationResult | null> {
    const skill = await this.get(skillId)
    if (!skill) return null

    return validateSkillContent(skill.spec, skill.content)
  }

  private resolveLegacyPath(skillId: string): string {
    return join(this.skillsDir, `${skillId}.md`)
  }

  private async listSupportingFiles(baseDir: string, dir: string, results: string[]): Promise<void> {
    let entries: string[]
    try {
      entries = await readdir(dir)
    } catch {
      return
    }

    for (const entry of entries) {
      if (entry === "SKILL.md") continue
      const fullPath = join(dir, entry)
      const info = await stat(fullPath)

      if (info.isDirectory()) {
        await this.listSupportingFiles(baseDir, fullPath, results)
      } else {
        results.push(relative(baseDir, fullPath))
      }
    }
  }

  private async scanDir(dir: string, results: SkillSpec[]): Promise<void> {
    let entries: string[]
    try {
      entries = await readdir(dir)
    } catch {
      return
    }

    for (const entry of entries) {
      const fullPath = join(dir, entry)
      const info = await stat(fullPath)

      if (info.isDirectory()) {
        const skillMd = join(fullPath, "SKILL.md")
        try {
          const raw = await readFile(skillMd, "utf-8")
          const parsed = matter(raw)
          const skillId = relative(this.skillsDir, fullPath).replace(/\\/g, "/")
          const spec = SkillSpecSchema.parse({ ...parsed.data, id: skillId })
          results.push(spec)
        } catch {
          await this.scanDir(fullPath, results)
        }
      } else if (entry.endsWith(".md")) {
        try {
          const raw = await readFile(fullPath, "utf-8")
          const parsed = matter(raw)
          const skillId = relative(this.skillsDir, fullPath).replace(/\.md$/, "").replace(/\\/g, "/")
          const spec = SkillSpecSchema.parse({ ...parsed.data, id: skillId })
          results.push(spec)
        } catch {
          // skip malformed
        }
      }
    }
  }

  private cleanFrontmatter(fields: Record<string, unknown>): Record<string, unknown> {
    const cleaned: Record<string, unknown> = {}
    for (const [key, value] of Object.entries(fields)) {
      if (value === undefined) continue
      if (Array.isArray(value) && value.length === 0) continue
      if (typeof value === "string" && value === "") continue
      if (key === "invocation") {
        const inv = value as { modelCanInvoke: boolean; userCanInvoke: boolean }
        if (inv.modelCanInvoke && inv.userCanInvoke) continue
      }
      if (key === "context" && value === "inline") continue
      if (key === "type" && value === "reference") continue
      cleaned[key] = value
    }
    return cleaned
  }
}

export function getDefaultSkillManager(baseDir?: string): SkillManager {
  const dir = baseDir ? join(baseDir, "skills") : join(process.cwd(), "skills")
  return new SkillManager(dir)
}

function sortFamilyCandidates(skills: SkillSpec[], agentId?: string): SkillSpec[] {
  return [...skills].sort((a, b) => {
    const aAgent = a.agent === agentId ? 1 : 0
    const bAgent = b.agent === agentId ? 1 : 0
    if (aAgent !== bAgent) return bAgent - aAgent

    const rankStatus = (status: SkillSpec["status"]): number => {
      if (status === "active") return 2
      if (status === "shadow") return 1
      return 0
    }
    const aStatus = rankStatus(a.status)
    const bStatus = rankStatus(b.status)
    if (aStatus !== bStatus) return bStatus - aStatus

    const rankComposition = (spec: SkillSpec): number => {
      if (spec.provenance?.strategy === "composed") return 2
      if (spec.provenance?.strategy === "distilled") return 1
      return 0
    }
    const aComposition = rankComposition(a)
    const bComposition = rankComposition(b)
    if (aComposition !== bComposition) return bComposition - aComposition

    const aManual = !a.generatedBy || a.generatedBy === "manual" ? 1 : 0
    const bManual = !b.generatedBy || b.generatedBy === "manual" ? 1 : 0
    if (aManual !== bManual) return bManual - aManual

    const aRound = a.provenance?.sourceRound ?? -1
    const bRound = b.provenance?.sourceRound ?? -1
    if (aRound !== bRound) return bRound - aRound

    return b.id.localeCompare(a.id)
  })
}
