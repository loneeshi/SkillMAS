import type { SkillManager } from "../skill/manager.js"
import type { VerifiedTrace } from "./retention.js"

export interface ValidationPoolOptions {
  minUses?: number
  minSuccessRate?: number
}

export interface ValidationPoolDecision {
  skillId: string
  action: "promote" | "hold" | "disable"
  uses: number
  successes: number
  successRate: number
  reason: string
}

export class ValidationPool {
  private skillManager: SkillManager
  private minUses: number
  private minSuccessRate: number

  constructor(skillManager: SkillManager, options?: ValidationPoolOptions) {
    this.skillManager = skillManager
    this.minUses = options?.minUses ?? 3
    this.minSuccessRate = options?.minSuccessRate ?? 0.6
  }

  async add(skillId: string): Promise<boolean> {
    const skill = await this.skillManager.get(skillId)
    if (!skill) return false
    if (skill.spec.status === "disabled") return false
    return this.skillManager.update(skillId, { status: "shadow" })
  }

  async addMany(skillIds: string[]): Promise<string[]> {
    const added: string[] = []
    for (const skillId of skillIds) {
      if (await this.add(skillId)) added.push(skillId)
    }
    return added
  }

  async evaluate(traces: VerifiedTrace[]): Promise<ValidationPoolDecision[]> {
    const shadowSkills = (await this.skillManager.list())
      .filter((skill) => skill.status === "shadow")

    const decisions: ValidationPoolDecision[] = []
    for (const skill of shadowSkills) {
      const evidence = traces.filter((trace) => traceUsesSkill(trace, skill.id))
      const uses = evidence.length
      const successes = evidence.filter((trace) => trace.success).length
      const successRate = uses > 0 ? successes / uses : 0

      if (uses < this.minUses) {
        decisions.push({
          skillId: skill.id,
          action: "hold",
          uses,
          successes,
          successRate,
          reason: `needs ${this.minUses - uses} more verified use(s)`,
        })
        continue
      }

      if (successRate >= this.minSuccessRate) {
        await this.skillManager.update(skill.id, { status: "active" })
        decisions.push({
          skillId: skill.id,
          action: "promote",
          uses,
          successes,
          successRate,
          reason: `shadow skill passed validation (${successRate.toFixed(2)})`,
        })
        continue
      }

      decisions.push({
        skillId: skill.id,
        action: "hold",
        uses,
        successes,
        successRate,
        reason: `success rate ${successRate.toFixed(2)} below ${this.minSuccessRate.toFixed(2)}`,
      })
    }

    return decisions
  }
}

function traceUsesSkill(trace: VerifiedTrace, skillId: string): boolean {
  for (const skillIds of Object.values(trace.usedSkills ?? {})) {
    if (skillIds.includes(skillId)) return true
  }
  return false
}
