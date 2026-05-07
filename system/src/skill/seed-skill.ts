type SeedSkillLike = {
  generatedBy?: string
  origin?: string
}

/** Seed skills are protected expert/manual skills, including legacy skills with no generatedBy. */
export function isSeedSkill(spec: SeedSkillLike): boolean {
  return !spec.generatedBy || spec.generatedBy === "manual" || spec.origin === "manual"
}

export function isAutoSkill(spec: SeedSkillLike): boolean {
  return !isSeedSkill(spec)
}
