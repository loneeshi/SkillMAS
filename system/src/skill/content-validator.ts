/**
 * Skill content validation — ensures auto-generated skills have meaningful structure.
 *
 * Philosophy: we don't force exact section names (LLMs vary in naming),
 * but we DO verify that essential patterns are present:
 *   - Skills of type "task" MUST have a procedure/workflow section
 *   - Skills of type "task"/"workflow" MUST have a mistakes/pitfalls section
 *   - All skills MUST have at least one ## heading
 *   - Skills should reference concrete tool calls, not just generic advice
 *
 * Validation is ADVISORY for manual skills, ENFORCED for auto-generated ones.
 */

import type { SkillSpec } from "../spec/skill.js"

// ═══ Validation Results ═══════════════════════════════════════════════

export interface SkillValidationResult {
  valid: boolean
  warnings: string[]
  errors: string[]
  sections: string[]
  metrics: {
    wordCount: number
    sectionCount: number
    hasToolReferences: boolean
    hasNumberedSteps: boolean
    hasBulletPoints: boolean
  }
}

// ═══ Required Patterns by Skill Type ══════════════════════════════════

interface ContentRequirement {
  pattern: RegExp
  description: string
  severity: "error" | "warning"
}

const COMMON_REQUIREMENTS: ContentRequirement[] = [
  {
    pattern: /^## /m,
    description: "At least one ## section heading",
    severity: "error",
  },
  {
    pattern: /\S{20,}/,
    description: "Non-trivial content (more than just headings)",
    severity: "error",
  },
]

const TASK_REQUIREMENTS: ContentRequirement[] = [
  {
    pattern: /(?:procedure|workflow|step-by-step|steps|how to|process|approach)/i,
    description: "Procedure/workflow section (task skills must describe a process)",
    severity: "error",
  },
  {
    pattern: /(?:\d+\.\s|\d+\)\s)/m,
    description: "Numbered steps (task skills should have ordered instructions)",
    severity: "warning",
  },
  {
    pattern: /(?:mistake|error|don't|avoid|pitfall|wrong|incorrect|never|warning)/i,
    description: "Common mistakes / pitfalls section",
    severity: "warning",
  },
]

const WORKFLOW_REQUIREMENTS: ContentRequirement[] = [
  {
    pattern: /(?:procedure|workflow|step|phase|stage)/i,
    description: "Workflow stages or phases described",
    severity: "error",
  },
  {
    pattern: /(?:mistake|error|don't|avoid|pitfall|recovery|fallback|fail)/i,
    description: "Error handling / recovery section",
    severity: "warning",
  },
]

const KNOWLEDGE_REQUIREMENTS: ContentRequirement[] = [
  {
    pattern: /(?:rule|pattern|heuristic|principle|guideline|strategy|observation)/i,
    description: "Rules or patterns described",
    severity: "warning",
  },
]

function getRequirements(type: string): ContentRequirement[] {
  const reqs = [...COMMON_REQUIREMENTS]
  switch (type) {
    case "task":
      reqs.push(...TASK_REQUIREMENTS)
      break
    case "workflow":
      reqs.push(...WORKFLOW_REQUIREMENTS)
      break
    case "knowledge":
      reqs.push(...KNOWLEDGE_REQUIREMENTS)
      break
  }
  return reqs
}

// ═══ Content Validation ═══════════════════════════════════════════════

const MIN_WORD_COUNT = 30
const MAX_WORD_COUNT = 2000

/**
 * Validate skill content structure.
 *
 * For auto-generated skills (generatedBy !== "manual"), errors are blocking.
 * For manual skills, all issues are downgraded to warnings.
 */
export function validateSkillContent(
  spec: SkillSpec,
  content: string,
): SkillValidationResult {
  const isManual = spec.generatedBy === "manual" || !spec.generatedBy
  const warnings: string[] = []
  const errors: string[] = []

  const sections = extractSections(content)
  const words = content.split(/\s+/).filter(Boolean)
  const hasToolRefs = /`?(?:env\.\w+|tool\.\w+)`?/i.test(content)
  const hasNumberedSteps = /^\s*\d+[\.\)]\s/m.test(content)
  const hasBulletPoints = /^\s*[-*]\s/m.test(content)

  if (words.length < MIN_WORD_COUNT) {
    errors.push(`Content too short: ${words.length} words (minimum ${MIN_WORD_COUNT})`)
  }
  if (words.length > MAX_WORD_COUNT) {
    warnings.push(`Content very long: ${words.length} words (consider trimming to <${MAX_WORD_COUNT})`)
  }

  const requirements = getRequirements(spec.type)
  for (const req of requirements) {
    if (!req.pattern.test(content)) {
      const severity = isManual ? "warning" : req.severity
      if (severity === "error") {
        errors.push(`Missing: ${req.description}`)
      } else {
        warnings.push(`Missing: ${req.description}`)
      }
    }
  }

  if (!hasToolRefs && (spec.type === "task" || spec.type === "workflow")) {
    const msg = "No tool references found — skills should reference concrete tool calls"
    if (isManual) warnings.push(msg)
    else warnings.push(msg)
  }

  return {
    valid: errors.length === 0,
    warnings,
    errors,
    sections,
    metrics: {
      wordCount: words.length,
      sectionCount: sections.length,
      hasToolReferences: hasToolRefs,
      hasNumberedSteps,
      hasBulletPoints,
    },
  }
}

// ═══ Section Extraction ═══════════════════════════════════════════════

/**
 * Extract ## section headings from markdown content.
 * Returns normalized heading text (lowercase, trimmed).
 */
function extractSections(content: string): string[] {
  const headings: string[] = []
  for (const line of content.split("\n")) {
    const match = line.match(/^#{1,3}\s+(.+)/)
    if (match) {
      headings.push(match[1].trim().toLowerCase())
    }
  }
  return headings
}

// ═══ Few-Shot Exemplars ═══════════════════════════════════════════════

/**
 * Get a few-shot exemplar skill for the given type.
 * These are used in designer prompts to anchor LLM generation.
 *
 * Exemplars are domain-AGNOSTIC — they show structure, not content.
 */
export function getSkillExemplar(type: "task" | "reference" | "knowledge" | "workflow"): string {
  return EXEMPLARS[type] ?? EXEMPLARS.reference
}

const EXEMPLARS: Record<string, string> = {
  task: `## Objective
Find the target object and transport it to the destination.

## Procedure
1. Identify the target object and destination from the task description
2. Use observation tools to check current surroundings for the target
3. If not visible, navigate to high-probability locations systematically
4. Once found, pick up the target using the appropriate interaction tool
5. Navigate to the destination location
6. Place the object at the destination

## Critical Rules
- Always read observations carefully — available objects are listed in text
- Check containers (they may need to be opened first)
- Track visited locations to avoid redundant searches
- If an action fails, try a different approach before retrying the same one

## Common Mistakes
- Trying to interact with objects that aren't in the current location
- Forgetting to open containers before looking inside
- Revisiting the same locations without a clear reason
- Not reading the observation text — skipping to the next action

## Recovery
- If stuck in a loop: stop, observe, and try a completely different location
- If an object can't be found: expand search to less obvious locations
- If an action returns an error: verify you're in the right location first`,

  reference: `## Overview
Efficient strategies for locating objects in the environment.

## Search Priority by Category
- **Kitchen items** → kitchen surfaces, appliances, storage
- **Personal items** → bedroom furniture, desk areas
- **Cleaning supplies** → bathroom, utility areas

## Search Efficiency Rules
- Check the most likely locations first (reduces average steps)
- Always read observation output — don't skip ahead
- Open closed containers before assuming they're empty
- Mark locations as "visited" to avoid backtracking

## Location Heuristics
| Object Type | Primary Locations | Secondary |
|------------|-------------------|-----------|
| Food | Counters, fridge, table | Cabinet, shelf |
| Tools | Drawer, shelf, desk | Counter |
| Containers | Counter, table, shelf | Cabinet |`,

  knowledge: `## Learned Patterns
- Actions that return "nothing happens" indicate a precondition failure
- Objects have stable locations across episodes of the same type
- Appliance operations require the object to be held first

## Performance Rules
- Optimal episode length for simple tasks: 8-12 steps
- Optimal episode length for multi-phase tasks: 15-20 steps
- More than 25 steps usually indicates a search loop

## Error Prevention
- Before using an appliance, verify you're holding the right object
- After navigation, always check the observation to confirm arrival
- Object names include instance numbers — use exact names from observations`,

  workflow: `## Phase 1: Search
1. Identify target from task description
2. Navigate to high-probability locations
3. Open containers and check contents
4. Report: found object + exact location

## Phase 2: Acquisition
1. Navigate to object location (if not already there)
2. Pick up the target object
3. Verify holding the object via observation
4. Report: holding object, ready for next phase

## Phase 3: Processing (if required)
1. Identify required processing (heat/cool/clean)
2. Navigate to the appropriate appliance
3. Apply processing action
4. Verify result in observation

## Phase 4: Delivery
1. Navigate to destination
2. Place object at destination
3. Verify placement in observation

## Error Handling
- Phase fails → retry once, then report failure with details
- Object not found → expand search to all rooms before failing
- Appliance error → verify correct object and appliance combination`,
}
