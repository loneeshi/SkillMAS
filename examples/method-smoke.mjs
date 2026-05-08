import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"

import { AgentStore } from "../system/dist/spec/store.js"
import { SkillManager } from "../system/dist/skill/manager.js"
import { AgentQTable } from "../system/dist/skill/agent-q-table.js"
import { SkillQTable } from "../system/dist/skill/q-table.js"
import { ValidationPool } from "../system/dist/round/validation-pool.js"
import { MASRestructurer } from "../system/dist/round/restructuring.js"
import { SkillMASRoundController } from "../system/dist/round/round-controller.js"

const workspace = await mkdtemp(path.join(tmpdir(), "skillmas-method-smoke-"))

try {
  const skillsDir = path.join(workspace, "skills")
  const agentsDir = path.join(workspace, "agents")
  const skillManager = new SkillManager(skillsDir)
  const agentStore = new AgentStore(agentsDir)
  const agentQTable = new AgentQTable()
  const skillQTable = new SkillQTable()

  await seedMethodState(skillManager, agentStore)

  const validationPool = new ValidationPool(skillManager, {
    minUses: 5,
    minSuccessRate: 0.4,
  })
  const restructurer = new MASRestructurer(agentStore, agentQTable)
  const controller = new SkillMASRoundController({
    skillManager,
    skillQTable,
    agentQTable,
    validationPool,
    restructurer,
    executeBatch: async () => ({ traces: methodOnlyTraces() }),
  })

  const result = await controller.runRound(1)
  const promoted = result.validationDecisions.filter((item) => item.action === "promote")
  const generalist = await agentStore.get("generalist")
  const shadowAfter = await skillManager.get("shadow/verify-before-submit")

  console.log(JSON.stringify({
    round: result.round,
    traces: result.traces.length,
    retainedEvidence: result.retainedEvidence.map((item) => ({
      taskType: item.trace.taskType,
      success: item.trace.success,
      reasons: item.reasons,
    })),
    promotedSkills: promoted.map((item) => item.skillId),
    restructuring: {
      action: result.restructuring?.decision.action,
      changed: result.restructuring?.changed,
      patchedAgentIds: result.restructuring?.patchedAgentIds,
      createdAgentIds: result.restructuring?.createdAgentIds,
      removedAgentIds: result.restructuring?.removedAgentIds,
    },
    finalSkillStatus: shadowAfter?.spec.status,
    generalistSkillPool: generalist?.spec.skills,
  }, null, 2))
} finally {
  await rm(workspace, { recursive: true, force: true })
}

async function seedMethodState(skillManager, agentStore) {
  await skillManager.create({
    id: "seed/task-checklist",
    description: "Baseline checklist for decomposing and checking generic tasks",
    type: "workflow",
    whenToUse: "Use before acting on a multi-step task that requires an explicit answer.",
    tags: ["method-smoke", "seed"],
    generatedBy: "manual",
    origin: "manual",
    status: "active",
  }, [
    "## Workflow",
    "EvidenceAlignmentCheckpoint keeps the example content non-trivial for the core validator.",
    "1. Restate the task goal and identify the expected final artifact.",
    "2. List the smallest concrete checks needed before answering.",
    "3. Use `tool.inspect` only when evidence is missing from the current trace.",
    "4. Return a concise answer after checking the required evidence.",
    "",
    "## Recovery",
    "- Avoid skipping the verification step when the final answer depends on an external observation.",
    "- If a tool result contradicts the plan, revise the plan before submitting.",
  ].join("\n"))

  await skillManager.create({
    id: "shadow/verify-before-submit",
    description: "Shadow skill that verifies the final answer against observed evidence before submit",
    type: "task",
    whenToUse: "Use when a task previously failed because the executor answered before checking the evidence.",
    tags: ["method-smoke", "shadow"],
    generatedBy: "manual",
    origin: "failure-pattern",
    status: "shadow",
  }, [
    "## Procedure",
    "VerificationBeforeSubmissionCheckpoint keeps the example content non-trivial for the core validator.",
    "1. Compare the planned final answer with the observations already collected in the trace.",
    "2. If the trace lacks decisive evidence, call `tool.inspect` to retrieve the missing observation.",
    "3. Check that every final claim is supported by a specific observation, not just by the initial plan.",
    "4. Submit the answer only after the evidence and final response agree.",
    "",
    "## Common Mistakes",
    "- Do not treat a selected skill as used unless a trace step actually follows it.",
    "- Avoid final answers that ignore a failed or missing inspection call.",
  ].join("\n"))

  await agentStore.create({
    id: "generalist",
    name: "Generalist",
    role: "worker",
    mode: "all",
    description: "General worker for method smoke tasks",
    role_description: "Handle generic tasks and use skills only when their applicability conditions match.",
    tools: { allow: ["tool.inspect", "tool.submit"], deny: [] },
    memory: { mode: "light", store: "md", capacity: 32 },
    skills: ["seed/task-checklist"],
    skill_pool: ["seed/task-checklist"],
    metadata: { example: "method-smoke" },
  }, "You solve generic tasks. Verify evidence before final submission.")
}

function methodOnlyTraces() {
  const base = {
    taskType: "evidence-check",
    task: "Answer a generic task only after checking the provided evidence.",
    executors: ["generalist"],
    primaryExecutor: "generalist",
    selectedSkills: {
      generalist: ["seed/task-checklist", "shadow/verify-before-submit"],
    },
    usedSkills: {
      generalist: ["seed/task-checklist", "shadow/verify-before-submit"],
    },
    maxSteps: 6,
    toolCalls: [{ tool: "tool.inspect", args: { target: "evidence" } }],
    stepAttribution: [{
      agentId: "generalist",
      skillIds: ["shadow/verify-before-submit"],
      tool: "tool.inspect",
      stepOutcome: "success",
    }],
  }

  return [
    {
      ...base,
      episodeId: "fail-1",
      success: false,
      score: 0.25,
      steps: 6,
      errorMessage: "submitted before evidence was verified",
      summary: "The executor selected the verification skill but still submitted too early.",
      retrievalExecutionMismatch: true,
    },
    {
      ...base,
      episodeId: "fail-2",
      success: false,
      score: 0.25,
      steps: 6,
      errorMessage: "submitted before evidence was verified",
      summary: "Repeated premature submission creates retained failure evidence.",
      retrievalExecutionMismatch: true,
    },
    {
      ...base,
      episodeId: "fail-3",
      success: false,
      score: 0.25,
      steps: 6,
      errorMessage: "submitted before evidence was verified",
      summary: "The same failure recurs under the same executor and task type.",
      retrievalExecutionMismatch: true,
    },
    {
      ...base,
      episodeId: "success-1",
      success: true,
      score: 1,
      steps: 3,
      summary: "The executor verified evidence before the final answer.",
      retrievalExecutionMismatch: false,
    },
    {
      ...base,
      episodeId: "success-2",
      success: true,
      score: 1,
      steps: 3,
      summary: "The verification skill produced a reusable success trace.",
      retrievalExecutionMismatch: false,
    },
  ]
}
