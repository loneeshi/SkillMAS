import { z } from "zod"

export const MemoryEntrySchema = z.object({
  timestamp: z.string(),
  agent: z.string(),
  type: z.enum(["lesson", "error", "action"]),
  content: z.string(),
  tags: z.array(z.string()).default([]),
  ttlHours: z.number().positive().optional(),
})

export type MemoryEntry = z.infer<typeof MemoryEntrySchema>
