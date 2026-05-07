/**
 * MessageBus — central pub/sub hub for inter-agent communication.
 *
 * Every agent subscribes to its own ID + the "*" (broadcast) channel.
 * Messages are delivered asynchronously; the bus is purely in-process.
 */

import type { AgentMessage, MessageHandler, MessageType } from "./types"

let _nextId = 0
function nextMessageId(): string {
  return `msg_${Date.now()}_${++_nextId}`
}

export class MessageBus {
  private subs = new Map<string, MessageHandler[]>()
  private history: AgentMessage[] = []
  private maxHistory = 500

  subscribe(channel: string, handler: MessageHandler): () => void {
    const list = this.subs.get(channel) ?? []
    list.push(handler)
    this.subs.set(channel, list)

    return () => {
      const idx = list.indexOf(handler)
      if (idx >= 0) list.splice(idx, 1)
    }
  }

  async send(params: {
    from: string
    to: string
    type: MessageType
    content: string
    metadata?: Record<string, unknown>
    replyTo?: string
  }): Promise<AgentMessage> {
    const msg: AgentMessage = {
      id: nextMessageId(),
      from: params.from,
      to: params.to,
      type: params.type,
      content: params.content,
      metadata: params.metadata,
      timestamp: new Date().toISOString(),
      replyTo: params.replyTo,
    }

    this.record(msg)

    const handlers = [
      ...(this.subs.get(msg.to) ?? []),
      ...(msg.to !== "*" ? (this.subs.get("*") ?? []) : []),
    ]

    await Promise.allSettled(handlers.map((h) => h(msg)))
    return msg
  }

  async broadcast(params: {
    from: string
    type: MessageType
    content: string
    metadata?: Record<string, unknown>
  }): Promise<AgentMessage> {
    return this.send({ ...params, to: "*" })
  }

  getHistory(filter?: {
    from?: string
    to?: string
    type?: MessageType
    limit?: number
  }): AgentMessage[] {
    let items = this.history

    if (filter?.from) items = items.filter((m) => m.from === filter.from)
    if (filter?.to) items = items.filter((m) => m.to === filter.to)
    if (filter?.type) items = items.filter((m) => m.type === filter.type)

    if (filter?.limit) items = items.slice(-filter.limit)
    return items
  }

  clear(): void {
    this.history = []
  }

  private record(msg: AgentMessage): void {
    this.history.push(msg)
    if (this.history.length > this.maxHistory) {
      this.history = this.history.slice(-this.maxHistory)
    }
  }
}
