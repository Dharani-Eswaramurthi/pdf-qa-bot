export interface Citation {
  chunk_id: string
  section_title?: string
  page_start: number
  page_end: number
  score: number
  text: string
}

export interface QueryResponse {
  answer: string
  citations: Citation[]
  used_llm: boolean
  debug?: Record<string, unknown>
}

export async function ingest() {
  const res = await fetch('/api/ingest', { method: 'POST' })
  if (!res.ok) throw new Error('Failed to ingest')
  return res.json()
}

export type ChatMsg = { role: 'user' | 'assistant'; content: string }

export async function query(question: string, top_k = 5, use_llm?: boolean, history?: ChatMsg[]): Promise<QueryResponse> {
  const res = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k, use_llm, history }),
  })
  if (!res.ok) throw new Error('Query failed')
  return res.json()
}

export async function stats() {
  const res = await fetch('/api/stats')
  if (!res.ok) throw new Error('Stats failed')
  return res.json()
}

// Indexing controls
export type IndexStatus = { status: 'idle' | 'indexing' | 'ready' | 'error'; progress: number; message: string }

export async function indexStart() {
  const res = await fetch('/api/index/start', { method: 'POST' })
  if (!res.ok) throw new Error('Failed to start indexing')
  return res.json()
}

export async function indexStatus(): Promise<IndexStatus> {
  const res = await fetch('/api/index/status')
  if (!res.ok) throw new Error('Failed to get indexing status')
  return res.json()
}
