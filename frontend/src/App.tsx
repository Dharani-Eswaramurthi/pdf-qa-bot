import { useEffect, useMemo, useRef, useState } from 'react'
import { query as queryApi, ingest as ingestApi, stats as statsApi, Citation, ChatMsg, indexStart, indexStatus } from './api'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { motion, AnimatePresence } from 'framer-motion'
import { FiSend, FiCpu } from 'react-icons/fi'

type Msg = { id: string; role: 'user' | 'assistant'; content: string; citations?: Citation[] }

function Avatar({ role }: { role: 'user' | 'assistant' }) {
  return (
    <div className="avatar" title={role === 'assistant' ? 'Assistant' : 'You'}>
      {role === 'assistant' ? 'P' : 'U'}
    </div>
  )
}

function Sources({ items }: { items: Citation[] }) {
  return (
    <div className="sources">
      <div className="muted" style={{ marginBottom: 6 }}>Sources</div>
      <div style={{ display: 'grid', gap: 8 }}>
        {items.map((c) => (
          <details key={c.chunk_id} className="sourceCard">
            <summary>
              p.{c.page_start}-{c.page_end} • {c.section_title || 'Untitled'} • score {c.score.toFixed(3)}
            </summary>
            <div style={{ marginTop: 6, whiteSpace: 'pre-wrap' }}>{c.text}</div>
          </details>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([
    {
      id: typeof crypto !== 'undefined' && 'randomUUID' in crypto ? crypto.randomUUID() : Math.random().toString(36).slice(2),
      role: 'assistant',
      content:
        "Hi, I’m Patrick AI — your PDF manual assistant. Ask me anything about your manual and I’ll answer with clear citations.",
    },
  ])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [info, setInfo] = useState<any>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const [indexing, setIndexing] = useState(false)
  const [indexProgress, setIndexProgress] = useState(0)
  const [indexMsg, setIndexMsg] = useState('')

  useEffect(() => {
    statsApi().then(async (s) => {
      setInfo(s)
      if (!s.has_index) {
        setIndexing(true)
        try { await indexStart() } catch {}
        const poll = async () => {
          try {
            const st = await indexStatus()
            setIndexProgress(st.progress || 0)
            setIndexMsg(st.message || '')
            setIndexing(st.status === 'indexing')
            if (st.status !== 'indexing') {
              setInfo(await statsApi())
              return
            }
          } catch {}
          setTimeout(poll, 1000)
        }
        poll()
      }
    }).catch(() => setInfo(null))
  }, [])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])

  async function handleIngest() {
    setLoading(true)
    try {
      const res = await ingestApi()
      setInfo(await statsApi())
      setMessages((m) => m.concat({ id: crypto.randomUUID(), role: 'assistant', content: `Indexed ${res.chunks_indexed} chunks across ${res.pages_processed} pages. Ready to answer questions.` }))
    } catch (e: any) {
      setMessages((m) => m.concat({ id: crypto.randomUUID(), role: 'assistant', content: `Ingest failed: ${e.message || 'Unknown error'}` }))
    } finally { setLoading(false) }
  }

  async function handleAsk() {
    if (indexing) return
    const q = question.trim()
    if (!q) return
    setQuestion('')
    const userMsg: Msg = { id: crypto.randomUUID(), role: 'user', content: q }
    setMessages((m) => m.concat(userMsg))
    setLoading(true)
    try {
      const history: ChatMsg[] = messages.map(m => ({ role: m.role, content: m.content }))
      const res: any = await queryApi(q, 5, undefined, history)
      if (res?.status?.status === 'indexing') {
        setIndexing(true)
        setIndexProgress(res.status.progress || 0)
        setIndexMsg(res.status.message || 'Indexing…')
        const poll = async () => {
          try {
            const st = await indexStatus()
            setIndexProgress(st.progress || 0)
            setIndexMsg(st.message || '')
            setIndexing(st.status === 'indexing')
            if (st.status !== 'indexing') {
              setInfo(await statsApi())
              return
            }
          } catch {}
          setTimeout(poll, 1000)
        }
        poll()
      } else {
        const ans: Msg = { id: crypto.randomUUID(), role: 'assistant', content: res.answer, citations: res.citations }
        setMessages((m) => m.concat(ans))
      }
    } catch (e: any) {
      setMessages((m) => m.concat({ id: crypto.randomUUID(), role: 'assistant', content: `Error: ${e.message || 'Failed to query'}` }))
    } finally { setLoading(false) }
  }

  return (
    <div className="app">
      <header className="header">
        {/* <div className="logo" /> */}
        <div>
          <div className="title">Manual Q&A Bot</div>
          <div className="subtitle">Chat over the PDF manual</div>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          {info ? (
            <span className="pill">{info.has_index ? 'Indexed' : 'Not indexed'}</span>
          ) : (
            <span className="pill">Waiting for backend…</span>
          )}
          {info && <span className="pill">{info.chunks} chunks • {info.pages} pages</span>}
          {info && <span className="pill"><FiCpu style={{ marginRight: 6 }} /> {info.embedding_model}</span>}
        </div>
      </header>

      <main className="content" aria-busy={indexing || undefined}>
        <section className="panel chat">
          <div className="messages">
            <AnimatePresence initial={false}>
              {messages.map((m) => (
                <motion.div
                  key={m.id}
                  className={`msg ${m.role}`}
                  initial={{ opacity: 0, y: 10, filter: 'blur(4px)' }}
                  animate={{ opacity: 1, y: 0, filter: 'blur(0)' }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ type: 'spring', stiffness: 280, damping: 24 }}
                >
                  {m.role === 'assistant' ? (
                    <>
                      <Avatar role={m.role} />
                      <div className={`bubble ${m.role}`}>
                        <div className="md">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                        </div>
                        {m.citations && m.citations.length > 0 && <Sources items={m.citations} />}
                      </div>
                    </>
                  ) : (
                    <>
                      <div className={`bubble ${m.role}`}>
                        <div className="md">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                        </div>
                      </div>
                      <Avatar role={m.role} />
                    </>
                  )}
                </motion.div>
              ))}
              {loading && (
                <motion.div key="typing" className="msg" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <Avatar role="assistant" />
                  <div className="bubble assistant"><span className="typing"><span className="dot" /><span className="dot" /><span className="dot" /></span></div>
                </motion.div>
              )}
            </AnimatePresence>
            <div ref={endRef} />
          </div>
        </section>

        <footer className="panel footer">
          <div className="footer-inner">
            <div className="row" style={{ gap: 8 }}>
              <div className="inputWrap" style={{ width: '100%' }}>
                <input
                  className="input"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a question about your manual…"
                  onKeyDown={(e) => { if (!indexing && e.key === 'Enter' && !e.shiftKey) handleAsk() }}
                  disabled={indexing}
                />
                <button className="sendBtn" onClick={handleAsk} disabled={loading || !question.trim()} title="Send">
                  <FiSend />
                </button>
              </div>
            </div>
          </div>
        </footer>
      </main>
      {indexing && (
        <div className="overlay">
          <div className="overlayCard">
            <div style={{ marginBottom: 10, fontWeight: 600 }}>Building the index…</div>
            <div className="progressBar"><div style={{ width: `${Math.min(100, Math.max(0, indexProgress))}%` }} /></div>
            <div className="muted" style={{ marginTop: 8 }}>{indexMsg || 'Preparing vectors and sections…'}</div>
          </div>
        </div>
      )}
    </div>
  )
}
