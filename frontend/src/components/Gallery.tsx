import { useEffect, useState } from 'react'
import { deleteProject, listProjects, renameProject, type Project } from '../api'
import { signInWithGoogle, supabase, type User } from '../supabase'

const when = (iso: string) => new Date(iso).toLocaleString()

/** The archive: your saved projects (stored in your account, so they follow you to any machine). */
export function Gallery({ user, onOpen }: { user: User | null; onOpen: (scanId: string) => void }) {
  const [items, setItems] = useState<Project[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const load = () => listProjects().then(setItems).catch((e) => setErr(String(e)))
  useEffect(() => { if (user) load() }, [user?.id])
  if (!supabase) return <div className="panel gallery-empty">Saving is not configured on this build.</div>
  if (!user) {
    return (
      <div className="panel gallery-empty">
        <p>The archive keeps <b>your saved projects</b>. Scanning works without an account; saving needs one.</p>
        <button className="brass-btn" onClick={() => signInWithGoogle()}>Sign in with Google</button>
      </div>
    )
  }
  if (err) return <div className="panel error">Could not load your projects: {err}</div>
  if (!items) return <div className="panel">Loading the archive…</div>
  if (!items.length) return <div className="panel gallery-empty">No saved projects yet. Scan something and press <b>Save</b>.</div>
  return (
    <div className="gallery">
      {items.map((p) => (
        <div key={p.scan_id} className="card panel">
          <button className="card-open" onClick={() => onOpen(p.scan_id)} aria-label={`open ${p.name}`}>
            {p.thumbnail ? <img src={p.thumbnail} alt="" loading="lazy" /> : <div className="stage-empty" />}
          </button>
          <div className="card-title">{p.name}</div>
          <div className="card-meta">{p.method} · {p.dims_cm ? `${p.dims_cm.w} × ${p.dims_cm.h} × ${p.dims_cm.d} cm` : '–'}</div>
          <div className="card-meta">updated {when(p.updated_at)} · <span className={`scale-src src-${p.scale_source}`}>{p.scale_source}</span></div>
          <div className="card-actions">
            <button className="brass-btn" onClick={() => onOpen(p.scan_id)}>Open</button>
            <button className="steel-btn" onClick={async () => {
              const n = prompt('Rename project', p.name)
              if (n && n.trim()) { await renameProject(p.scan_id, n.trim()); load() }
            }}>Rename</button>
            <button className="steel-btn danger" onClick={async () => {
              if (confirm(`Delete "${p.name}"? This removes it from your account and cannot be undone.`)) {
                await deleteProject(p.scan_id); load()
              }
            }}>Delete</button>
          </div>
        </div>
      ))}
    </div>
  )
}
