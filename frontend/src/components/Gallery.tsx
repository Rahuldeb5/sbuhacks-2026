import { useEffect, useState } from 'react'
import { listScans, type ScanSummary } from '../api'

/** The archive: recent scans, newest first (no accounts yet, so every scan on this server). */
export function Gallery({ onOpen }: { onOpen: (scanId: string) => void }) {
  const [items, setItems] = useState<ScanSummary[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { listScans().then(setItems).catch((e) => setErr(String(e))) }, [])
  if (err) return <div className="panel error">Could not load the archive: {err}</div>
  if (!items) return <div className="panel">Loading the archive…</div>
  if (!items.length) return <div className="panel gallery-empty">Nothing in the archive yet. Scan something in the <a href="#/">Workshop</a>.</div>
  return (
    <div className="gallery">
      {items.map((s) => (
        <div key={s.scan_id} className="card panel">
          <button className="card-open" onClick={() => onOpen(s.scan_id)} aria-label={`open ${s.label}`}>
            <img src={s.thumbnail} alt="" loading="lazy" />
          </button>
          <div className="card-title">{s.label}</div>
          <div className="card-meta">{s.method} · {s.dims_cm ? `${s.dims_cm.w} × ${s.dims_cm.h} × ${s.dims_cm.d} cm` : '–'}</div>
          <div className="card-meta">{s.created} · <span className={`scale-src src-${s.scale_source}`}>{s.scale_source}</span></div>
          <div className="card-actions">
            <button className="brass-btn" onClick={() => onOpen(s.scan_id)}>Open</button>
          </div>
        </div>
      ))}
    </div>
  )
}
