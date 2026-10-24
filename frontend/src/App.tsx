import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getDescription, getScan, postThumbnail, scanStream, shape, type Description, type Method, type OffsetRange, type Payload, type ScanResult } from './api'
import { Controls } from './components/Controls'
import { Gallery } from './components/Gallery'
import { InspectorNotes } from './components/InspectorNotes'
import { StageStrip, type Stage } from './components/StageStrip'
import { Viewer } from './components/Viewer'
import type { ViewMode } from './components/VoxelModel'
import { benchPayload, cogPayload } from './placeholder'

type Status = 'idle' | 'scanning' | 'ready' | 'error'
const route = () => (location.hash.startsWith('#/gallery') ? 'gallery' : 'scan')

export default function App() {
  const [page, setPage] = useState(route())
  const [status, setStatus] = useState<Status>('idle')
  const [stages, setStages] = useState<Partial<Record<Stage['key'], Stage>>>({})
  const [result, setResult] = useState<ScanResult | null>(null)
  const [payload, setPayload] = useState<Payload | null>(null)
  const [method, setMethod] = useState<Method>('inflate')
  const [thickness, setThickness] = useState(1)
  const [planeOffset, setPlaneOffset] = useState(0)
  const [offsetRange, setOffsetRange] = useState<OffsetRange | null>(null)
  const [showGuide, setShowGuide] = useState(true)
  const [mode, setMode] = useState<ViewMode>('photo')
  const [animKey, setAnimKey] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [label, setLabel] = useState<string | null>(null)
  const [knownDim, setKnownDim] = useState<Record<string, number> | null>(null)
  const [snapshotKey, setSnapshotKey] = useState(0)
  const [desc, setDesc] = useState<Description | null>(null)
  const [fps, setFps] = useState(0)
  const [drag, setDrag] = useState(false)
  // the Inspector's hint may switch a borderline auto pick, but only until the user picks a method
  const touched = useRef(false)
  const hintApplied = useRef(false)
  const reshapeRef = useRef<((m: Method) => void) | null>(null)
  const methodRef = useRef<Method>('inflate')
  const [notice, setNotice] = useState<{ from: Method; to: Method; text: string } | null>(null)
  const needsThumb = useRef<string | null>(null)        // scan id whose canvas thumbnail is still to be uploaded
  const fileRef = useRef<HTMLInputElement>(null)
  const placeholder = useMemo(() => {
    const b = new URLSearchParams(location.search).get('bench')
    return b ? benchPayload(parseInt(b, 10)) : cogPayload()
  }, [])

  useEffect(() => {
    const f = () => setPage(route())
    window.addEventListener('hashchange', f)
    return () => window.removeEventListener('hashchange', f)
  }, [])

  // poll the object description for the current scan until it is no longer pending
  useEffect(() => {
    if (!result) return
    let stop = false
    let timer: number | undefined
    const tick = async (n: number) => {
      try {
        const d = await getDescription(result.scan_id)
        if (stop) return
        setDesc(d)
        const h = d.selection_hint
        if (d.status === 'ready' && h?.switch && !touched.current && !hintApplied.current) {
          hintApplied.current = true
          const from = result.recommended
          setMethod(h.recommended)
          reshapeRef.current?.(h.recommended)
          setNotice({ from, to: h.recommended, text: h.hint_effect })
        }
        if (d.status === 'ready' && d.result) setLabel(d.result.name)
        // the category changed the mirror plane prior: the mirror on screen is stale (the server pre-built the new one)
        if (d.status === 'ready' && (d.prior_scale ?? 1) !== 1 && methodRef.current === 'mirror' && !hintApplied.current)
          reshapeRef.current?.('mirror')
        if (d.status === 'pending' && n < 90) timer = window.setTimeout(() => tick(n + 1), 1000)
      } catch {
        if (!stop && n < 90) timer = window.setTimeout(() => tick(n + 1), 2000)
      }
    }
    tick(0)
    return () => { stop = true; window.clearTimeout(timer) }
  }, [result])

  const loadResult = (r: ScanResult, fresh: boolean) => {
    // a reopened scan already has a chosen method: never auto-switch it
    touched.current = !fresh || r.method !== r.recommended
    hintApplied.current = false
    needsThumb.current = fresh ? r.scan_id : null
    setNotice(null)
    setResult(r)
    setPayload(r.payload)
    setMethod(r.method)
    setThickness(r.thickness)
    setPlaneOffset(r.plane_offset_cm ?? 0)
    setOffsetRange(r.plane_offset_range)
    setLabel(r.label)
    setKnownDim(r.known_dim)
    setStages({
      photo: { key: 'photo', title: 'Photo', url: r.urls.photo, ms: r.timings.load_ms ?? 0 },
      depth: { key: 'depth', title: 'Depth', url: r.urls.depth, ms: r.timings.depth_ms ?? 0 },
      mask: { key: 'mask', title: 'Mask', url: r.urls.mask, ms: r.timings.segment_ms ?? 0 },
      model: { key: 'model', title: 'Model', url: `${r.urls.thumbnail}?v=${r.scan_id}`, ms: Object.entries(r.timings).filter(([k]) => k.startsWith('shape_') || k === 'voxel_cleanup_ms').reduce((a, [, v]) => a + v, 0) },
    })
    setStatus('ready')
    setAnimKey((k) => k + 1)
  }

  const onFile = async (file: File) => {
    if (!file.type.startsWith('image/')) return setError('That is not an image.')
    location.hash = '#/'
    setError(null)
    setStatus('scanning')
    setStages({})
    setLabel(null)
    setResult(null)
    setPayload(null)
    setDesc(null)
    try {
      await scanStream(file, null, (e) => {
        if (e.stage === 'photo' || e.stage === 'depth' || e.stage === 'mask')
          setStages((s) => ({ ...s, [e.stage]: { key: e.stage, title: e.stage, url: e.url, ms: e.ms } }))
        else if (e.stage === 'label') setLabel(e.label)
        else if (e.stage === 'done') loadResult(e.result, true)
        else if (e.stage === 'error') throw new Error(e.error)
      })
    } catch (err) {
      setStatus('error')
      setError(String(err))
    }
  }

  const reshape = async (m: Method, t: number, kd?: string | null, off = planeOffset) => {
    if (!result) return
    setBusy(true)
    try {
      const r = await shape(result.scan_id, m, t, kd, m === 'mirror' ? off : 0)
      setPayload(r.payload)
      setKnownDim(r.known_dim)
      if (r.plane_offset_range) {
        setOffsetRange(r.plane_offset_range)
        setPlaneOffset(r.plane_offset_cm)            // the server clamps to the range
      }
      setStages((s) => ({ ...s, model: { ...s.model, key: 'model', title: 'Model', ms: r.timings.total_ms } }))
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(false)
    }
  }

  reshapeRef.current = (m: Method) => { void reshape(m, thickness) }
  methodRef.current = method
  const pickMethod = (m: Method) => {
    touched.current = true
    setNotice(null)
    setMethod(m)
    reshape(m, thickness)
  }

  // after a fresh scan has assembled, its canvas snapshot becomes the archive thumbnail
  const onAssembled = useCallback(() => { if (needsThumb.current) setSnapshotKey((k) => k + 1) }, [])
  const onSnapshot = useCallback((b: Blob) => {
    const id = needsThumb.current
    needsThumb.current = null
    if (id) postThumbnail(id, b).catch(() => {})
  }, [])

  const openScan = async (id: string) => {
    location.hash = '#/'
    setStatus('scanning')
    setDesc(null)
    try {
      loadResult(await getScan(id), false)
    } catch (err) {
      setStatus('error')
      setError(String(err))
    }
  }

  const shown = payload ?? placeholder
  return (
    <div className={`app ${drag ? 'dragging' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) onFile(f) }}>
      <header className="topbar panel">
        <h1>⚙ Steel &amp; Steam <span>one-photo voxel foundry</span></h1>
        <nav>
          <a href="#/" className={page === 'scan' ? 'active' : ''}>Workshop</a>
          <a href="#/gallery" className={page === 'gallery' ? 'active' : ''}>Archive</a>
        </nav>
      </header>

      {page === 'gallery' ? (
        <Gallery onOpen={openScan} />
      ) : (
        <main className="workshop">
          <section className="viewer-wrap panel">
            <Viewer payload={shown} mode={payload ? mode : 'observed'} animKey={animKey} loop={!payload}
              fitKey={result ? `${result.scan_id}|${payload?.scale_source}|${JSON.stringify(knownDim)}` : 'placeholder'} showDims={!!payload}
              onAssembled={onAssembled} snapshotKey={snapshotKey} onSnapshot={onSnapshot}
              onFps={import.meta.env.DEV ? setFps : undefined} showGuide={showGuide && !!payload} />
            <div className="viewer-hud">
              <div className="hud-title">{label ?? (status === 'scanning' ? 'Stoking the boiler…' : 'Awaiting a photograph')}</div>
              {result && <div className="hud-sub">{result.reason}</div>}
              {import.meta.env.DEV && fps > 0 && <div className="hud-fps">{fps.toFixed(0)} fps · {shown.voxels.length / 3} voxels</div>}
            </div>
          </section>

          <aside className="side">
            <div className="capture panel">
              <input ref={fileRef} type="file" accept="image/*" capture="environment" hidden
                onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = '' }} />
              <button className="brass-btn big wide" onClick={() => fileRef.current?.click()} disabled={status === 'scanning'}>
                {status === 'scanning' ? 'Working…' : '📷 Take / choose photo'}
              </button>
              <div className="hint">or drop an image anywhere · side-on photo, object on a table, calibration plate in view for real measurements</div>
              {error && <div className="error">{error}</div>}
            </div>
            <StageStrip stages={stages} active={status === 'scanning'} />
            {notice && (
              <div className="inspector-notice panel" role="status">
                <span>Inspector recommends <b>{notice.to[0].toUpperCase() + notice.to.slice(1)}</b>
                  <small title={notice.text}> (borderline call)</small></span>
                <button className="steel-btn" onClick={() => pickMethod(notice.from)}>Undo</button>
              </div>
            )}
            {result && payload && (
              <Controls scanId={result.scan_id} payload={payload} recommended={notice ? notice.to : result.recommended}
                selection={result.selection} method={method}
                thickness={thickness} planeOffset={planeOffset} offsetRange={offsetRange}
                showGuide={showGuide} mode={mode} busy={busy} knownDim={knownDim}
                onMethod={pickMethod}
                onThickness={(t) => { setThickness(t); reshape(method, t) }}
                onPlaneOffset={(o) => { setPlaneOffset(o); reshape(method, thickness, undefined, o) }}
                onShowGuide={setShowGuide}
                onMode={setMode}
                onKnownDim={(s) => reshape(method, thickness, s, 0)}
                onReplay={() => setAnimKey((k) => k + 1)} />
            )}
            {result && <InspectorNotes d={desc} method={method} />}
          </aside>
        </main>
      )}
      {drag && <div className="drop-overlay">Drop the photograph into the furnace</div>}
    </div>
  )
}
