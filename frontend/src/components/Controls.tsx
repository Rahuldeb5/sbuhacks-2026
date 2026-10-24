import { useEffect, useRef, useState } from 'react'
import { exportUrl, type Method, type OffsetRange, type Payload, type Selection } from '../api'
import type { ViewMode } from './VoxelModel'

const NAME: Record<Method, string> = { mirror: 'Mirror', revolve: 'Revolve', inflate: 'Inflate' }
const fmt = (v: number) => (Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(1) : v.toFixed(2))

/** One-paragraph summary of the selection trace for the "auto" tooltip. */
export function traceSummary(sel?: Selection): string {
  const tr = sel?.selection_trace
  if (!tr) return sel?.reason ?? ''
  const line = (g: string) => tr.checks.filter((c) => c.group.startsWith(g)).map((c) =>
    `${c.check.split('.')[1]} ${fmt(c.value)} (needs ${c.op === '>=' ? '≥' : '≤'} ${c.threshold}) ${c.pass ? '✓' : '✗'}${c.borderline ? ' borderline' : ''}`).join(', ')
  const parts = [`Auto picked ${NAME[tr.final]}.`, `Revolve: ${line('revolve')}.`, `Mirror: ${line('mirror')}.`]
  if (tr.hint) parts.push(`Inspector: ${tr.hint} → ${tr.hint_effect}.`)
  return parts.join('\n')
}

/** A slider step that gives ~80 positions over the range, rounded to 0.05 / 0.1 / 0.25 / 0.5 / 1 cm. */
function niceStep(range: OffsetRange): number {
  const raw = (range[1] - range[0]) / 80
  return [0.05, 0.1, 0.25, 0.5, 1].find((s) => s >= raw) ?? 1
}

const SCALE_TEXT: Record<string, string> = {
  marker: 'measured with calibration plate',
  known_dim: 'scaled from your known dimension',
  estimate: 'estimated (no reference) — can be off by 3–10×',
}

interface Props {
  scanId: string
  payload: Payload
  recommended: Method
  selection?: Selection
  method: Method
  thickness: number
  planeOffset: number
  offsetRange: OffsetRange | null
  showGuide: boolean
  mode: ViewMode
  busy: boolean
  knownDim: Record<string, number> | null
  onMethod: (m: Method) => void
  onThickness: (t: number) => void
  onPlaneOffset: (o: number) => void
  onShowGuide: (b: boolean) => void
  onMode: (m: ViewMode) => void
  onKnownDim: (s: string | null) => void
  onReplay: () => void
}

export function Controls(p: Props) {
  const [t, setT] = useState(p.thickness)
  const [off, setOff] = useState(p.planeOffset)
  const timer = useRef<number | undefined>(undefined)
  useEffect(() => setT(p.thickness), [p.thickness, p.scanId])
  useEffect(() => setOff(p.planeOffset), [p.planeOffset, p.scanId])
  const slide = (v: number) => {
    setT(v)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => p.onThickness(v), 250) // debounced: one /api/shape per pause
  }
  const slideOff = (v: number) => {
    setOff(v)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => p.onPlaneOffset(v), 200)
  }
  const [dim, setDim] = useState<'h' | 'w' | 'd'>('h')
  const [val, setVal] = useState('')
  const m = p.payload.measurements_cm
  const src = p.payload.scale_source
  const range: OffsetRange = p.offsetRange ?? [0, 4]
  const step = niceStep(range)
  // a range input only stops at min + k*step: snap the ends to whole steps so 0 (the auto plane) is reachable;
  // the server clamps anything past the real bounds
  const lo = +(Math.floor(range[0] / step) * step).toFixed(4)
  const hi = +(Math.ceil(range[1] / step) * step).toFixed(4)
  return (
    <div className="controls panel">
      <div className="method-row">
        {(['mirror', 'revolve'] as Method[]).map((k) => (
          <button key={k} className={`brass-btn big ${p.method === k ? 'active' : ''}`} disabled={p.busy && p.method !== k}
            onClick={() => p.onMethod(k)} aria-pressed={p.method === k}
            title={p.recommended === k ? traceSummary(p.selection) : undefined}>
            {NAME[k]}
            {p.recommended === k && <span className="auto-tag">auto</span>}
          </button>
        ))}
      </div>
      <details className="advanced" open={p.method === 'inflate' || p.recommended === 'inflate' || undefined}>
        <summary>Advanced</summary>
        <div className="fallback-row">
          <button className={`steel-btn ${p.method === 'inflate' ? 'active' : ''}`} disabled={p.busy && p.method !== 'inflate'}
            onClick={() => p.onMethod('inflate')} aria-pressed={p.method === 'inflate'}
            title={p.recommended === 'inflate' ? traceSummary(p.selection) : undefined}>
            Fallback shape{p.recommended === 'inflate' && <span className="auto-tag small">auto</span>}
          </button>
          <small>puffs up the outline; use when neither assumption fits</small>
        </div>
      </details>

      {p.method === 'mirror' ? (
        <label className="slider-row">
          <span>Symmetry plane depth <b>{off >= 0 ? '+' : ''}{off.toFixed(step < 0.1 ? 2 : 1)} cm</b>
            <small> {Math.abs(off) < 1e-6 ? '(auto estimate)' : off > 0 ? 'deeper → fatter' : 'shallower → thinner'}</small></span>
          <input type="range" min={lo} max={hi} step={step} value={off}
            onChange={(e) => slideOff(parseFloat(e.target.value))} aria-label="symmetry plane depth" />
        </label>
      ) : (
        <label className="slider-row">
          <span>Thickness <b>{t.toFixed(2)}×</b></span>
          <input type="range" min={0.25} max={2.5} step={0.05} value={t} onChange={(e) => slide(parseFloat(e.target.value))}
            aria-label="thickness" />
        </label>
      )}

      <div className="toggle-row" role="group" aria-label="colour mode">
        <button className={`steel-btn ${p.mode === 'photo' ? 'active' : ''}`} onClick={() => p.onMode('photo')}>Photo colours</button>
        <button className={`steel-btn ${p.mode === 'observed' ? 'active' : ''}`} onClick={() => p.onMode('observed')}>
          Observed <i className="sw brass" /> / inferred <i className="sw steel" />
        </button>
        {p.method !== 'inflate' && (
          <label className="guide-toggle">
            <input type="checkbox" checked={p.showGuide} onChange={(e) => p.onShowGuide(e.target.checked)} />
            Show {p.method === 'mirror' ? 'symmetry plane' : 'axis'}
          </label>
        )}
      </div>

      <div className="measure">
        <div className="dims">{m.w.toFixed(1)} × {m.h.toFixed(1)} × {m.d.toFixed(1)} <small>cm (W × H × D)</small></div>
        <div className={`scale-src src-${src}`}>{SCALE_TEXT[src] ?? src}</div>
      </div>

      <form className="known-row" onSubmit={(e) => {
        e.preventDefault()
        const v = parseFloat(val)
        if (v > 0) p.onKnownDim(`${dim}=${v}`)
      }}>
        <span>I know the</span>
        <select value={dim} onChange={(e) => setDim(e.target.value as 'h' | 'w' | 'd')}>
          <option value="h">height</option><option value="w">width</option><option value="d">depth</option>
        </select>
        <input type="number" inputMode="decimal" min="0.1" step="0.1" placeholder="cm" value={val} onChange={(e) => setVal(e.target.value)} />
        <button className="steel-btn" type="submit">Apply</button>
        {p.knownDim && <button className="steel-btn" type="button" onClick={() => { setVal(''); p.onKnownDim(null) }}>Clear</button>}
      </form>

      <div className="export-row">
        <a className="brass-btn" href={exportUrl(p.scanId, 'blocky', p.method, t, off)} download>⬇ Blocky STL</a>
        <a className="brass-btn" href={exportUrl(p.scanId, 'smooth', p.method, t, off)} download>⬇ Smooth STL</a>
        <button className="steel-btn" onClick={p.onReplay}>↻ Replay assembly</button>
      </div>
    </div>
  )
}
