import type { Description } from '../api'

const fmt = (v: number) => (Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(1) : v.toFixed(3))

/** "Inspector's notes": the VLM's description + explanation next to the selection trace (each check, value, threshold). */
export function InspectorNotes({ d, method }: { d: Description | null; method: string }) {
  if (!d) return null
  const checks = d.selection?.selection_trace?.checks ?? []
  const r = d.result
  const pending = d.status === 'pending'
  return (
    <div className="inspector panel" aria-live="polite">
      <div className="inspector-head">
        <h2>Inspector&rsquo;s notes</h2>
        {pending && <span className="inspector-status">examining<span className="dots" /></span>}
        {d.status === 'fallback' && <span className="inspector-status">inspector unavailable</span>}
        {d.status === 'ready' && d.meta?.ms !== undefined && <span className="inspector-status">{(d.meta.ms / 1000).toFixed(1)} s</span>}
      </div>
      <div className="inspector-body">
        <div className="inspector-text">
          {pending && <p className="muted">Holding the specimen up to the lamp…</p>}
          {r && d.status === 'ready' && (
            <>
              <div className="inspector-name">{r.name}</div>
              <p>{r.description}</p>
              <p><b>Why {d.auto.recommended}:</b> {r.method_explanation}</p>
              <p className="inspector-tags">
                <span className="tag">{r.shape_category}</span>
                <span className={`tag ${r.suggested_method === d.auto.recommended ? 'agree' : 'disagree'}`}>
                  suggests {r.suggested_method} {r.suggested_method === d.auto.recommended ? '✓ agrees' : '≠ auto'}
                </span>
                {method !== d.auto.recommended && <span className="tag">showing {method}</span>}
                {d.prior_scale !== undefined && d.prior_scale !== 1 && (
                  <span className="tag" title="mirror symmetry-plane prior adjusted from the shape category">
                    mirror prior ×{d.prior_scale}</span>
                )}
              </p>
            </>
          )}
          {d.status === 'fallback' && <p className="muted">No description (the VLM is not running). Raw scores still apply.</p>}
        </div>
        <table className="scores">
          <caption>auto_select → <b>{d.auto.recommended}</b></caption>
          <tbody>
            {checks.map((c) => (
              <tr key={c.check} className={c.pass ? 'pass' : 'fail'}>
                <th>{c.check}</th>
                <td>{fmt(c.value)} <small>needs {c.op === '>=' ? '≥' : '≤'} {c.threshold}</small> {c.pass ? '✓' : '✗'}{c.borderline ? ' ~' : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="inspector-reason">{d.auto.reason}
        {d.selection_hint && <> · Inspector hint: {d.selection_hint.hint ?? 'none'} → {d.selection_hint.hint_effect}</>}</div>
    </div>
  )
}
