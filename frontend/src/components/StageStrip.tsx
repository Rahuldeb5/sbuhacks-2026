export interface Stage {
  key: 'photo' | 'depth' | 'mask' | 'model'
  title: string
  url?: string
  ms?: number
}

const ORDER: Stage['key'][] = ['photo', 'depth', 'mask', 'model']

/** Pressure-gauge progress: the needle sweeps one notch per finished pipeline stage. */
export function Gauge({ done, active }: { done: number; active: boolean }) {
  const frac = done / ORDER.length
  const angle = -90 + 180 * frac
  const labels = ['PHOTO', 'DEPTH', 'MASK', 'MODEL']
  return (
    <svg className={`gauge ${active ? 'gauge-active' : ''}`} viewBox="0 0 120 72" role="img"
      aria-label={`pipeline progress ${done} of 4`}>
      <defs>
        <radialGradient id="gface" cx="50%" cy="85%" r="80%">
          <stop offset="0" stopColor="#f6ead0" />
          <stop offset="1" stopColor="#cdb98f" />
        </radialGradient>
      </defs>
      <path d="M6 66 A54 54 0 0 1 114 66 Z" fill="url(#gface)" stroke="#7a5a1e" strokeWidth="3" />
      {Array.from({ length: 9 }, (_, i) => {
        const a = Math.PI * (1 - i / 8)
        const r1 = i % 2 ? 44 : 40
        return <line key={i} x1={60 + Math.cos(a) * r1} y1={66 - Math.sin(a) * r1} x2={60 + Math.cos(a) * 49} y2={66 - Math.sin(a) * 49}
          stroke="#3a2a14" strokeWidth={i % 2 ? 1 : 2} />
      })}
      {labels.map((l, i) => {
        const a = Math.PI * (1 - (i + 0.5) / 4)
        return <text key={l} x={60 + Math.cos(a) * 30} y={68 - Math.sin(a) * 30} fontSize="5.4" textAnchor="middle"
          fill={i < done ? '#7a3a10' : '#6b5a3a'} fontWeight={i < done ? 700 : 400}>{l}</text>
      })}
      <g className="gauge-needle" style={{ transform: `rotate(${angle}deg)`, transformOrigin: '60px 66px' }}>
        <line x1="60" y1="66" x2="60" y2="22" stroke="#8b1d10" strokeWidth="2.5" strokeLinecap="round" />
      </g>
      <circle cx="60" cy="66" r="5" fill="#a77d33" stroke="#5a3f12" strokeWidth="1.5" />
    </svg>
  )
}

export function StageStrip({ stages, active }: { stages: Partial<Record<Stage['key'], Stage>>; active: boolean }) {
  const done = ORDER.filter((k) => stages[k]?.ms !== undefined).length
  return (
    <div className="stage-strip panel">
      <Gauge done={done} active={active} />
      {ORDER.map((k) => {
        const s = stages[k]
        const title = { photo: 'Photo', depth: 'Depth', mask: 'Mask', model: 'Model' }[k]
        return (
          <figure key={k} className={`stage ${s?.ms !== undefined ? 'stage-done' : active ? 'stage-wait' : ''}`}>
            {s?.url ? <img src={s.url} alt={title} /> : <div className="stage-empty">{active ? '…' : ''}</div>}
            <figcaption>
              {title}
              {s?.ms !== undefined && <span className="ms">{Math.round(s.ms)} ms</span>}
            </figcaption>
          </figure>
        )
      })}
    </div>
  )
}
