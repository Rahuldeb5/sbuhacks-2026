import { accessToken } from './supabase'

export type Method = 'mirror' | 'revolve' | 'inflate'

/** fetch with the signed-in user's access token (if any): scans made or saved while signed in are private. */
export async function apiFetch(input: string, init: RequestInit = {}) {
  const t = await accessToken()
  const headers = new Headers(init.headers)
  if (t) headers.set('Authorization', `Bearer ${t}`)
  return fetch(input, { ...init, headers })
}

export interface Payload {
  voxel_size_mm: number
  origin_mm: [number, number, number]
  dims: [number, number, number]
  voxels: number[]
  colors: number[]
  observed: number[]
  measurements_cm: { w: number; h: number; d: number }
  method: string
  thickness: number
  scale_source: string
  n_solid?: number
  /** mirror: symmetry plane, world frame (mm); offset_cm relative to the auto estimate */
  plane?: { point: [number, number, number]; normal: [number, number, number]; offset_cm: number }
  /** revolve: axis end points, world frame (mm) */
  axis?: { p0: [number, number, number]; p1: [number, number, number] }
}

export interface SelectionCheck {
  check: string; group: string; value: number; op: '>=' | '<='; threshold: number; pass: boolean; borderline?: boolean
}
export interface SelectionTrace {
  checks: SelectionCheck[]; revolve_ok: boolean; mirror_ok: boolean; geometric: Method; borderline: boolean
  hint: string | null; hint_favors: Method | null; hint_effect: string; final: Method
}
export interface Selection {
  recommended: Method; geometric: Method; borderline: boolean; reason: string
  selection_trace: SelectionTrace | null; trace_text: string
}
export interface SelectionHint {
  recommended: Method; switch: boolean; reason: string; hint: string | null; hint_effect: string
  selection_trace: SelectionTrace
}

/** Mirror slider bounds in model cm (min keeps the object from collapsing; max ~ half the object size). */
export type OffsetRange = [number, number]

export interface ScanResult {
  scan_id: string
  label: string
  label_source?: string
  recommended: Method
  reason: string
  scores: Record<string, Record<string, number | boolean>>
  selection?: Selection
  scale_source: string
  scale: Record<string, unknown>
  known_dim: Record<string, number> | null
  urls: { photo: string; depth: string; mask: string; thumbnail: string }
  timings: Record<string, number>
  method: Method
  thickness: number
  plane_offset_cm: number
  plane_offset_range: OffsetRange | null
  payload: Payload
  created: string
  /** signature for <img>/download URLs of this scan (they can't send the Authorization header) */
  file_sig: string
  owned: boolean
  project: { name: string; saved_at: string | null } | null
}

export interface Project {
  scan_id: string
  name: string
  method: Method
  thickness: number
  plane_offset_cm: number
  known_dim: Record<string, number> | null
  scale_source: string
  dims_cm: { w: number; h: number; d: number } | null
  created_at: string
  updated_at: string
  thumbnail: string | null
}

export interface ShapeResult {
  method: Method
  thickness: number
  plane_offset_cm: number
  plane_offset_range: OffsetRange | null
  payload: Payload
  timings: Record<string, number>
  scale_source: string
  known_dim: Record<string, number> | null
}

export type StageEvent =
  | { stage: 'photo' | 'depth' | 'mask'; url: string; ms: number; scan_id?: string; cached?: boolean }
  | { stage: 'label'; label: string; source: string }
  | { stage: 'done'; result: ScanResult }
  | { stage: 'error'; error: string }

export interface ScanSummary {
  scan_id: string
  label: string
  created: string
  method: Method
  dims_cm: { w: number; h: number; d: number } | null
  scale_source: string
  thumbnail: string
  saved: boolean
}

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw Object.assign(new Error(`${r.status} ${await r.text()}`), { status: r.status })
  return (await r.json()) as T
}

/** POST /api/scan?stream=1 and call onEvent for every NDJSON stage line as it arrives. */
export async function scanStream(file: File, knownDim: string | null, onEvent: (e: StageEvent) => void) {
  const fd = new FormData()
  fd.append('image', file)
  if (knownDim) fd.append('known_dim', knownDim)
  const r = await apiFetch('/api/scan?stream=1', { method: 'POST', body: fd })
  if (!r.ok || !r.body) throw new Error(`scan failed: ${r.status} ${await r.text()}`)
  const reader = r.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let i
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim()
      buf = buf.slice(i + 1)
      if (line) onEvent(JSON.parse(line) as StageEvent)
    }
  }
}

/** knownDim: undefined = leave as is, null = clear, "h=6" = set. */
export async function shape(scanId: string, method: Method, thickness: number, knownDim?: string | null, planeOffsetCm = 0) {
  const body: Record<string, unknown> = { scan_id: scanId, method, thickness, plane_offset_cm: planeOffsetCm }
  if (knownDim !== undefined) body.known_dim = knownDim ?? ''
  return j<ShapeResult>(await apiFetch('/api/shape', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }))
}

export const listScans = async () => j<ScanSummary[]>(await apiFetch('/api/scans'))
export const getScan = async (id: string) => j<ScanResult>(await apiFetch(`/api/scans/${id}`))

export async function postThumbnail(id: string, blob: Blob) {
  await apiFetch(`/api/scans/${id}/thumbnail`, { method: 'POST', headers: { 'Content-Type': blob.type || 'image/jpeg' }, body: blob })
}

export const exportUrl = (id: string, sig: string, style: 'blocky' | 'smooth', method: string, thickness: number, planeOffsetCm = 0) =>
  `/api/export/${id}?style=${style}&method=${method}&thickness=${thickness.toFixed(2)}&plane_offset_cm=${planeOffsetCm.toFixed(2)}&s=${sig}`

export interface Description {
  scan_id: string
  status: 'pending' | 'ready' | 'fallback' | 'none'
  result?: {
    name: string
    description: string
    shape_category: string
    method_explanation: string
    suggested_method: string
    fallback?: string
  }
  meta?: { ms?: number; agrees_with_auto?: boolean; error?: string }
  auto: { recommended: string; reason: string; scores: Record<string, Record<string, number | boolean>> }
  prior_scale?: number
  selection?: Selection
  selection_hint?: SelectionHint | null
}

export const getDescription = async (id: string) => j<Description>(await apiFetch(`/api/scans/${id}/description`))

export interface SaveBody { scan_id: string; name?: string; method: Method; thickness: number; plane_offset_cm: number }
const json = (method: string, body: unknown): RequestInit =>
  ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
export const saveProject = async (b: SaveBody) => j<Project>(await apiFetch('/api/projects', json('POST', b)))
export const listProjects = async () => j<Project[]>(await apiFetch('/api/projects'))
export const renameProject = async (id: string, name: string) =>
  j<Project>(await apiFetch(`/api/projects/${id}`, json('PATCH', { name })))
export const deleteProject = async (id: string) => j<{ ok: boolean }>(await apiFetch(`/api/projects/${id}`, { method: 'DELETE' }))
