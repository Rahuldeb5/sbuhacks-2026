import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import type { Payload } from '../api'
import { LITE } from '../quality'

export type ViewMode = 'photo' | 'observed'

const BRASS = new THREE.Color('#e0b45a')
const STEEL = new THREE.Color('#9aa7b8')
const ASSEMBLE_S = 1.6 // bottom layer starts at 0, top layer at ASSEMBLE_S
const DROP_S = 0.45 // each voxel's fall time

const easeOutBack = (x: number) => {
  const c1 = 1.4, c3 = c1 + 1
  return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2)
}

/** Canvas-generated bump map: a rivet in each face corner + a slight edge lip (no image files, no network). */
function rivetBump(): THREE.Texture {
  const s = 128
  const c = document.createElement('canvas')
  c.width = c.height = s
  const g = c.getContext('2d')!
  g.fillStyle = '#808080'
  g.fillRect(0, 0, s, s)
  g.strokeStyle = '#5a5a5a'
  g.lineWidth = 6
  g.strokeRect(3, 3, s - 6, s - 6)
  for (const [x, y] of [[20, 20], [s - 20, 20], [20, s - 20], [s - 20, s - 20]]) {
    const rg = g.createRadialGradient(x - 2, y - 2, 1, x, y, 10)
    rg.addColorStop(0, '#ffffff')
    rg.addColorStop(0.7, '#b0b0b0')
    rg.addColorStop(1, '#808080')
    g.fillStyle = rg
    g.beginPath()
    g.arc(x, y, 10, 0, Math.PI * 2)
    g.fill()
  }
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.NoColorSpace
  return t
}

interface Props {
  payload: Payload
  mode: ViewMode
  animKey: number // change -> replay the assembly; payload changes without it snap to the final pose
  loop?: boolean // placeholder: keep re-assembling
  onAssembled?: () => void
}

export function VoxelModel({ payload, mode, animKey, loop = false, onAssembled }: Props) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const n = payload.voxels.length / 3
  const geom = useMemo(() => (LITE === 2 ? new THREE.BoxGeometry(1, 1, 1) : new RoundedBoxGeometry(1, 1, 1, 1, 0.12)), [])
  const bump = useMemo(rivetBump, [])
  const material = useMemo(() => new THREE.MeshStandardMaterial({ color: '#ffffff' }), [])

  useEffect(() => {
    if (mode === 'photo') {
      material.metalness = 0.15
      material.roughness = 0.55
      material.envMapIntensity = 1.0
      material.bumpMap = null
    } else {
      material.metalness = 0.9
      material.roughness = 0.32
      material.envMapIntensity = 1.8
      material.bumpMap = bump
      material.bumpScale = 1.2
    }
    material.needsUpdate = true
  }, [mode, material, bump])

  const data = useMemo(() => {
    const v = payload.voxel_size_mm / 10 // scene units: cm
    const [ox, oy, oz] = payload.origin_mm.map((x) => x / 10)
    const pos = new Float32Array(n * 3)
    const layer = new Float32Array(n)
    const ny = Math.max(1, payload.dims[1] - 1)
    for (let i = 0; i < n; i++) {
      const a = payload.voxels[3 * i], b = payload.voxels[3 * i + 1], c = payload.voxels[3 * i + 2]
      pos[3 * i] = ox + (a + 0.5) * v
      pos[3 * i + 1] = oy + (b + 0.5) * v
      pos[3 * i + 2] = oz + (c + 0.5) * v
      layer[i] = b / ny
    }
    return { pos, layer, v, height: payload.dims[1] * v }
  }, [payload, n])

  // per-instance color
  useEffect(() => {
    const m = mesh.current
    if (!m) return
    const col = new THREE.Color()
    for (let i = 0; i < n; i++) {
      if (mode === 'photo') {
        col.setRGB(payload.colors[3 * i] / 255, payload.colors[3 * i + 1] / 255, payload.colors[3 * i + 2] / 255, THREE.SRGBColorSpace)
      } else {
        col.copy(payload.observed[i] ? BRASS : STEEL)
      }
      m.setColorAt(i, col)
    }
    if (m.instanceColor) m.instanceColor.needsUpdate = true
  }, [payload, mode, n])

  const anim = useRef({ t0: -1, done: true, notified: false })
  const lastKey = useRef(animKey)

  const writePose = (t: number | null) => {
    const m = mesh.current
    if (!m) return
    const arr = m.instanceMatrix.array as Float32Array
    const { pos, layer, v, height } = data
    const drop = Math.max(height * 0.9, v * 8)
    const s = v * 0.97
    for (let i = 0; i < n; i++) {
      let y = pos[3 * i + 1], sc = s
      if (t !== null) {
        const p = Math.min(1, Math.max(0, (t - layer[i] * ASSEMBLE_S) / DROP_S))
        if (p <= 0) sc = 0
        else y += (1 - easeOutBack(p)) * drop
      }
      const o = 16 * i
      arr[o] = sc; arr[o + 5] = sc; arr[o + 10] = sc
      arr[o + 12] = pos[3 * i]; arr[o + 13] = y; arr[o + 14] = pos[3 * i + 2]; arr[o + 15] = 1
    }
    m.instanceMatrix.needsUpdate = true
  }

  useEffect(() => {
    if (animKey !== lastKey.current || loop) {
      lastKey.current = animKey
      anim.current = { t0: -1, done: false, notified: false }
      writePose(0)
    } else {
      anim.current.done = true
      writePose(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [payload, animKey, loop, data])

  useFrame((state) => {
    const a = anim.current
    if (a.done) return
    const now = state.clock.elapsedTime
    if (a.t0 < 0) a.t0 = now
    const t = now - a.t0
    if (t < 0) return // loop hold: keep the finished pose on screen
    if (t >= ASSEMBLE_S + DROP_S) {
      writePose(null)
      if (loop) {
        a.t0 = now + 0.7 // hold, then go again
      } else {
        a.done = true
        if (!a.notified) {
          a.notified = true
          onAssembled?.()
        }
      }
      return
    }
    writePose(t)
  })

  return (
    <instancedMesh key={n} ref={mesh} args={[geom, material, n]} frustumCulled={false} castShadow receiveShadow />
  )
}
