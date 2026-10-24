import { ContactShadows, Environment, Lightformer, Line, OrbitControls } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { EffectComposer, N8AO, ToneMapping } from '@react-three/postprocessing'
import { ToneMappingMode } from 'postprocessing'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import type { Payload } from '../api'
import { LITE } from '../quality'
import { VoxelModel, type ViewMode } from './VoxelModel'

interface Props {
  payload: Payload
  mode: ViewMode
  animKey: number
  loop?: boolean
  fitKey: string // refit the camera when this changes (new scan), not on every slider move
  showDims: boolean
  onAssembled?: () => void
  snapshotKey?: number
  onSnapshot?: (b: Blob) => void
  onFps?: (fps: number) => void
  showGuide?: boolean
}

function FitCamera({ size, fitKey }: { size: [number, number, number]; fitKey: string }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null
  useEffect(() => {
    const [w, h, d] = size
    const r = Math.max(w, h, d, 1)
    const dist = (r * 1.25) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))
    const dir = new THREE.Vector3(0.55, 0.42, 0.85).normalize()
    camera.position.copy(dir.multiplyScalar(dist)).add(new THREE.Vector3(0, h / 2, 0))
    camera.near = r / 200
    camera.far = r * 200
    camera.updateProjectionMatrix()
    if (controls) {
      controls.target.set(0, h / 2, 0)
      controls.update()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitKey, controls])
  return null
}

/** Dimension label as a canvas-texture sprite (no DOM overlay, no font download). */
function LabelSprite({ text, position, height }: { text: string; position: [number, number, number]; height: number }) {
  const tex = useMemo(() => {
    const c = document.createElement('canvas')
    const g = c.getContext('2d')!
    const font = 'bold 44px system-ui, sans-serif'
    g.font = font
    const w = Math.ceil(g.measureText(text).width) + 36
    c.width = w
    c.height = 64
    const grad = g.createLinearGradient(0, 0, 0, 64)
    grad.addColorStop(0, '#ecd08d')
    grad.addColorStop(1, '#c99a3e')
    g.fillStyle = grad
    g.strokeStyle = '#7a5a1e'
    g.lineWidth = 4
    g.beginPath()
    g.roundRect(2, 2, w - 4, 60, 10)
    g.fill()
    g.stroke()
    g.font = font
    g.fillStyle = '#1d1406'
    g.textBaseline = 'middle'
    g.fillText(text, 18, 34)
    const t = new THREE.CanvasTexture(c)
    t.colorSpace = THREE.SRGBColorSpace
    t.anisotropy = 4
    return { t, aspect: w / 64 }
  }, [text])
  useEffect(() => () => tex.t.dispose(), [tex])
  return (
    <sprite position={position} scale={[height * tex.aspect, height, 1]} renderOrder={10}>
      <spriteMaterial map={tex.t} depthTest={false} transparent toneMapped={false} />
    </sprite>
  )
}

function Dimensions({ m }: { m: { w: number; h: number; d: number } }) {
  const { w, h, d } = m
  const g = Math.max(w, h, d) * 0.06
  const col = '#e8c67a'
  const tick = g * 0.4
  const lines: { pts: [number, number, number][]; label: string; at: [number, number, number] }[] = [
    { pts: [[-w / 2, 0, d / 2 + g], [w / 2, 0, d / 2 + g]], label: `W ${w.toFixed(1)} cm`, at: [0, 0, d / 2 + g * 1.8] },
    { pts: [[-w / 2 - g, 0, d / 2], [-w / 2 - g, h, d / 2]], label: `H ${h.toFixed(1)} cm`, at: [-w / 2 - g * 1.8, h / 2, d / 2] },
    { pts: [[w / 2 + g, 0, -d / 2], [w / 2 + g, 0, d / 2]], label: `D ${d.toFixed(1)} cm`, at: [w / 2 + g * 1.8, 0, 0] },
  ]
  return (
    <group>
      {lines.map((l, i) => {
        const [a, b] = l.pts
        const dir = new THREE.Vector3(...b).sub(new THREE.Vector3(...a)).normalize()
        const perp = Math.abs(dir.y) > 0.5 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0)
        const t1 = new THREE.Vector3(...a).addScaledVector(perp, tick).toArray() as [number, number, number]
        const t2 = new THREE.Vector3(...a).addScaledVector(perp, -tick).toArray() as [number, number, number]
        const t3 = new THREE.Vector3(...b).addScaledVector(perp, tick).toArray() as [number, number, number]
        const t4 = new THREE.Vector3(...b).addScaledVector(perp, -tick).toArray() as [number, number, number]
        return (
          <group key={i}>
            <Line points={[a, b]} color={col} lineWidth={2} />
            <Line points={[t1, t2]} color={col} lineWidth={2} />
            <Line points={[t3, t4]} color={col} lineWidth={2} />
            <LabelSprite text={l.label} position={l.at} height={Math.max(w, h, d) * 0.055} />
          </group>
        )
      })}
    </group>
  )
}

const GUIDE_BRASS = '#e6be6a'

/** Mirror: the symmetry plane as a translucent brass sheet with a thin rim. Payload units are mm. */
function PlaneGuide({ plane, size }: { plane: NonNullable<Payload['plane']>; size: number }) {
  const pos = plane.point.map((x) => x / 10) as [number, number, number]
  const q = useMemo(
    () => new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(...plane.normal).normalize()),
    [plane.normal],
  )
  const h = size / 2
  const rim: [number, number, number][] = [[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0], [-h, -h, 0]]
  return (
    <group position={pos} quaternion={q}>
      <mesh renderOrder={5}>
        <planeGeometry args={[size, size]} />
        <meshStandardMaterial color={GUIDE_BRASS} metalness={0.6} roughness={0.35} transparent opacity={0.22}
          side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <Line points={rim} color={GUIDE_BRASS} lineWidth={2} />
    </group>
  )
}

/** Revolve: the axis as a thin brass rod (extended a little past both ends). */
function AxisGuide({ axis, radius }: { axis: NonNullable<Payload['axis']>; radius: number }) {
  const a = new THREE.Vector3(...axis.p0).multiplyScalar(0.1)
  const b = new THREE.Vector3(...axis.p1).multiplyScalar(0.1)
  const dir = b.clone().sub(a)
  const len = dir.length() * 1.15
  const mid = a.clone().add(b).multiplyScalar(0.5)
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize())
  return (
    <mesh position={mid} quaternion={q} renderOrder={5}>
      <cylinderGeometry args={[radius, radius, len, 12]} />
      <meshStandardMaterial color={GUIDE_BRASS} metalness={0.9} roughness={0.3} />
    </mesh>
  )
}

function Snapshotter({ snapshotKey, onSnapshot }: { snapshotKey?: number; onSnapshot?: (b: Blob) => void }) {
  const gl = useThree((s) => s.gl)
  useEffect(() => {
    if (!snapshotKey || !onSnapshot) return
    const id = requestAnimationFrame(() => {
      // centred square crop (the camera frames the object there), downscaled into a small 2D canvas + JPEG:
      // a full-size PNG toBlob took 5 s under software GL
      const src = gl.domElement
      const side = Math.round(Math.min(src.width, src.height) * 0.8)
      const sx = Math.round((src.width - side) / 2), sy = Math.round((src.height - side) / 2)
      const c = document.createElement('canvas')
      c.width = c.height = Math.min(360, side)
      c.getContext('2d')!.drawImage(src, sx, sy, side, side, 0, 0, c.width, c.height)
      c.toBlob((b) => b && onSnapshot(b), 'image/jpeg', 0.85)
    })
    return () => cancelAnimationFrame(id)
  }, [snapshotKey, onSnapshot, gl])
  return null
}

function FpsMeter({ onFps }: { onFps?: (fps: number) => void }) {
  const acc = useRef({ frames: 0, t: performance.now(), logged: 0 })
  useFrame(() => {
    const a = acc.current
    a.frames++
    const now = performance.now()
    if (now - a.t >= 1000) {
      const fps = (a.frames * 1000) / (now - a.t)
      ;(window as unknown as { __fps: number }).__fps = fps
      onFps?.(fps)
      if (import.meta.env.DEV && now - a.logged > 5000) {
        console.log(`[viewer] ${fps.toFixed(1)} fps`)
        a.logged = now
      }
      a.frames = 0
      a.t = now
    }
  })
  return null
}

export function Viewer(p: Props) {
  const m = p.payload.measurements_cm
  const size: [number, number, number] = [m.w, m.h, m.d]
  const r = Math.max(m.w, m.h, m.d, 1)
  return (
    <Canvas
      className="viewer-canvas"
      dpr={LITE ? 1 : [1, 2]}
      gl={{ preserveDrawingBuffer: true, antialias: false, powerPreference: 'high-performance' }}
      camera={{ fov: 38, position: [30, 20, 40] }}
    >
      <color attach="background" args={['#15110d']} />
      <fog attach="fog" args={['#15110d', r * 6, r * 14]} />
      {/* Studio environment generated from light panels: no HDR download, works offline. */}
      <Environment resolution={256} frames={1}>
        <Lightformer form="rect" intensity={3} color="#fff1d6" position={[0, 5, -6]} scale={[10, 5, 1]} />
        <Lightformer form="rect" intensity={2.2} color="#ffd59a" position={[-6, 2, 3]} rotation-y={Math.PI / 2} scale={[8, 3, 1]} />
        <Lightformer form="rect" intensity={1.3} color="#b9d0ff" position={[6, 1, 2]} rotation-y={-Math.PI / 2} scale={[8, 2, 1]} />
        <Lightformer form="ring" intensity={2} color="#ffffff" position={[0, 8, 0]} rotation-x={Math.PI / 2} scale={4} />
      </Environment>
      <directionalLight position={[r, r * 2, r * 1.5]} intensity={0.8} color="#fff0dc" />
      <VoxelModel payload={p.payload} mode={p.mode} animKey={p.animKey} loop={p.loop} onAssembled={p.onAssembled} />
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.02, 0]}>
        <circleGeometry args={[r * 3, 64]} />
        <meshStandardMaterial color="#2b231a" roughness={0.92} metalness={0.05} />
      </mesh>
      <ContactShadows position={[0, 0.005, 0]} scale={r * 4} far={r} blur={2.4} opacity={0.7} resolution={LITE ? 256 : 512} frames={LITE === 2 ? 1 : Infinity} />
      {p.showDims && <Dimensions m={m} />}
      {p.showGuide && p.payload.plane && <PlaneGuide plane={p.payload.plane} size={r * 1.25} />}
      {p.showGuide && p.payload.axis && <AxisGuide axis={p.payload.axis} radius={Math.max(0.05, r * 0.008)} />}
      <OrbitControls makeDefault enableDamping dampingFactor={0.12} />
      <FitCamera size={size} fitKey={p.fitKey} />
      <Snapshotter snapshotKey={p.snapshotKey} onSnapshot={p.onSnapshot} />
      <FpsMeter onFps={p.onFps} />
      {LITE === 0 ? (
        <EffectComposer multisampling={4}>
          <N8AO aoRadius={Math.max(0.3, r * 0.06)} intensity={2.2} distanceFalloff={1} halfRes />
          <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
        </EffectComposer>
      ) : null}
    </Canvas>
  )
}
