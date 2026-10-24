import type { Payload } from './api'

function surfaceOnly(occ: Uint8Array, nx: number, ny: number, nz: number, color: (i: number, j: number, k: number) => number[]): Payload {
  const at = (i: number, j: number, k: number) =>
    i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz ? 0 : occ[(i * ny + j) * nz + k]
  const voxels: number[] = []
  const colors: number[] = []
  const observed: number[] = []
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++)
      for (let k = 0; k < nz; k++) {
        if (!at(i, j, k)) continue
        if (at(i + 1, j, k) && at(i - 1, j, k) && at(i, j + 1, k) && at(i, j - 1, k) && at(i, j, k + 1) && at(i, j, k - 1)) continue
        voxels.push(i, j, k)
        colors.push(...color(i, j, k))
        observed.push(k > nz / 2 ? 1 : 0)
      }
  const v = 2
  return {
    voxel_size_mm: v, origin_mm: [(-nx * v) / 2, 0, (-nz * v) / 2], dims: [nx, ny, nz], voxels, colors, observed,
    measurements_cm: { w: (nx * v) / 10, h: (ny * v) / 10, d: (nz * v) / 10 }, method: 'placeholder', thickness: 1,
    scale_source: 'placeholder',
  }
}

/** A brass cog standing on edge: shown (assembling on loop) while /api/scan works. */
export function cogPayload(): Payload {
  const R = 24, r0 = 9, teeth = 10, depth = 8
  const nx = 2 * R + 2, ny = 2 * R + 2, nz = depth
  const occ = new Uint8Array(nx * ny * nz)
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++) {
      const x = i - nx / 2 + 0.5, y = j - ny / 2 + 0.5
      const rad = Math.hypot(x, y), ang = Math.atan2(y, x)
      const tooth = Math.cos(teeth * ang) > 0.25
      const outer = tooth ? R : R - 4
      const spoke = Math.abs(Math.sin(3 * ang)) < 0.22
      const inRing = rad <= outer && rad >= R - 9
      const inHub = rad <= r0 && rad >= 3.5
      const inSpoke = spoke && rad < R - 8 && rad > r0 - 1
      if (inRing || inHub || inSpoke) for (let k = 0; k < nz; k++) occ[(i * ny + j) * nz + k] = 1
    }
  return surfaceOnly(occ, nx, ny, nz, () => [196, 150, 70])
}

/** Dev bench: ?bench=30000 -> roughly that many surface voxels (a hollow-ish blob) to check FPS. */
export function benchPayload(target: number): Payload {
  const side = Math.max(8, Math.round(Math.sqrt(target / 6)))
  const n = side
  const occ = new Uint8Array(n * n * n).fill(1)
  return surfaceOnly(occ, n, n, n, (i, j, k) => [80 + ((i * 7) % 150), 90 + ((j * 5) % 140), 100 + ((k * 3) % 120)])
}
