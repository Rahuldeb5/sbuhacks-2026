// Scripted UI check through the real app (Vite dev server + FastAPI must be running).
//   LD_LIBRARY_PATH=$HOME/.local/lib/chromium-deps OUT=/some/dir node scripts/ui_check.mjs
// Saves screenshots to $OUT and prints a JSON report. Headless Chromium renders WebGL with SwiftShader (CPU),
// so FPS here is a floor, not what a GPU or phone browser gets.
import { chromium } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'

const BASE = process.env.BASE ?? 'http://localhost:5173'
const Q = process.env.Q ?? '?lite=2' // plain boxes, no AO: SwiftShader is far too slow for the full renderer
const OUT = process.env.OUT ?? path.resolve(import.meta.dirname, '../ui_shots')
const DATA = path.resolve(import.meta.dirname, '../../data')
fs.mkdirSync(OUT, { recursive: true })
const report = { steps: [], console_errors: [] }
const step = (name, extra = {}) => { report.steps.push({ name, ...extra }); console.log('✓', name, JSON.stringify(extra)) }

const browser = await chromium.launch({ args: ['--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader', '--ignore-gpu-blocklist'] })
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, acceptDownloads: true })
const page = await ctx.newPage()
page.on('console', (m) => { if (m.type() === 'error') report.console_errors.push(m.text()) })
page.on('pageerror', (e) => report.console_errors.push(String(e)))

const shot = async (name) => page.screenshot({ path: path.join(OUT, name), timeout: 120000 })
const dims = () => page.locator('.measure .dims').innerText()
const shapeResp = () => page.waitForResponse((r) => r.url().includes('/api/shape'))
const serverMs = async (r) => (await r.json()).timings.total_ms
// range inputs reject fill() for values off the step grid: set it like a user drag would (native setter + input event)
const setRange = (loc, v) => loc.evaluate((el, v) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v))
  el.dispatchEvent(new Event('input', { bubbles: true }))
}, v)

await page.goto(BASE + Q)
await page.waitForSelector('canvas')
await page.waitForTimeout(2500)
await shot('01_idle.png')
step('idle page with the assembling placeholder cog')

for (const tag of ['duck', 'bottle']) {
  const t0 = Date.now()
  await page.setInputFiles('#photo-file', path.join(DATA, `${tag}.jpeg`))
  await page.waitForSelector('.stage-done img[alt=Depth]', { timeout: 30000 })
  await page.waitForSelector('.controls', { timeout: 60000 })
  const scanMs = Date.now() - t0
  await page.waitForTimeout(2600)                    // assembly animation
  await shot(`02_${tag}_auto.png`)
  const auto = (await page.locator('.method-row .active').innerText()).replace('\n', ' ')
  step(`${tag}: scanned via UI`, { scan_ms_wall: scanMs, auto_method: auto, dims: await dims() })
  if (process.env.VLM === '1') {          // live VLM: the Inspector's notes replace the fallback name
    await page.waitForSelector('.inspector-name', { timeout: 30000 })
    await page.waitForTimeout(500)
    await page.locator('.inspector').scrollIntoViewIfNeeded()
    await shot(`02b_${tag}_inspector.png`)
    step(`${tag}: inspector notes`, { name: await page.locator('.inspector-name').innerText(),
      hud: await page.locator('.hud-title').innerText(), tags: await page.locator('.inspector-tags').innerText() })
  }

  for (const m of ['Mirror', 'Revolve']) {
    const tt = Date.now()
    const resp = shapeResp()
    await page.getByRole('button', { name: new RegExp(`^${m}`) }).click()
    const r = await resp
    const rt = Date.now() - tt
    await page.waitForTimeout(800)
    await shot(`03_${tag}_${m.toLowerCase()}.png`)
    step(`${tag}: ${m}`, { status: r.status(), server_ms: await serverMs(r), browser_roundtrip_ms: rt, dims: await dims() })
  }
  await page.locator('details.advanced summary').click()
  let resp = shapeResp()
  await page.getByRole('button', { name: /^Fallback shape/ }).click()
  let r = await resp
  await page.waitForTimeout(800)
  step(`${tag}: Fallback shape`, { status: r.status(), server_ms: await serverMs(r), dims: await dims() })
  resp = shapeResp()
  await setRange(page.locator('input[aria-label=thickness]'), 1.8)
  r = await resp
  await page.waitForTimeout(800)
  step(`${tag}: thickness 1.8`, { server_ms: await serverMs(r), dims: await dims() })

  // known height, then the mirror slider at true scale
  const known = tag === 'duck' ? '6' : '28'
  await page.locator('.known-row input[type=number]').fill(known)
  resp = shapeResp()
  await page.getByRole('button', { name: 'Apply' }).click()
  await resp
  await page.waitForTimeout(800)
  step(`${tag}: known height ${known} cm`, { dims: await dims(), scale: await page.locator('.scale-src').innerText() })
  resp = shapeResp()
  await page.getByRole('button', { name: /^Mirror/ }).click()
  const srvRange = (await (await resp).json()).plane_offset_range
  await page.waitForTimeout(800)
  const slider = page.locator('input[aria-label="symmetry plane depth"]')
  const [min, max, stp] = await Promise.all(['min', 'max', 'step'].map((a) => slider.getAttribute(a)))
  const zeroOnGrid = Math.abs(Math.round(-Number(min) / Number(stp)) * Number(stp) + Number(min)) < 1e-9
  resp = shapeResp()
  await setRange(slider, Math.min(1, Number(max)))
  const rr = await (await resp).json()
  await page.waitForTimeout(2600)
  await shot(`04_${tag}_mirror_known_offset.png`)
  step(`${tag}: mirror slider`, { range: [min, max], step: stp, server_range: srvRange,
    matches_server: Number(min) <= srvRange[0] && srvRange[0] - Number(min) < Number(stp) &&
      Number(max) >= srvRange[1] && Number(max) - srvRange[1] < Number(stp), zero_reachable: zeroOnGrid, server_ms: rr.timings.total_ms, sent_offset: rr.plane_offset_cm, dims: await dims() })

  const dl = page.waitForEvent('download')
  await page.getByRole('link', { name: /Smooth STL/ }).click()
  const d = await dl
  const f = path.join(OUT, `${tag}.stl`)
  await d.saveAs(f)
  step(`${tag}: smooth STL download`, { file: d.suggestedFilename(), bytes: fs.statSync(f).size })
  await shot(`05_${tag}_final_sidebar.png`)
}

await page.goto(BASE + Q + '#/gallery')
await page.waitForSelector('.gallery .card', { timeout: 10000 })
await page.waitForTimeout(1000)
await shot('06_archive.png')
step('archive', { cards: await page.locator('.gallery .card').count() })
await page.locator('.gallery .card .brass-btn').first().click()
await page.waitForSelector('.controls', { timeout: 30000 })
await page.waitForTimeout(2600)
await shot('07_reopened.png')
step('reopen from archive', { method: (await page.locator('.method-row .active').innerText()).replace('\n', ' '), dims: await dims() })

await page.setViewportSize({ width: 390, height: 844 })
await page.waitForTimeout(1500)
await shot('08_phone_width.png')
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
step('phone width', { horizontal_overflow_px: overflow })

console.log(JSON.stringify({ console_errors: report.console_errors }, null, 1))
await browser.close()
