// Sign-in + save flow through the real app (Vite + FastAPI running) against the real Supabase project.
//   LD_LIBRARY_PATH=$HOME/.local/lib/chromium-deps OUT=/some/dir node scripts/flow_auth.mjs
// Google can't be scripted, so a throwaway email/password user is created with the admin API, signed in, and its
// session is put where supabase-js keeps it (localStorage) - to the app that is the same as a Google sign-in.
// The user, its projects and its Storage files are deleted at the end.
import { chromium } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'

const BASE = process.env.BASE ?? 'http://localhost:5173'
const Q = process.env.Q ?? '?lite=2'
const OUT = process.env.OUT ?? path.resolve(import.meta.dirname, '../ui_shots')
const DATA = path.resolve(import.meta.dirname, '../../data')
const env = Object.fromEntries(fs.readFileSync(path.resolve(import.meta.dirname, '../../.env'), 'utf8').split('\n')
  .filter((l) => l.includes('=')).map((l) => { const i = l.indexOf('='); return [l.slice(0, i).trim(), l.slice(i + 1).trim().replace(/^"|"$/g, '')] }))
const URL_ = env.SUPABASE_URL, SECRET = env.SUPABASE_KEY, PUB = env.VITE_SUPABASE_PUBLISHABLE_KEY
const admin = { apikey: SECRET, Authorization: `Bearer ${SECRET}`, 'Content-Type': 'application/json' }
fs.mkdirSync(OUT, { recursive: true })
const report = { steps: [], console_errors: [] }
const step = (name, extra = {}) => { report.steps.push({ name, ...extra }); console.log('✓', name, JSON.stringify(extra)) }
const check = (cond, msg) => { if (!cond) throw new Error(`check failed: ${msg}`) }

const email = `sns-ui-${Date.now()}@example.com`, password = `pw-${Math.random().toString(36).slice(2)}-X9`
const user = await (await fetch(`${URL_}/auth/v1/admin/users`, { method: 'POST', headers: admin,
  body: JSON.stringify({ email, password, email_confirm: true, user_metadata: { full_name: 'Ada Lovelace' } }) })).json()
const browser = await chromium.launch({ args: ['--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader', '--ignore-gpu-blocklist'] })
try {
  const session = await (await fetch(`${URL_}/auth/v1/token?grant_type=password`, { method: 'POST',
    headers: { apikey: PUB, 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password }) })).json()
  check(session.access_token, 'password sign-in')
  const storageKey = `sb-${new URL(URL_).hostname.split('.')[0]}-auth-token`

  // ---- signed out: Save offers sign-in; the Google button really goes to Google
  const anon = await browser.newContext({ viewport: { width: 1440, height: 900 } })
  const ap = await anon.newPage()
  ap.on('pageerror', (e) => report.console_errors.push(String(e)))
  await ap.goto(BASE + Q)
  await ap.getByRole('button', { name: 'Sign in with Google' }).waitFor()
  await ap.setInputFiles('input[type=file]', path.join(DATA, 'bottle.jpeg'))
  await ap.waitForSelector('.savebar', { timeout: 60000 })
  const anonBtn = await ap.locator('.savebar .brass-btn').innerText()
  check(anonBtn === 'Sign in & save', `anonymous save button: ${anonBtn}`)
  step('signed out: scan works, Save says "Sign in & save"')
  await ap.getByRole('button', { name: 'Sign in with Google' }).click()
  await ap.waitForURL(/accounts\.google\.com/, { timeout: 20000 })
  step('Sign in with Google reaches Google', { host: new URL(ap.url()).host, has_supabase_callback: ap.url().includes(encodeURIComponent(`${URL_}/auth/v1/callback`)) })
  await anon.close()

  // ---- signed in (session injected)
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, acceptDownloads: true })
  await ctx.addInitScript(([k, v]) => { if (!localStorage.getItem(k)) localStorage.setItem(k, v) }, [storageKey, JSON.stringify(session)])
  const page = await ctx.newPage()
  page.on('console', (m) => { if (m.type() === 'error') report.console_errors.push(m.text()) })
  page.on('pageerror', (e) => report.console_errors.push(String(e)))
  page.on('dialog', (d) => d.type() === 'prompt' ? d.accept('Brass Duck Mk II') : d.accept())
  const shot = (n) => page.screenshot({ path: path.join(OUT, n), timeout: 120000 })
  await page.goto(BASE + Q)
  await page.locator('.account-name').waitFor()
  step('signed in', { account: await page.locator('.account-name').innerText() })

  await page.setInputFiles('input[type=file]', path.join(DATA, 'duck.jpeg'))
  await page.waitForSelector('.savebar', { timeout: 60000 })
  await page.waitForTimeout(2600)
  const state0 = await page.locator('.save-state').innerText()
  check(state0.includes('draft'), state0)
  const save = page.waitForResponse((r) => r.url().includes('/api/projects') && r.request().method() === 'POST')
  await page.locator('.savebar .brass-btn').click()
  const sr = await save
  check(sr.status() === 200, `save ${sr.status()}`)
  await page.locator('.save-state.clean').waitFor()
  await shot('a1_saved.png')
  step('saved', { name: await page.locator('.name-input').inputValue(), state: await page.locator('.save-state').innerText() })

  const resp = page.waitForResponse((r) => r.url().includes('/api/shape'))
  await page.getByRole('button', { name: /^Revolve/ }).click()
  await resp
  await page.locator('.save-state.dirty').waitFor()
  step('changed method -> unsaved changes', { state: await page.locator('.save-state').innerText() })
  await page.locator('.savebar .brass-btn').click()
  await page.locator('.save-state.clean').waitFor({ timeout: 30000 })
  step('re-saved')

  // the STL link carries the file signature: it works for this private scan
  const dl = page.waitForEvent('download')
  await page.getByRole('link', { name: /Blocky STL/ }).click()
  const f = path.join(OUT, 'private.stl')
  await (await dl).saveAs(f)
  step('private scan STL download', { bytes: fs.statSync(f).size })

  await page.goto(BASE + Q + '#/gallery')
  await page.locator('.gallery .card').first().waitFor({ timeout: 20000 })
  await page.waitForFunction(() => { const i = document.querySelector('.gallery .card img'); return i && i.complete && i.naturalWidth > 0 }, null, { timeout: 20000 })
  await shot('a2_archive.png')
  step('archive', { cards: await page.locator('.gallery .card').count(), title: await page.locator('.card-title').first().innerText(),
    meta: await page.locator('.card-meta').first().innerText() })
  await page.getByRole('button', { name: 'Rename' }).first().click()
  await page.locator('.card-title', { hasText: 'Brass Duck Mk II' }).waitFor()
  step('renamed via prompt')
  await page.getByRole('button', { name: 'Open' }).first().click()
  await page.waitForSelector('.controls', { timeout: 30000 })
  await page.waitForTimeout(2600)
  await shot('a3_reopened.png')
  step('reopened', { method: (await page.locator('.method-row .active').innerText()).replace('\n', ' '),
    name: await page.locator('.name-input').inputValue(), state: await page.locator('.save-state').innerText() })

  await page.getByRole('button', { name: 'Sign out' }).click()
  await page.getByRole('button', { name: 'Sign in with Google' }).waitFor()
  await page.waitForTimeout(500)
  check(await page.locator('.controls').count() === 0, 'private scan still on screen after sign-out')
  step('signed out: private scan cleared from the screen')
  await page.goto(BASE + Q + '#/gallery')
  await page.locator('.gallery-empty').waitFor()
  step('archive asks to sign in')
  await shot('a4_signed_out_archive.png')
} finally {
  const projects = await (await fetch(`${URL_}/rest/v1/projects?user_id=eq.${user.id}&select=scan_id`, { headers: admin })).json()
  for (const p of projects) {
    const files = await (await fetch(`${URL_}/storage/v1/object/list/scans`, { method: 'POST', headers: admin,
      body: JSON.stringify({ prefix: `${user.id}/${p.scan_id}/` }) })).json()
    if (files.length) await fetch(`${URL_}/storage/v1/object/scans`, { method: 'DELETE', headers: admin,
      body: JSON.stringify({ prefixes: files.map((o) => `${user.id}/${p.scan_id}/${o.name}`) }) })
  }
  await fetch(`${URL_}/auth/v1/admin/users/${user.id}`, { method: 'DELETE', headers: admin })
  console.log('cleaned up test user + files')
  await browser.close()
}
console.log(JSON.stringify({ console_errors: report.console_errors }, null, 1))
