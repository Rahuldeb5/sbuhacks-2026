import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// --host: a phone on the same Wi-Fi opens http://<lan-ip>:5173 ; /api is proxied to the FastAPI backend,
// so the browser only ever talks to one origin (no CORS, one port to open).
const proxy = { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } }

export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173, proxy },
  preview: { host: true, port: 4173, proxy },
})
