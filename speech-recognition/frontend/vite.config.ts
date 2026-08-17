import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  // 'spa' mode enables history API fallback so refreshing /predict, /about, etc.
  // returns index.html instead of 404 — required for React Router BrowserRouter.
  appType: 'spa',
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      '/ws': {
        target: 'ws://127.0.0.1:8001',
        ws: true
      }
    }
  }
})
