import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The Flask API runs on 8090; in dev Vite proxies /api there so the frontend
// always talks to one origin. In production Flask serves the built dist itself.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8090', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,
  },
});