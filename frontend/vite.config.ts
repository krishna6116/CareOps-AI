import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => ({
  // The static GitHub Pages demo is served under /CareOps-AI/; VITE_BASE is
  // set in .env.demo (npm run build:demo). Other builds use the site root.
  base: loadEnv(mode, '.', '').VITE_BASE || '/',
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
  },
}));
