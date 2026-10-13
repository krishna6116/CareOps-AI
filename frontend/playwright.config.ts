import { defineConfig } from '@playwright/test';

// E2E_MODE=live (default): the real FastAPI backend plus a production
// preview of the dashboard. E2E_MODE=demo: the static GitHub Pages build
// (run `npm run build:demo` first). E2E_BROWSER_CHANNEL=chrome or msedge
// uses an installed browser instead of Playwright's bundled Chromium.
// E2E_BASE_URL tests an already deployed site (e.g. the GitHub Pages demo,
// with E2E_MODE=demo) and starts no local servers.
const demo = process.env.E2E_MODE === 'demo';
const deployed = process.env.E2E_BASE_URL;
const python = process.env.E2E_PYTHON ?? 'python';
const baseURL = deployed ?? (demo ? 'http://127.0.0.1:4174/CareOps-AI/' : 'http://127.0.0.1:4173/');

export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  expect: { timeout: 30_000 },
  workers: 1,
  reporter: [['list']],
  outputDir: 'e2e-results',
  use: {
    baseURL,
    channel: process.env.E2E_BROWSER_CHANNEL || undefined,
    viewport: { width: 1440, height: 900 },
    trace: 'retain-on-failure',
  },
  webServer: deployed ? [] : demo
    ? [{
        command: 'npx vite preview --mode demo --port 4174 --strictPort',
        url: baseURL,
        reuseExistingServer: false,
        timeout: 60_000,
      }]
    : [
        {
          command: `${python} -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`,
          cwd: '..',
          url: 'http://127.0.0.1:8000/ready',
          env: { CAREOPS_DATABASE_URL: 'sqlite:///./e2e-careops.db', CAREOPS_CORS_ORIGINS: '["http://127.0.0.1:4173"]' },
          reuseExistingServer: false,
          timeout: 120_000,
        },
        {
          command: 'npx vite preview --port 4173 --strictPort',
          url: baseURL,
          reuseExistingServer: false,
          timeout: 60_000,
        },
      ],
});
