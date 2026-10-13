# Frontend dashboard

React 18 + TypeScript + Vite 8, using React Router. The dashboard reads the
FastAPI service at `VITE_API_URL` (default `http://localhost:8000`).

| Page | Shows |
|---|---|
| Executive overview | Snapshot, forecast and replay KPIs, each labelled as one of those three; forecast charts; recommendations; alerts |
| Forecasting | Horizon selector (7/14/30 days); history vs forecast charts with the 80% bound; model evaluation table (published vs baseline, interval coverage, alert thresholds); LOS and no-show samples |
| Bed management | Department inventory; request-level allocation replay (baseline vs CP-SAT, reasons, unassigned requests by policy, sensitivity); availability forecast |
| Staff scheduling / Appointments | Optimized schedules and coverage; no-show risk bands computed on the server |
| Optimization | Department/day plan, emergency capacity, KPIs, scorecard, run manifest |
| Alerts | Audit and saved alerts with a severity filter |
| What-if simulator | Percentage inputs validated against the API limits; baseline/scenario/change table; states what was rerun |
| AI assistant | Example questions from the brief; answer with typed evidence, sources, limitations and any fallback reason |
| Audit / run history | Run manifest, data-quality audit, assistant audit, simulation history |

## Presentation rules

`src/format.ts` sets how numbers are shown:

- Counts (beds, admissions, staff, requests) are shown as whole numbers,
  rounded half away from zero and never negative. This applies to KPI cards,
  chart axes and tooltips.
- Table cells print integral numbers without decimals and keep up to three
  decimals for metrics and probabilities.
- Rates are shown as percentages.

The API keeps full precision. Rounding happens only at presentation.

Failed API calls show the server's error message with a retry button. Empty
results show an empty state. Nothing is replaced with demo values.

## Development and checks

Node 24 (Vitest 5 and jsdom need Node 22.12 or later):

```powershell
Set-Location frontend
npm ci
$env:VITE_API_URL = "http://localhost:8000"
npm run dev          # http://localhost:5173
npm run typecheck    # tsc -b
npm test             # Vitest + Testing Library
npm run build        # production bundle in dist/
```

The tests (`src/*.test.ts[x]`) cover:

- count formatting
- chart axis and tooltip labels
- table filtering and pagination
- loading, error and empty states
- simulator validation and request payloads
- the assistant evidence card and its error handling

## Browser end-to-end tests

`e2e/dashboard.spec.ts` (Playwright) checks that:

- every page opens with no error states and no console errors, and saves a
  full-page screenshot to `e2e-screenshots/`;
- overview counts have no decimals;
- the assistant answers an example question with evidence;
- a simulator preset runs and shows the baseline comparison;
- bed allocation lists unassigned requests for both policies;
- invalid simulator input is blocked before any request.

```powershell
npm run build                      # live mode: starts the API and a preview server
$env:E2E_PYTHON = ".venv\Scripts\python.exe"     # relative to the repo root; CI uses "python"
$env:E2E_BROWSER_CHANNEL = "chrome" # or msedge; omit to use Playwright's Chromium
npx playwright test

npm run build:demo                 # static demo (export the snapshot first)
$env:E2E_MODE = "demo"; npx playwright test

# The deployed GitHub Pages demo, without starting any local server
$env:E2E_MODE = "demo"; $env:E2E_BASE_URL = "https://krishna6116.github.io/CareOps-AI/"; npx playwright test
```

On 2026-10-12 all 16 tests passed in the installed Google Chrome, against
both the live API and the demo build. On 2026-10-13 they also passed against
the deployed site at https://krishna6116.github.io/CareOps-AI/.

## Static demo mode

`VITE_DEMO_MODE=snapshot` (set by `npm run build:demo` through `.env.demo`)
makes `src/api.ts` read exported JSON files instead of calling the API. See
[deployment.md](deployment.md#public-demo-github-pages).
