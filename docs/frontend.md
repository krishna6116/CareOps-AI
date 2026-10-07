# Frontend dashboard

The React + TypeScript dashboard uses React Router and consumes the FastAPI
service at `VITE_API_URL` (default `http://localhost:8000`). It has separate
routes for:

- Executive overview
- Forecasting and model outputs
- Bed management
- Staff scheduling
- Appointments and no-show risk
- OR-Tools optimization
- Alerts
- What-if simulator
- CareOps assistant
- Audit and run history
- About and limitations

The overview, charts, tables, and metrics are populated from saved generated
outputs exposed by the API. The ICU view labels its forecast as admissions/day;
it does not present those forecasts as occupied ICU beds. Empty results remain
empty, and failed API calls are shown as errors rather than replaced by demo
values.

## Local development

Use Node.js 20.19+ (or 22.12+) and run from the repository root:

```powershell
Set-Location frontend
npm ci
$env:VITE_API_URL = "http://localhost:8000"
npm run dev
```

The dashboard is served at `http://localhost:5173`. Create a production build
with `npm run build`.
