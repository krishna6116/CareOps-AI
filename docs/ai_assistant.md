# CareOps assistant

`POST /assistant/ask` provides explanations using
the latest saved manifest, admission/ICU forecasts, no-show model metrics, bed
inventory, audit alerts, resource gaps, OR-Tools KPI results, and persisted
what-if simulations. For scenario questions, it only explains a matching saved
simulation; otherwise it directs the user to run one. The React assistant page
sends user questions to this endpoint and displays the source run and data
sources.

OpenAI is the primary hosted provider. Set its API key only in the backend
environment (or an ignored local `.env` file), then configure the provider and
model before starting the API:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:CAREOPS_LLM_PROVIDER = "openai"
$env:CAREOPS_LLM_MODEL = "gpt-4o-mini"
```

The API key is never returned by the API or sent to the React application.
The OpenAI integration sends the question and allowlisted aggregate project
context only; patient-level prediction rows are excluded. Unsupported numeric
claims are rejected unless the value appears in the question or saved context.
When a key is not configured, OpenAI is unavailable and the API reports that
explicitly. Set `CAREOPS_LLM_PROVIDER=demo` to intentionally use the
deterministic rules-based provider without a key. Local model providers are
optional and not required by the application; the default supported runtime is
OpenAI-backed reasoning with synthetic project data.

Neither provider creates forecasts or metrics. Both are instructed to use the
saved outputs and say when requested details are absent. Provider failures,
timeouts, invalid credentials, and malformed responses are reported rather than
silently falling back. For ICU questions the assistant distinguishes predicted
admissions/day from occupied ICU bed census; the current ICU model does not
forecast that census. Assistant text is explanatory only and must not be used
for clinical decisions.

`GET /audit/assistant` returns recent assistant audit entries: question,
provider/model, response status, source names, elapsed milliseconds, source run,
and sanitized error text where applicable. API keys and request authorization
headers are never logged. Questions may contain sensitive information; this
synthetic demo has no user authentication, so do not expose it publicly or use
real patient information.
