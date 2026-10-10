import { FormEvent, useEffect, useMemo, useState } from 'react';
import {
  BrowserRouter,
  NavLink,
  Navigate,
  Route,
  Routes,
} from 'react-router-dom';

type Row = Record<string, unknown>;
type Series = { name: string; values: Array<number | null> };
type Overview = {
  total_beds: number;
  occupied_beds: number;
  available_beds: number;
  unavailable_beds: number;
  utilization: number;
  predicted_admissions: number;
  predicted_occupancy: number | null;
  icu_total_beds: number;
  icu_occupied_beds: number;
  icu_available_beds: number;
  emergency_total_beds: number;
  emergency_available_beds: number;
  alerts: number;
  optimization_status: string;
  scenario: string;
  latest_forecast_date: string | null;
  forecast_horizon_days: number;
  latest_optimization_run: string | null;
  operational_recommendations: Row[];
};

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const navigation = [
  ['/', 'Executive overview'],
  ['/forecasting', 'Forecasting'],
  ['/beds', 'Bed management'],
  ['/staff', 'Staff scheduling'],
  ['/appointments', 'Appointments'],
  ['/optimization', 'Optimization'],
  ['/alerts', 'Alerts'],
  ['/simulator', 'What-if simulator'],
  ['/assistant', 'AI assistant'],
  ['/audit', 'Audit / run history'],
  ['/about', 'About / how it works'],
] as const;

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(detail?.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

function useApi<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    api<T>(path)
      .then((result) => {
        if (active) setData(result);
      })
      .catch((cause: unknown) => {
        if (active) setError(cause instanceof Error ? cause.message : 'Request failed');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [path]);

  return { data, error, loading };
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <aside className="sidebar">
          <div className="brand">
            <span className="brand-mark">C</span>
            <div><strong>CareOps AI</strong><small>Resource planning</small></div>
          </div>
          <nav aria-label="Main navigation">
            {navigation.map(([path, label]) => (
              <NavLink key={path} to={path} end={path === '/'}>
                {label}
              </NavLink>
            ))}
          </nav>
          <p className="sidebar-note">Synthetic scenario data · Not for clinical use</p>
        </aside>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/forecasting" element={<ForecastingPage />} />
            <Route path="/beds" element={<BedsPage />} />
            <Route path="/staff" element={<StaffPage />} />
            <Route path="/appointments" element={<AppointmentsPage />} />
            <Route path="/optimization" element={<OptimizationPage />} />
            <Route path="/alerts" element={<AlertsPage />} />
            <Route path="/simulator" element={<SimulatorPage />} />
            <Route path="/assistant" element={<AssistantPage />} />
            <Route path="/audit" element={<AuditPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

function OverviewPage() {
  const overview = useApi<Overview>('/dashboard/overview');
  const admissions = useApi<Row[]>('/forecast/admissions');
  const occupancy = useApi<Row[]>('/forecast/occupancy');
  const icu = useApi<Row[]>('/forecast/icu');
  const alerts = useApi<Row[]>('/alerts');
  const error = overview.error ?? admissions.error ?? occupancy.error ?? icu.error ?? alerts.error;
  if (overview.loading || admissions.loading || occupancy.loading || icu.loading || alerts.loading) {
    return <Loading />;
  }
  if (error || !overview.data) return <RequestError message={error ?? 'Dashboard data unavailable'} />;

  const admissionsByDate = sumByDate(departmentRows(admissions.data ?? []), 'predicted_admissions');
  const occupancyForecast = (occupancy.data ?? []).map((row) => number(row.forecast_occupied_beds));
  const icuForecast = (icu.data ?? []).map((row) => number(row.predicted_icu_admissions));
  const summary = overview.data;

  return (
    <>
      <PageHeader title="Executive overview" eyebrow="Portfolio operations" detail={`Scenario run ${summary.scenario}`} />
      <section className="metric-grid">
        <Metric label="Total beds" value={summary.total_beds} />
        <Metric label="Occupied beds" value={summary.occupied_beds} />
        <Metric label="Available beds" value={summary.available_beds} />
        <Metric label="Utilization" value={`${summary.utilization}%`} />
        <Metric label="Forecast admissions" value={format(number(summary.predicted_admissions))} note={`${summary.forecast_horizon_days}-day model horizon`} />
        <Metric label="Forecast occupancy" value={summary.predicted_occupancy === null ? 'Unavailable' : format(summary.predicted_occupancy)} note="Final saved occupancy forecast point" />
        <Metric label="ICU available" value={`${summary.icu_available_beds} / ${summary.icu_total_beds}`} />
        <Metric label="Emergency beds available" value={summary.emergency_available_beds} />
        <Metric label="Audit alerts" value={summary.alerts} />
        <Metric label="Optimization status" value={summary.optimization_status} />
      </section>
      <div className="content-grid">
        <Panel title="Admission demand" detail={`Latest model run · ${summary.latest_forecast_date ?? 'no forecast date'}`}>
          <TrendChart labels={admissionsByDate.labels} series={[{ name: 'Predicted admissions', values: admissionsByDate.values }]} />
        </Panel>
        <Panel title="Occupancy forecast" detail="Predicted occupied beds, not a live census">
          <TrendChart labels={(occupancy.data ?? []).map((row) => String(row.date ?? ''))} series={[{ name: 'Predicted occupancy', values: occupancyForecast }]} />
        </Panel>
        <Panel title="ICU admission flow" detail="The ICU model predicts admissions, not ICU bed census">
          <TrendChart labels={(icu.data ?? []).map((row) => String(row.date ?? ''))} series={[{ name: 'Predicted ICU admissions', values: icuForecast }]} />
        </Panel>
        <Panel title="Current alerts" detail="Generated from saved data-quality audit results">
          {(alerts.data ?? []).length
            ? <DataTable rows={alerts.data ?? []} />
            : <Empty>No audit alerts are present in the current run.</Empty>}
        </Panel>
        <Panel title="Capacity pressure recommendations" detail="Sorted from the saved resource-gap analysis">
          {summary.operational_recommendations.length
            ? <DataTable rows={summary.operational_recommendations} />
            : <Empty>No resource-gap recommendations are available.</Empty>}
        </Panel>
      </div>
    </>
  );
}

function ForecastingPage() {
  const history = useApi<Row[]>('/forecast/history?days=90');
  const admissions = useApi<Row[]>('/forecast/admissions');
  const occupancy = useApi<Row[]>('/forecast/occupancy');
  const icu = useApi<Row[]>('/forecast/icu');
  const los = useApi<Row[]>('/los/predictions?limit=5000');
  const noShow = useApi<Row[]>('/no-show/predictions?limit=100000');
  const metrics = useApi<Record<string, Row>>('/models/metrics');
  const loading = [history, admissions, occupancy, icu, los, noShow, metrics].some((item) => item.loading);
  const error = [history, admissions, occupancy, icu, los, noShow, metrics].find((item) => item.error)?.error;
  if (loading) return <Loading />;
  if (error) return <RequestError message={error} />;

  const historyRows = history.data ?? [];
  const admissionHistory = historyRows.map((row) => number(row.admissions));
  const dailyAdmissionForecast = sumByDate(departmentRows(admissions.data ?? []), 'predicted_admissions');
  const admissionLabels = [...historyRows.map((row) => String(row.date ?? '')), ...dailyAdmissionForecast.labels];
  const admissionActual = [...admissionHistory, ...dailyAdmissionForecast.labels.map(() => null)];
  const admissionPredicted = [...historyRows.map(() => null), ...dailyAdmissionForecast.values];
  const occupancyRows = occupancy.data ?? [];
  const actualOccupancy = historyRows.map((row) => number(row.occupancy));
  const occupancyLabels = [...historyRows.map((row) => String(row.date ?? '')), ...occupancyRows.map((row) => String(row.date ?? ''))];
  const occupancyActual = [...actualOccupancy, ...occupancyRows.map(() => null)];
  const occupancyPredicted = [...historyRows.map(() => null), ...occupancyRows.map((row) => number(row.forecast_occupied_beds))];
  const icuRows = icu.data ?? [];
  const icuActual = historyRows.map((row) => number(row.ICU_admissions));
  const icuLabels = [...historyRows.map((row) => String(row.date ?? '')), ...icuRows.map((row) => String(row.date ?? ''))];
  const icuActualSeries = [...icuActual, ...icuRows.map(() => null)];
  const icuPredicted = [...historyRows.map(() => null), ...icuRows.map((row) => number(row.predicted_icu_admissions))];
  const losRows = los.data ?? [];
  const riskRows = noShow.data ?? [];

  return (
    <>
      <PageHeader title="Forecasting & model outputs" eyebrow="Actual generated results" detail="Historical values, saved forecasts, prediction samples, and saved validation metrics." />
      <div className="content-grid">
        <Panel title="Admission demand" detail="Historical daily admissions and forecast totals aggregated across departments">
          <TrendChart labels={admissionLabels} series={[
            { name: 'Historical admissions', values: admissionActual },
            { name: 'Forecast admissions', values: admissionPredicted },
          ]} />
          <DataTable rows={dailyAdmissionForecast.labels.map((date, index) => ({ date, predicted_admissions: dailyAdmissionForecast.values[index] }))} />
        </Panel>
        <Panel title="Occupancy forecast" detail="Historical occupied census and generated 30-day forecast">
          <TrendChart labels={occupancyLabels} series={[
            { name: 'Historical occupancy', values: occupancyActual },
            { name: 'Forecast occupied beds', values: occupancyPredicted },
          ]} />
          <DataTable rows={occupancyRows} />
        </Panel>
        <Panel title="ICU demand forecast" detail="Admissions per day; the model does not forecast occupied ICU bed census">
          <TrendChart labels={icuLabels} series={[
            { name: 'Historical ICU admissions', values: icuActualSeries },
            { name: 'Predicted ICU admissions', values: icuPredicted },
          ]} />
          <DataTable rows={icuRows} />
        </Panel>
        <Panel title="LOS prediction sample" detail={`${losRows.length} saved predictions shown · predicted length of stay in days`}>
          <DataTable rows={losRows} />
          <Metric label="Mean predicted LOS (sample)" value={format(mean(losRows.map((row) => number(row.predicted_los_days))))} />
        </Panel>
        <Panel title="No-show risk sample" detail={`${riskRows.length} predictions, ordered by highest saved probability`}>
          <div className="metric-grid compact">
            <Metric label="High risk ≥ 0.5" value={riskRows.length ? riskRows.filter((row) => (number(row.predicted_no_show_probability) ?? -1) >= 0.5).length : null} />
            <Metric label="Medium risk 0.2–0.5" value={riskRows.length ? riskRows.filter((row) => { const value = number(row.predicted_no_show_probability); return value !== null && value >= 0.2 && value < 0.5; }).length : null} />
            <Metric label="Lower risk < 0.2" value={riskRows.length ? riskRows.filter((row) => { const value = number(row.predicted_no_show_probability); return value !== null && value < 0.2; }).length : null} />
          </div>
          <DataTable rows={riskRows} />
        </Panel>
        <Panel title="Saved model metrics" detail="Validation metrics from generated model metadata files">
          <JsonBlock value={metrics.data ?? {}} />
        </Panel>
      </div>
    </>
  );
}

function BedsPage() {
  const beds = useApi<Row[]>('/dashboard/beds');
  const availability = useApi<Row[]>('/beds/availability');
  const allocations = useApi<Row[]>('/optimization/runs');
  if (beds.loading || availability.loading || allocations.loading) return <Loading />;
  const error = beds.error ?? availability.error ?? allocations.error;
  if (error) return <RequestError message={error} />;
  const forecasts = availability.data ?? [];
  return (
    <>
      <PageHeader title="Bed management" eyebrow="Inventory & demand" detail="Inventory counts use the generated bed snapshot; forecast and allocation values use saved planning outputs." />
      <Panel title="Department inventory and next forecast date">
        <DataTable rows={beds.data ?? []} />
      </Panel>
      <Panel title="Bed availability forecast" detail={`${forecasts.length} saved department/day rows`}>
        <DataTable rows={forecasts} />
      </Panel>
    </>
  );
}

function StaffPage() {
  const result = useApi<Row>('/dashboard/staff');
  if (result.loading) return <Loading />;
  if (result.error || !result.data) return <RequestError message={result.error ?? 'Staff data unavailable'} />;
  const data = result.data;
  return (
    <>
      <PageHeader title="Staff scheduling" eyebrow="Optimized coverage" detail="The schedule is the saved OR-Tools output for the latest planning run." />
      <div className="metric-grid">
        <Metric label="Source staff" value={number(data.staff_count)} />
        <Metric label="Scheduled shifts" value={number(data.scheduled_shift_count)} />
        <Metric label="Scheduled hours" value={format(number(data.scheduled_hours))} />
        <Metric label="Unfilled positions" value={number(data.unfilled_positions)} />
      </div>
      <Panel title="Coverage by date, department, role and shift"><DataTable rows={asRows(data.coverage)} /></Panel>
      <Panel title="Optimized schedule"><DataTable rows={asRows(data.schedule)} /></Panel>
      <Panel title="Source staff roster"><DataTable rows={asRows(data.staff)} /></Panel>
    </>
  );
}

function AppointmentsPage() {
  const result = useApi<Row>('/dashboard/appointments');
  if (result.loading) return <Loading />;
  if (result.error || !result.data) return <RequestError message={result.error ?? 'Appointment data unavailable'} />;
  const data = result.data;
  const risk = asRows(data.no_show_predictions);
  return (
    <>
      <PageHeader title="Appointments" eyebrow="Provider optimization & attendance risk" detail="The schedule and no-show probabilities are read from actual generated outputs." />
      <div className="metric-grid">
        <Metric label="Source appointments" value={number(data.source_appointment_count)} />
        <Metric label="Optimized appointments" value={number(data.optimized_count)} />
        <Metric label="Unassigned appointments" value={number(data.unassigned_count)} />
        <Metric label="Risk predictions available" value={number(data.no_show_prediction_count)} />
      </div>
      <Panel title="Optimized appointment schedule"><DataTable rows={asRows(data.appointment_schedule)} /></Panel>
      <Panel title="Highest predicted no-show risk" detail="Risk model probabilities are signals, not determinations">
        <DataTable rows={risk} />
      </Panel>
    </>
  );
}

function OptimizationPage() {
  const result = useApi<Row>('/dashboard/optimization');
  if (result.loading) return <Loading />;
  if (result.error || !result.data) return <RequestError message={result.error ?? 'Optimization output unavailable'} />;
  const data = result.data;
  const manifest = asObject(data.manifest);
  return (
    <>
      <PageHeader title="OR-Tools optimization" eyebrow="Saved solver output" detail={`Planning run ${String(manifest.scenario_as_of_date ?? 'unknown')}`} />
      <div className="metric-grid">
        <Metric label="Solver status" value={String(manifest.pipeline_status ?? 'unknown')} />
        <Metric label="Forecast horizon" value={`${number(manifest.forecast_horizon_days)} days`} />
        <Metric label="Assigned appointments" value={number(manifest.optimized_appointment_count)} />
        <Metric label="Run ready" value={String(manifest.next_process_ready ?? false)} />
      </div>
      <Panel title="Baseline vs optimized metrics" detail="Values below come directly from bed_allocation_kpis.csv">
        <DataTable rows={asRows(data.bed_kpis)} />
      </Panel>
      <Panel title="Daily department allocation detail"><DataTable rows={asRows(data.allocations)} /></Panel>
      <Panel title="ICU and emergency capacity"><DataTable rows={asRows(data.emergency_capacity)} /></Panel>
      <Panel title="Appointment optimization KPIs"><DataTable rows={asRows(data.appointment_kpis)} /></Panel>
      <Panel title="Staffing coverage KPIs"><DataTable rows={asRows(data.staff_kpis)} /></Panel>
      <Panel title="Scenario capacity expansion"><DataTable rows={asRows(data.capacity_expansion)} /></Panel>
      <Panel title="Department resource gaps"><DataTable rows={asRows(data.resource_gaps)} /></Panel>
      <Panel title="Forecast backtest results"><DataTable rows={asRows(data.forecast_backtest)} /></Panel>
      <Panel title="Reliability scorecard"><DataTable rows={asRows(data.reliability_scorecard)} /></Panel>
      <Panel title="Reliability evidence by dataset and forecast"><DataTable rows={asRows(data.reliability_details)} /></Panel>
      <Panel title="Run manifest"><JsonBlock value={manifest} /></Panel>
    </>
  );
}

function AlertsPage() {
  const result = useApi<Row[]>('/alerts');
  const [severity, setSeverity] = useState('all');
  if (result.loading) return <Loading />;
  if (result.error) return <RequestError message={result.error} />;
  const alerts = (result.data ?? []).filter((alert) => severity === 'all' || String(alert.severity) === severity);
  return (
    <>
      <PageHeader title="Alerts" eyebrow="Data quality & capacity signals" detail="Items are derived from the saved audit and pipeline manifest; no synthetic alerts are fabricated." />
      <label className="filter-control">Severity
        <select value={severity} onChange={(event) => setSeverity(event.target.value)}>
          <option value="all">All severities</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option>
        </select>
      </label>
      <Panel title={`${alerts.length} matching alert(s)`}>
        {alerts.length ? <DataTable rows={alerts} /> : <Empty>No alerts match this filter.</Empty>}
      </Panel>
    </>
  );
}

type SimulationResult = {
  simulation_id: string;
  scenario_name: string;
  assumptions: Row;
  baseline_metrics: Row;
  simulated_metrics: Row;
  operational_recommendation: string;
  timestamp: string;
  source_run_id: string;
  solver: string;
  scope_note: string;
};

function SimulatorPage() {
  const [admissionsMultiplier, setAdmissionsMultiplier] = useState(1.2);
  const [icuBedDelta, setIcuBedDelta] = useState(-3);
  const [staffDelta, setStaffDelta] = useState(-0.1);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await api<SimulationResult>('/simulation/run', {
        method: 'POST',
        body: JSON.stringify({
          scenario_name: 'dashboard_what_if',
          admissions_multiplier: admissionsMultiplier,
          icu_bed_delta: icuBedDelta,
          staff_availability_delta: staffDelta,
        }),
      });
      setResult(response);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Simulation failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <PageHeader title="What-if simulator" eyebrow="Scenario solver" detail="Runs the existing OR-Tools bed allocator against adjusted inputs. Results remain synthetic planning scenarios." />
      <Panel title="Scenario controls">
        <form className="scenario-form" onSubmit={run}>
          <label>Admissions multiplier
            <input type="number" min="0" max="3" step="0.05" value={admissionsMultiplier} onChange={(event) => setAdmissionsMultiplier(Number(event.target.value))} />
          </label>
          <label>ICU bed capacity change
            <input type="number" min="-20" max="20" step="1" value={icuBedDelta} onChange={(event) => setIcuBedDelta(Number(event.target.value))} />
          </label>
          <label>Staff availability change
            <input type="number" min="-1" max="1" step="0.05" value={staffDelta} onChange={(event) => setStaffDelta(Number(event.target.value))} />
          </label>
          <button className="button" type="submit" disabled={loading}>{loading ? 'Running solver…' : 'Run scenario'}</button>
        </form>
        {error && <RequestError message={error} />}
      </Panel>
      {result && (
        <>
          <Panel title="Baseline vs scenario" detail={`${result.solver} · source run ${result.source_run_id}`}>
            <div className="comparison-grid">
              <div><h3>Saved optimized baseline</h3><JsonBlock value={result.baseline_metrics} /></div>
              <div><h3>Scenario solver result</h3><JsonBlock value={result.simulated_metrics} /></div>
            </div>
            <p className="recommendation"><strong>Solver-based recommendation:</strong> {result.operational_recommendation}</p>
            <p className="muted">{result.scope_note}</p>
            <p className="muted">Saved simulation ID: {result.simulation_id} · {result.timestamp}</p>
          </Panel>
        </>
      )}
    </>
  );
}

function AssistantPage() {
  const [question, setQuestion] = useState('Why is ICU capacity under pressure?');
  const [answer, setAnswer] = useState<Row | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const prompts = [
    'Why is ICU capacity under pressure?',
    'What is the latest admission forecast?',
    'Explain the latest optimization run.',
    'What happens if admissions increase by 20%?',
    'Which department has the highest capacity pressure?',
  ];

  async function ask(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setAnswer(null);
    try {
      setAnswer(await api<Row>('/assistant/ask', { method: 'POST', body: JSON.stringify({ question }) }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Assistant request failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <PageHeader title="CareOps assistant" eyebrow="Grounded explanations" detail="OpenAI explains saved CareOps results and actual OR-Tools what-if runs. Ollama is not required." />
      <Panel title="Ask about this portfolio run">
        <div className="prompt-list">{prompts.map((prompt) => <button key={prompt} type="button" className="prompt" onClick={() => setQuestion(prompt)}>{prompt}</button>)}</div>
        <form className="assistant-form" onSubmit={ask}>
          <label htmlFor="assistant-question">Question</label>
          <textarea id="assistant-question" value={question} maxLength={1000} onChange={(event) => setQuestion(event.target.value)} required />
          <button className="button" type="submit" disabled={loading}>{loading ? 'Asking assistant…' : 'Ask'}</button>
        </form>
        {error && <RequestError message={error} />}
        {answer && <div className="answer"><h3>Answer · run {String(answer.source_run_id)}</h3><p>{String(answer.answer)}</p><p className="muted">Assistant provider: {String(answer.assistant_provider)} · {answer.llm_configured ? 'configured' : 'no API key required'}</p><p className="muted">Sources: {asRows(answer.sources).map((source) => String(source)).join(', ') || JSON.stringify(answer.sources)}</p></div>}
      </Panel>
    </>
  );
}

function AuditPage() {
  const runs = useApi<Row[]>('/optimization/runs');
  const checks = useApi<Row[]>('/audit/runs');
  const assistantAudit = useApi<Row[]>('/audit/assistant');
  const simulations = useApi<Row[]>('/simulation/runs');
  const [selectedRun, setSelectedRun] = useState('');
  const [runDetail, setRunDetail] = useState<Row | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [runDetailLoading, setRunDetailLoading] = useState(false);
  const runId = selectedRun || String(runs.data?.[0]?.scenario_as_of_date ?? '');

  useEffect(() => {
    if (!runId) {
      setRunDetail(null);
      setDetailError(null);
      setRunDetailLoading(false);
      return;
    }
    let active = true;
    setRunDetail(null);
    setDetailError(null);
    setRunDetailLoading(true);
    api<Row>(`/optimization/runs/${encodeURIComponent(runId)}`)
      .then((data) => { if (active) { setRunDetail(data); setDetailError(null); } })
      .catch((cause: unknown) => { if (active) setDetailError(cause instanceof Error ? cause.message : 'Run detail unavailable'); })
      .finally(() => { if (active) setRunDetailLoading(false); });
    return () => { active = false; };
  }, [runId]);

  if (runs.loading || checks.loading || assistantAudit.loading || simulations.loading) return <Loading />;
  const error = runs.error ?? checks.error ?? assistantAudit.error ?? simulations.error;
  if (error) return <RequestError message={error} />;
  return (
    <>
      <PageHeader title="Audit & run history" eyebrow="Reproducibility" detail="Pipeline artifacts, assistant requests, and persisted what-if simulations from the current environment." />
      <Panel title="Portfolio runs">
        {(runs.data ?? []).length ? (
          <>
            <label className="filter-control">Select run
              <select value={runId} onChange={(event) => setSelectedRun(event.target.value)}>
                {(runs.data ?? []).map((run) => <option key={String(run.scenario_as_of_date)} value={String(run.scenario_as_of_date)}>{String(run.scenario_as_of_date)} · {String(run.pipeline_status)}</option>)}
              </select>
            </label>
            {detailError ? <RequestError message={detailError} /> : runDetailLoading ? <Loading /> : runDetail && <JsonBlock value={runDetail} />}
          </>
        ) : <Empty>No completed pipeline runs are recorded.</Empty>}
      </Panel>
      <Panel title="Data quality audit"><DataTable rows={checks.data ?? []} /></Panel>
      <Panel title="AI assistant request audit"><DataTable rows={assistantAudit.data ?? []} /></Panel>
      <Panel title="What-if simulation history"><DataTable rows={simulations.data ?? []} /></Panel>
    </>
  );
}

function AboutPage() {
  return (
    <>
      <PageHeader title="About CareOps AI" eyebrow="How it works" detail="A reproducible, synthetic hospital resource-planning project." />
      <Panel title="Data-to-decision flow">
        <ol className="flow-list">
          <li>Synthetic patient, admission, bed, staff, appointment, operation, and shift records</li>
          <li>Admission, occupancy, ICU-admission, LOS, and no-show model outputs</li>
          <li>Bed availability forecast combines admissions, LOS, occupancy, and discharge estimates</li>
          <li>OR-Tools CP-SAT allocates compatible beds and schedules staff/providers under modeled constraints</li>
          <li>FastAPI serves the saved artifacts; this dashboard visualizes those API responses</li>
          <li>The assistant uses saved CareOps results; OpenAI is the primary optional cloud provider, with a deterministic demo provider available without a key</li>
        </ol>
      </Panel>
      <Panel title="Project capabilities">
        <ul>
          <li><strong>Forecasting:</strong> 30-day admission, occupied-bed, and ICU-admission forecasts, plus length-of-stay and appointment no-show predictions.</li>
          <li><strong>Resource planning:</strong> CP-SAT bed allocation, staff and provider scheduling, emergency capacity analysis, and a scenario-based capacity expansion plan.</li>
          <li><strong>Model evaluation:</strong> chronological holdout backtests against simple baselines, per-model metrics, and a reliability scorecard with dataset-level evidence.</li>
          <li><strong>Operational review:</strong> data-quality audit results, resource-gap analysis, persisted what-if runs, assistant request audit, and a reproducible run manifest.</li>
        </ul>
      </Panel>
      <Panel title="Technology and reproducibility">
        <ul>
          <li><strong>Data and modeling:</strong> Python, pandas, scikit-learn, and seeded synthetic datasets.</li>
          <li><strong>Optimization and API:</strong> Google OR-Tools CP-SAT and FastAPI; optional PostgreSQL persistence for pipeline handoff.</li>
          <li><strong>Dashboard:</strong> React, TypeScript, Vite, and a REST connection to the FastAPI service.</li>
          <li><strong>Re-run:</strong> use <code>.\.venv\Scripts\python.exe run_portfolio.py</code> from the project root; the run manifest and detailed CSV outputs are saved under <code>optimization/resource_planning/outputs/</code>.</li>
        </ul>
      </Panel>
      <Panel title="Project deliverables">
        <ul>
          <li>Forecast charts and prediction samples for the latest saved synthetic scenario.</li>
          <li>Baseline-versus-optimized bed KPIs, per-day allocations, staff coverage, and appointment scheduling results.</li>
          <li>Forecast backtest tables, reliability details, capacity assumptions, and source-data audit results.</li>
          <li>An interactive what-if solver and a grounded assistant that explains saved results and scenario runs.</li>
        </ul>
      </Panel>
      <Panel title="Limitations">
        <ul>
          <li>All bundled source data and generated results are synthetic.</li>
          <li>ICU demand is an admissions-flow forecast; the model does not estimate future ICU census.</li>
          <li>Optimization relies on scenario-only surge capacity and flex staffing where stated in the run manifest.</li>
          <li>OpenAI use requires a server-side API key and sends the question plus aggregate synthetic project outputs to the provider. Ollama is optional, not required.</li>
          <li>The demo mode needs no API key; it is a rules-based explanation provider, not a language model.</li>
          <li>Never use these outputs as clinical decisions or real hospital staffing/placement orders.</li>
        </ul>
      </Panel>
      <Panel title="Service links"><p><a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">FastAPI documentation</a> · <a href={`${API_URL}/health`} target="_blank" rel="noreferrer">Backend health</a></p></Panel>
    </>
  );
}

function PageHeader({ title, eyebrow, detail }: { title: string; eyebrow: string; detail?: string }) {
  return <header className="page-header"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1>{detail && <p className="page-detail">{detail}</p>}</div><span className="synthetic-badge">Synthetic data</span></header>;
}

function Panel({ title, detail, children }: { title: string; detail?: string; children: React.ReactNode }) {
  return <section className="panel"><div className="panel-heading"><h2>{title}</h2>{detail && <p>{detail}</p>}</div>{children}</section>;
}

function Metric({ label, value, note }: { label: string; value: string | number | null; note?: string }) {
  return <article className="metric-card"><span>{label}</span><strong>{value ?? 'Unavailable'}</strong>{note && <small>{note}</small>}</article>;
}

function TrendChart({ labels, series }: { labels: string[]; series: Series[] }) {
  const width = 760;
  const height = 230;
  const pad = 28;
  const allValues = series.flatMap((line) => line.values.filter((value): value is number => value !== null && Number.isFinite(value)));
  if (!labels.length || !allValues.length) return <Empty>No chart data is available.</Empty>;
  const min = Math.min(0, ...allValues);
  const max = Math.max(...allValues, min + 1);
  const colors = ['#65a5ff', '#47d7ae', '#ffc66e'];
  const x = (index: number) => pad + (labels.length <= 1 ? 0 : index * (width - pad * 2) / (labels.length - 1));
  const y = (value: number) => height - pad - (value - min) * (height - pad * 2) / (max - min || 1);
  const paths = series.map((line) => {
    let started = false;
    return line.values.map((value, index) => {
      if (value === null || !Number.isFinite(value)) { started = false; return ''; }
      const point = `${started ? 'L' : 'M'}${x(index).toFixed(1)},${y(value).toFixed(1)}`;
      started = true;
      return point;
    }).join(' ');
  });
  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={series.map((line) => line.name).join(', ')}>
        {[0, 1, 2, 3].map((tick) => {
          const value = min + (max - min) * tick / 3;
          const lineY = y(value);
          return <g key={tick}><line x1={pad} x2={width - pad} y1={lineY} y2={lineY} className="chart-grid" /><text x="2" y={lineY + 4} className="chart-label">{format(value)}</text></g>;
        })}
        {paths.map((path, index) => <path key={series[index].name} d={path} fill="none" stroke={colors[index % colors.length]} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />)}
      </svg>
      <div className="chart-legend">{series.map((line, index) => <span key={line.name}><i style={{ backgroundColor: colors[index % colors.length] }} />{line.name}</span>)}</div>
      <div className="chart-range"><span>{labels[0]}</span><span>{labels[labels.length - 1]}</span></div>
    </div>
  );
}

function DataTable({ rows }: { rows: Row[] }) {
  const [query, setQuery] = useState('');
  const [pageSize, setPageSize] = useState(25);
  const [page, setPage] = useState(0);
  const columns = useMemo(() => Array.from(new Set(rows.flatMap((row) => Object.keys(row)))), [rows]);
  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase();
    if (!normalizedQuery) return rows;
    return rows.filter((row) => Object.values(row).some((value) => display(value).toLocaleLowerCase().includes(normalizedQuery)));
  }, [query, rows]);
  const pageCount = Math.ceil(filteredRows.length / pageSize);
  const currentPage = Math.min(page, Math.max(0, pageCount - 1));
  const visibleRows = filteredRows.slice(currentPage * pageSize, (currentPage + 1) * pageSize);

  useEffect(() => setPage(0), [query, pageSize, rows]);

  if (!rows.length || !columns.length) return <Empty>No rows are available.</Empty>;
  return (
    <>
      <div className="table-toolbar">
        <label>Search records<input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search any column" /></label>
        <label>Rows per page<select value={pageSize} onChange={(event) => setPageSize(Number(event.target.value))}><option value={10}>10</option><option value={25}>25</option><option value={50}>50</option><option value={100}>100</option></select></label>
        <span className="table-count">{filteredRows.length ? `Showing ${currentPage * pageSize + 1}–${Math.min((currentPage + 1) * pageSize, filteredRows.length)} of ${filteredRows.length} records` : 'No matching records'}</span>
      </div>
      {visibleRows.length ? (
        <>
          <div className="table-wrap"><table><thead><tr>{columns.map((column) => <th key={column}>{column.replace(/_/g, ' ')}</th>)}</tr></thead><tbody>
            {visibleRows.map((row, index) => <tr key={`${String(row.id ?? row.admission_id ?? row.appointment_id ?? row.date ?? 'row')}-${currentPage * pageSize + index}`}>{columns.map((column) => <td key={column}>{display(row[column])}</td>)}</tr>)}
          </tbody></table></div>
          <div className="table-pagination">
            <button type="button" className="button button-secondary" onClick={() => setPage(currentPage - 1)} disabled={currentPage === 0}>Previous</button>
            <span>Page {currentPage + 1} of {pageCount}</span>
            <button type="button" className="button button-secondary" onClick={() => setPage(currentPage + 1)} disabled={currentPage + 1 >= pageCount}>Next</button>
          </div>
        </>
      ) : <Empty>No records match this search.</Empty>}
    </>
  );
}

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>;
}

function Loading() {
  return <div className="state-card" role="status">Loading real CareOps data…</div>;
}

function RequestError({ message }: { message: string }) {
  return <div className="state-card error-state" role="alert"><strong>Could not load this data</strong><p>{message}</p><small>Check that the FastAPI backend is running and reachable at {API_URL}.</small><p><button className="button" type="button" onClick={() => window.location.reload()}>Retry</button></p></div>;
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="empty-state">{children}</p>;
}

// Admission forecasts include an "All Departments" total row per day; sum departments only.
function departmentRows(rows: Row[]): Row[] {
  return rows.filter((row) => String(row.department ?? '').toLowerCase() !== 'all departments');
}

function sumByDate(rows: Row[], field: string) {
  const totals = new Map<string, number>();
  rows.forEach((row) => {
    const date = String(row.date ?? '');
    const value = number(row[field]);
    if (!date || value === null) return;
    totals.set(date, (totals.get(date) ?? 0) + value);
  });
  const labels = [...totals.keys()].sort();
  return { labels, values: labels.map((label) => totals.get(label) ?? 0) };
}

function asRows(value: unknown): Row[] {
  return Array.isArray(value) ? value.filter((row): row is Row => typeof row === 'object' && row !== null) : [];
}

function asObject(value: unknown): Row {
  return typeof value === 'object' && value !== null ? value as Row : {};
}

function number(value: unknown): number | null {
  if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function mean(values: Array<number | null>): number | null {
  const valid = values.filter((value): value is number => value !== null && Number.isFinite(value));
  return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : null;
}

function format(value: number | null): string {
  return value === null || !Number.isFinite(value) ? 'Unavailable' : new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value);
}

function display(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}
