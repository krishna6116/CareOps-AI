import { FormEvent, useState } from 'react';
import {
  BrowserRouter,
  HashRouter,
  NavLink,
  Navigate,
  Route,
  Routes,
} from 'react-router-dom';

import { API_URL, DEMO_MODE, api, asObject, asRows, type Row, useApi } from './api';
import demoConfig from './demo-config.json';
import {
  ComparisonTable,
  DataTable,
  Empty,
  JsonBlock,
  Loading,
  Metric,
  PageHeader,
  Panel,
  RequestError,
  TrendChart,
  type ComparisonRow,
} from './components';
import { formatCount, formatDecimal, formatPercent, label, toNumber } from './format';

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

// GitHub Pages cannot rewrite deep links to index.html, so the static demo
// uses hash routes.
const Router = DEMO_MODE ? HashRouter : BrowserRouter;

function DemoBanner() {
  const snapshot = useApi<Row>(DEMO_MODE ? '/meta' : null);
  if (!DEMO_MODE) return null;
  return (
    <div className="demo-banner" role="note">
      <strong>Static demo.</strong> A snapshot of pipeline run {String(snapshot.data?.scenario_as_of_date ?? '')} on fully
      synthetic data, exported from the real API. The assistant answers the example questions and the simulator
      runs the preset scenarios from precomputed results.{' '}
      <a href="https://github.com/krishna6116/CareOps-AI#how-to-run" target="_blank" rel="noreferrer">Run it locally</a> for the live API.
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <div className="app-layout">
        <aside className="sidebar">
          <div className="brand">
            <span className="brand-mark">C</span>
            <div><strong>CareOps AI</strong><small>Resource planning</small></div>
          </div>
          <nav aria-label="Main navigation">
            {navigation.map(([path, name]) => (
              <NavLink key={path} to={path} end={path === '/'}>
                {name}
              </NavLink>
            ))}
          </nav>
          <p className="sidebar-note">Synthetic scenario data · Not for clinical use</p>
        </aside>
        <main className="main-content">
          <DemoBanner />
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
    </Router>
  );
}

export function OverviewPage() {
  const overview = useApi<Overview>('/dashboard/overview');
  const admissions = useApi<Row[]>('/forecast/admissions');
  const occupancy = useApi<Row[]>('/forecast/occupancy');
  const icu = useApi<Row[]>('/forecast/icu');
  const alerts = useApi<Row[]>('/alerts');
  const assignment = useApi<Row>('/optimization/bed-assignments');
  const requests = [overview, admissions, occupancy, icu, alerts];
  if (requests.some((item) => item.loading)) return <Loading />;
  const error = requests.find((item) => item.error)?.error;
  if (error || !overview.data) return <RequestError message={error ?? 'Dashboard data unavailable'} />;

  const admissionsByDate = sumByDate(departmentRows(admissions.data ?? []), 'predicted_admissions');
  const occupancyForecast = (occupancy.data ?? []).map((row) => toNumber(row.forecast_occupied_beds));
  const icuForecast = (icu.data ?? []).map((row) => toNumber(row.predicted_icu_admissions));
  const summary = overview.data;
  const optimized = asRows(assignment.data?.kpis).find((row) => row.policy === 'optimized');

  return (
    <>
      <PageHeader title="Executive overview" eyebrow="Portfolio operations" detail={`Scenario cutoff ${summary.scenario}. Snapshot figures describe the synthetic hospital on that date; forecast figures cover the next ${summary.forecast_horizon_days} days.`} />
      <section className="metric-grid">
        <Metric label="Total beds" value={formatCount(summary.total_beds)} note="Snapshot" />
        <Metric label="Occupied beds" value={formatCount(summary.occupied_beds)} note="Snapshot" />
        <Metric label="Available beds" value={formatCount(summary.available_beds)} note="Snapshot · excludes maintenance" />
        <Metric label="Occupancy" value={formatPercent(summary.total_beds ? summary.occupied_beds / summary.total_beds : null)} note="Snapshot · occupied / total beds" />
        <Metric label="ICU available" value={`${formatCount(summary.icu_available_beds)} / ${formatCount(summary.icu_total_beds)}`} note="Snapshot" />
        <Metric label="Forecast admissions" value={formatCount(summary.predicted_admissions)} note={`Forecast · next ${summary.forecast_horizon_days} days, rounded expected values`} />
        <Metric label="Forecast occupied beds" value={summary.predicted_occupancy === null ? 'Unavailable' : formatCount(summary.predicted_occupancy)} note={`Forecast · ${summary.latest_forecast_date ?? 'final day'}`} />
        <Metric label="Bed requests placed" value={optimized ? `${formatCount(optimized.assigned)} / ${formatCount(optimized.requests)}` : 'Unavailable'} note="Replay · last 28 days, CP-SAT" />
        <Metric label="Open alerts" value={formatCount(summary.alerts)} note="Data-quality audit and saved alerts" />
        <Metric label="Pipeline status" value={summary.optimization_status === 'ready_for_next_process' ? 'Ready' : 'Review required'} note={summary.optimization_status.replace(/_/g, ' ')} />
      </section>
      <div className="content-grid">
        <Panel title="Admission demand" detail="Forecast · all departments, admissions per day">
          <TrendChart labels={admissionsByDate.labels} series={[{ name: 'Forecast admissions', values: admissionsByDate.values }]} />
        </Panel>
        <Panel title="Occupancy forecast" detail="Forecast · occupied beds, not a live census">
          <TrendChart labels={(occupancy.data ?? []).map((row) => String(row.date ?? ''))} series={[{ name: 'Forecast occupied beds', values: occupancyForecast }]} />
        </Panel>
        <Panel title="ICU admission flow" detail="Forecast · ICU admissions per day, not ICU bed census">
          <TrendChart labels={(icu.data ?? []).map((row) => String(row.date ?? ''))} series={[{ name: 'Forecast ICU admissions', values: icuForecast }]} />
        </Panel>
        <Panel title="Recommendations" detail="From the saved resource-gap analysis">
          {summary.operational_recommendations.length
            ? <ul>{summary.operational_recommendations.map((item, index) => <li key={index}><strong>[{String(item.priority)}] {String(item.title)}:</strong> {String(item.detail)}</li>)}</ul>
            : <Empty>No resource-gap recommendations are available.</Empty>}
        </Panel>
        <Panel title="Current alerts" detail="Generated by audit rules and saved alerts" wide>
          {(alerts.data ?? []).length
            ? <DataTable rows={alerts.data ?? []} />
            : <Empty>No alerts are open in the current run.</Empty>}
        </Panel>
      </div>
    </>
  );
}

const HORIZONS = [7, 14, 30];

export function ForecastingPage() {
  const [horizon, setHorizon] = useState(30);
  const history = useApi<Row[]>('/forecast/history?days=90');
  const admissions = useApi<Row[]>('/forecast/admissions');
  const occupancy = useApi<Row[]>('/forecast/occupancy');
  const icu = useApi<Row[]>('/forecast/icu');
  const los = useApi<Row[]>('/los/predictions?limit=500');
  const noShow = useApi<Row[]>('/no-show/predictions?limit=500');
  const noShowSummary = useApi<Row>('/no-show/summary');
  const evaluation = useApi<Row>('/models/evaluation');
  const all = [history, admissions, occupancy, icu, los, noShow, noShowSummary, evaluation];
  if (all.some((item) => item.loading)) return <Loading />;
  const error = all.find((item) => item.error)?.error;
  if (error) return <RequestError message={error} />;

  const historyRows = history.data ?? [];
  const historyLabels = historyRows.map((row) => String(row.date ?? ''));
  const pad = (values: Array<number | null>, count: number) => [...values, ...Array<null>(count).fill(null)];
  const lead = (count: number, values: Array<number | null>) => [...Array<null>(count).fill(null), ...values];

  const admissionForecast = sumByDate(departmentRows(admissions.data ?? []), 'predicted_admissions');
  const admissionLabels = admissionForecast.labels.slice(0, horizon);
  const admissionValues = admissionForecast.values.slice(0, horizon);
  const occupancyRows = (occupancy.data ?? []).slice(0, horizon);
  const icuRows = (icu.data ?? []).slice(0, horizon);
  const losRows = los.data ?? [];
  const riskRows = noShow.data ?? [];
  const bands = asObject(noShowSummary.data?.risk_bands);

  return (
    <>
      <PageHeader title="Forecasting & model evaluation" eyebrow="Saved model outputs" detail="Charts show 90 days of synthetic history followed by the saved forecast. Model evaluation metrics are kept separate from operational figures." />
      <label className="filter-control">Forecast horizon
        <select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}>
          {HORIZONS.map((days) => <option key={days} value={days}>{days} days</option>)}
        </select>
      </label>
      <ModelEvaluationPanel evaluation={evaluation.data ?? {}} />
      <div className="content-grid">
        <Panel title="Admission demand" detail="History vs forecast · all departments">
          <TrendChart labels={[...historyLabels, ...admissionLabels]} series={[
            { name: 'Historical admissions', values: pad(historyRows.map((row) => toNumber(row.admissions)), admissionLabels.length) },
            { name: 'Forecast admissions', values: lead(historyRows.length, admissionValues) },
          ]} />
        </Panel>
        <Panel title="Occupied beds" detail="History vs forecast with 80% interval">
          <TrendChart labels={[...historyLabels, ...occupancyRows.map((row) => String(row.date ?? ''))]} series={[
            { name: 'Historical occupied beds', values: pad(historyRows.map((row) => toNumber(row.occupied_beds)), occupancyRows.length) },
            { name: 'Forecast occupied beds', values: lead(historyRows.length, occupancyRows.map((row) => toNumber(row.forecast_occupied_beds))) },
            { name: 'Upper 80% bound', values: lead(historyRows.length, occupancyRows.map((row) => toNumber(row.occupied_beds_upper_80))) },
          ]} />
        </Panel>
        <Panel title="ICU admissions" detail="History vs forecast · admissions per day, not ICU census">
          <TrendChart labels={[...historyLabels, ...icuRows.map((row) => String(row.date ?? ''))]} series={[
            { name: 'Historical ICU admissions', values: pad(historyRows.map((row) => toNumber(row.ICU_admissions)), icuRows.length) },
            { name: 'Forecast ICU admissions', values: lead(historyRows.length, icuRows.map((row) => toNumber(row.predicted_icu_admissions))) },
          ]} />
        </Panel>
        <Panel title="Forecast table" detail={`Next ${horizon} days · whole-number counts with 80% / 95% intervals`}>
          <DataTable rows={occupancyRows} />
        </Panel>
        <Panel title="LOS predictions" detail={`${formatCount(losRows.length)} saved predictions · days, predicted at admission`}>
          <Metric label="Mean predicted LOS" value={`${formatDecimal(mean(losRows.map((row) => toNumber(row.predicted_los_days))), 1)} days`} />
          <DataTable rows={losRows} />
        </Panel>
        <Panel title="No-show risk" detail={`${formatCount(noShowSummary.data?.prediction_count)} holdout predictions · observed no-show rate ${formatPercent(noShowSummary.data?.observed_no_show_rate)} · probabilities are planning signals, not determinations`}>
          <div className="metric-grid compact">
            <Metric label="Probability ≥ 0.5" value={formatCount(bands['0_5_and_above'])} />
            <Metric label="0.2 – 0.5" value={formatCount(bands['0_2_to_0_5'])} />
            <Metric label="Below 0.2" value={formatCount(bands.below_0_2)} />
          </div>
          <h3>Highest predicted risk (top {formatCount(riskRows.length)})</h3>
          <DataTable rows={riskRows} />
        </Panel>
      </div>
    </>
  );
}

function ModelEvaluationPanel({ evaluation }: { evaluation: Row }) {
  const summary = asRows(evaluation.summary);
  const coverage = asRows(evaluation.interval_coverage);
  const risk = asRows(evaluation.occupancy_risk_thresholds);
  if (!summary.length) return <Panel title="Model evaluation"><Empty>No evaluation outputs are available.</Empty></Panel>;
  return (
    <Panel title="Model evaluation" detail={`Backtests from run ${String(evaluation.scenario_as_of_date ?? 'unknown')} · lower is better for MAE and Brier · details in ${String(evaluation.report ?? 'docs/MODEL_METRICS.md')}`}>
      <div className="table-wrap">
        <table>
          <thead><tr><th scope="col">Target</th><th scope="col">Evaluation</th><th scope="col">Metric</th><th scope="col">Published model</th><th scope="col">Published</th><th scope="col">Best baseline</th><th scope="col">Best learned</th><th scope="col">Learned beats baseline</th></tr></thead>
          <tbody>{summary.map((row) => (
            <tr key={String(row.target)}>
              <th scope="row">{String(row.target)}</th><td>{String(row.evaluation)}</td><td>{String(row.metric)}</td>
              <td>{String(row.published_model)}</td><td>{String(row.published)}</td><td>{String(row.best_baseline)}</td>
              <td>{String(row.best_learned)}</td><td className={row.learned_beats_baseline ? 'delta-better' : 'delta-worse'}>{row.learned_beats_baseline ? 'yes' : 'no — baseline published'}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <div className="content-grid evaluation-grid">
        <div>
          <h3>Interval coverage (published intervals)</h3>
          <DataTable rows={coverage.map((row) => ({
            target: row.target, nominal: formatPercent(row.level, 0), horizon_days: row.bucket,
            empirical_coverage: formatPercent(row.empirical_coverage), origins: row.evaluation_origins,
          }))} />
        </div>
        <div>
          <h3>Occupancy alert thresholds</h3>
          <DataTable rows={risk.map((row) => ({
            threshold: label(String(row.threshold_name)), beds: formatCount(row.threshold),
            actual_days: row.actual_events, predicted_days: row.predicted_events,
            recall: row.recall === null ? 'untested' : formatPercent(row.recall),
          }))} />
          <p className="muted">A threshold with 0 actual days was never reached in the backtest, so the forecast's alerting there is untested.</p>
        </div>
      </div>
    </Panel>
  );
}

export function BedsPage() {
  const beds = useApi<Row[]>('/dashboard/beds');
  const availability = useApi<Row[]>('/beds/availability');
  if (beds.loading || availability.loading) return <Loading />;
  const error = beds.error ?? availability.error;
  if (error) return <RequestError message={error} />;
  const forecasts = availability.data ?? [];
  return (
    <>
      <PageHeader title="Bed management" eyebrow="Inventory, availability & allocation" detail="Inventory is the synthetic snapshot on the cutoff date. Availability is forecast from admissions, predicted LOS and the current census. Allocation replays actual synthetic arrivals." />
      <Panel title="Department inventory and next forecast day">
        <DataTable rows={beds.data ?? []} />
      </Panel>
      <BedAssignmentSection />
      <Panel title="Bed availability forecast" detail={`${formatCount(forecasts.length)} department/day rows · available = operational capacity − projected occupied, never below zero`}>
        <DataTable rows={forecasts} />
      </Panel>
    </>
  );
}

const REASON_LABELS: Record<string, string> = {
  no_compatible_bed_in_inventory: 'No compatible bed in inventory',
  all_compatible_beds_occupied: 'All compatible beds occupied',
  staffing_limit: 'Staffing limit',
  protected_reserve: 'Protected ICU/emergency reserve',
  taken_by_other_requests: 'Taken by other requests',
};

export function BedAssignmentSection() {
  const [policy, setPolicy] = useState<'optimized' | 'baseline'>('optimized');
  const summary = useApi<Row>('/optimization/bed-assignments');
  const unassigned = useApi<Row>(`/optimization/bed-assignments/requests?status=UNASSIGNED&policy=${policy}&limit=500`);
  if (summary.loading) return <Panel title="Request-level bed allocation"><Loading /></Panel>;
  if (summary.error || !summary.data) return <Panel title="Request-level bed allocation"><RequestError message={summary.error ?? 'Allocation results unavailable'} /></Panel>;
  const data = summary.data;
  const kpis = Object.fromEntries(asRows(data.kpis).map((row) => [String(row.policy), row]));
  const base = kpis.baseline ?? {};
  const opt = kpis.optimized ?? {};
  const span = Array.isArray(data.window) ? data.window.map(String) : [];
  const statuses = asObject(data.solver_statuses);
  const rows: ComparisonRow[] = [
    { metric: 'Requests', baseline: base.requests, scenario: opt.requests },
    { metric: 'Assigned', baseline: base.assigned, scenario: opt.assigned, lowerIsBetter: false },
    { metric: 'Unassigned', baseline: base.unassigned, scenario: opt.unassigned, lowerIsBetter: true },
    { metric: 'Assignment rate', baseline: base.assignment_rate, scenario: opt.assignment_rate, format: (value) => formatPercent(value), lowerIsBetter: false },
    { metric: 'High-severity assignment rate', baseline: base.high_severity_assignment_rate, scenario: opt.high_severity_assignment_rate, format: (value) => formatPercent(value), lowerIsBetter: false },
    { metric: 'Off-service placements', baseline: base.off_service_placements, scenario: opt.off_service_placements },
    { metric: 'Compatibility violations', baseline: base.compatibility_violations, scenario: opt.compatibility_violations, lowerIsBetter: true },
    { metric: 'Placements above rostered staffing', baseline: base.staffing_exceedances, scenario: opt.staffing_exceedances, lowerIsBetter: true },
  ];
  const reasons = Object.keys(REASON_LABELS).map((reason) => ({
    reason: REASON_LABELS[reason],
    baseline: base[`unassigned_${reason}`] ?? 0,
    optimized: opt[`unassigned_${reason}`] ?? 0,
    definition: String(asObject(data.reason_definitions)[reason] ?? ''),
  }));
  return (
    <Panel title="Request-level bed allocation" detail={`Replay ${span[0] ?? ''} to ${span[1] ?? ''} · solver ${Object.entries(statuses).map(([status, count]) => `${status} on ${formatCount(count)} days`).join(', ')}`}>
      <p className="muted">Each day's actual synthetic arrivals are placed against the census rebuilt from the admission history. Baseline: {String(data.baseline_policy)} Both policies use the same compatibility, reserve and staffing rules.</p>
      <ComparisonTable rows={rows} baselineLabel="Baseline" scenarioLabel="CP-SAT optimized" />
      <h3>Why requests were not placed</h3>
      <DataTable rows={reasons} />
      <h3>Unassigned requests</h3>
      <label className="filter-control">Policy
        <select value={policy} onChange={(event) => setPolicy(event.target.value as 'optimized' | 'baseline')}>
          <option value="optimized">CP-SAT optimized</option><option value="baseline">Baseline</option>
        </select>
      </label>
      {unassigned.loading ? <Loading /> : unassigned.error ? <RequestError message={unassigned.error} /> : (
        asRows(unassigned.data?.rows).length
          ? <DataTable rows={asRows(unassigned.data?.rows).map((row) => ({
            date: row.planning_date, admission: row.admission_id, department: row.department, severity: row.severity,
            icu: row.icu_required, isolation: row.isolation_required, compatible_free_beds: row.compatible_free_beds,
            reason: REASON_LABELS[String(row[`${policy}_reason`])] ?? row[`${policy}_reason`],
          }))} />
          : <Empty>Every request was placed under this policy.</Empty>
      )}
      <details>
        <summary>Sensitivity to weights, reserve and staffing</summary>
        <DataTable rows={asRows(data.sensitivity)} />
      </details>
      <p className="muted">Synthetic replay: this shows the allocator respects its rules. It is not evidence of improvement in a real hospital. See docs/BED_ALLOCATION_DIAGNOSIS.md.</p>
    </Panel>
  );
}

export function StaffPage() {
  const result = useApi<Row>('/dashboard/staff');
  if (result.loading) return <Loading />;
  if (result.error || !result.data) return <RequestError message={result.error ?? 'Staff data unavailable'} />;
  const data = result.data;
  const policies = Object.fromEntries(asRows(data.policy_comparison).map((row) => [String(row.policy), row]));
  const base = policies.first_fit_baseline ?? {};
  const opt = policies.optimized ?? {};
  const rows: ComparisonRow[] = [
    { metric: 'Required positions', baseline: base.required_positions, scenario: opt.required_positions },
    { metric: 'Unfilled positions', baseline: base.unfilled_positions, scenario: opt.unfilled_positions, lowerIsBetter: true },
    { metric: 'Base-staff shifts', baseline: base.base_staff_shifts, scenario: opt.base_staff_shifts },
    { metric: 'Synthetic surge shifts', baseline: base.surge_staff_shifts, scenario: opt.surge_staff_shifts, lowerIsBetter: true },
    { metric: 'Overtime shifts', baseline: base.overtime_shifts, scenario: opt.overtime_shifts, lowerIsBetter: true },
    { metric: 'Overtime hours', baseline: base.overtime_hours, scenario: opt.overtime_hours, lowerIsBetter: true },
    { metric: 'Workload spread (std of hours, base staff)', baseline: base.base_hours_std, scenario: opt.base_hours_std, format: (value) => `${formatDecimal(value, 1)} h`, lowerIsBetter: true },
    { metric: 'Shift-preference match', baseline: base.shift_preference_match_rate, scenario: opt.shift_preference_match_rate, format: (value) => formatPercent(value), lowerIsBetter: false },
  ];
  return (
    <>
      <PageHeader title="Staff scheduling" eyebrow="Optimized coverage" detail="Required staff per shift comes from projected census (1 nurse : 4 beds) and historical role coverage. Both policies use the same requirements, availability, rest and hour rules." />
      <div className="metric-grid">
        <Metric label="Source staff" value={formatCount(data.staff_count)} />
        <Metric label="Scheduled shifts" value={formatCount(data.scheduled_shift_count)} />
        <Metric label="Scheduled hours" value={formatCount(data.scheduled_hours)} />
        <Metric label="Unfilled positions" value={formatCount(data.unfilled_positions)} />
      </div>
      {asRows(data.policy_comparison).length > 0 && (
        <Panel title="First-fit baseline vs CP-SAT schedule" detail="30-day horizon · simulated on synthetic data">
          <ComparisonTable rows={rows} baselineLabel="First-fit baseline" scenarioLabel="CP-SAT optimized" />
          <p className="muted">The baseline fills each position with the first eligible person in roster order, base staff before surge, and never schedules overtime. The optimizer may give base staff up to one extra shift a week; a full overtime shift costs more than a surge shift, so overtime is mostly partial. Surge staff are a scenario assumption, not hires.</p>
        </Panel>
      )}
      <Panel title="Coverage by date, department, role and shift"><DataTable rows={asRows(data.coverage)} /></Panel>
      <Panel title="Optimized schedule"><DataTable rows={asRows(data.schedule)} /></Panel>
      <Panel title="Source staff roster"><DataTable rows={asRows(data.staff)} /></Panel>
    </>
  );
}

export function AppointmentsPage() {
  const result = useApi<Row>('/dashboard/appointments');
  if (result.loading) return <Loading />;
  if (result.error || !result.data) return <RequestError message={result.error ?? 'Appointment data unavailable'} />;
  const data = result.data;
  const kpi = asRows(data.kpis)[0] ?? {};
  const rows: ComparisonRow[] = [
    { metric: 'Doctor double-bookings', baseline: kpi.baseline_doctor_time_conflicts, scenario: kpi.optimized_doctor_time_conflicts, lowerIsBetter: true },
    { metric: 'Bookings with an off-shift doctor', baseline: kpi.baseline_off_shift_doctor_bookings, scenario: kpi.optimized_off_shift_doctor_bookings, lowerIsBetter: true },
    { metric: 'Bookings with an unavailable doctor', baseline: kpi.baseline_unavailable_doctor_bookings, scenario: kpi.optimized_unavailable_doctor_bookings, lowerIsBetter: true },
  ];
  return (
    <>
      <PageHeader title="Appointments" eyebrow="Provider scheduling & attendance risk" detail={`Last ${formatCount(kpi.planning_days)} clinic days (${String(kpi.planning_date_start ?? '')} to ${String(kpi.planning_date_end ?? '')}) re-slotted with one CP-SAT model per day.`} />
      <div className="metric-grid">
        <Metric label="Appointments scheduled" value={`${formatCount(kpi.appointments_scheduled)} / ${formatCount(kpi.appointments_requested)}`} />
        <Metric label="Doctor changes" value={formatCount(kpi.doctor_changes)} note="Needed to move off-shift or unavailable bookings" />
        <Metric label="Slot utilization" value={formatPercent(kpi.slot_utilization)} note="Booked ÷ rostered provider minutes" />
        <Metric label="Utilization after expected no-shows" value={formatPercent(kpi.expected_attended_utilization)} />
        <Metric label="Reminder outreach list" value={formatCount(kpi.reminder_outreach_count)} note={`Risk ≥ ${String(kpi.no_show_threshold ?? '')} (validation threshold)`} />
        <Metric label="Expected idle minutes" value={formatCount(kpi.expected_no_show_minutes)} note="Σ risk × duration, for overbooking review" />
      </div>
      <Panel title="Original bookings vs optimized schedule">
        <ComparisonTable rows={rows} baselineLabel="Original bookings" scenarioLabel="Optimized" />
        <p className="muted">No-show risk never changes who is booked, their priority or their slot. It only flags reminder outreach and estimates idle minutes so staff can review overbooking.</p>
      </Panel>
      <Panel title="Optimized appointment schedule"><DataTable rows={asRows(data.appointment_schedule)} /></Panel>
      <Panel title="Highest predicted no-show risk" detail="Probabilities are signals, not determinations">
        <DataTable rows={asRows(data.no_show_predictions)} />
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
      <PageHeader title="OR-Tools optimization" eyebrow="Saved solver output" detail={`Planning run ${String(manifest.scenario_as_of_date ?? 'unknown')} · department/day planning over the forecast horizon`} />
      <div className="metric-grid">
        <Metric label="Pipeline status" value={String(manifest.pipeline_status ?? 'unknown').replace(/_/g, ' ')} />
        <Metric label="Forecast horizon" value={`${formatCount(manifest.forecast_horizon_days)} days`} />
        <Metric label="Scheduled appointments" value={formatCount(manifest.optimized_appointment_count)} />
        <Metric label="Next stage ready" value={manifest.next_process_ready ? 'yes' : 'no'} />
      </div>
      <Panel title="Department/day bed plan: baseline vs optimized" detail="From bed_allocation_kpis.csv. Identical rows mean projected demand fits capacity, so pooling has nothing to improve.">
        <DataTable rows={asRows(data.bed_kpis)} />
      </Panel>
      <Panel title="Daily department allocation detail"><DataTable rows={asRows(data.allocations)} /></Panel>
      <Panel title="ICU and emergency capacity"><DataTable rows={asRows(data.emergency_capacity)} /></Panel>
      <Panel title="Appointment optimization KPIs"><DataTable rows={asRows(data.appointment_kpis)} /></Panel>
      <Panel title="Staffing coverage KPIs"><DataTable rows={asRows(data.staff_kpis)} /></Panel>
      <Panel title="Scenario capacity expansion" detail="Synthetic surge beds and staff are scenario assumptions, not resources that exist"><DataTable rows={asRows(data.capacity_expansion)} /></Panel>
      <Panel title="Department resource gaps"><DataTable rows={asRows(data.resource_gaps)} /></Panel>
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
      <PageHeader title="Alerts" eyebrow="Data quality & capacity signals" detail="Alerts come from the saved audit rules, the pipeline manifest and alerts recorded through the API." />
      <label className="filter-control">Severity
        <select value={severity} onChange={(event) => setSeverity(event.target.value)}>
          <option value="all">All severities</option><option value="critical">Critical</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option>
        </select>
      </label>
      <Panel title={`${formatCount(alerts.length)} matching alert(s)`}>
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

export const SIMULATION_LIMITS = {
  admissions_multiplier: { min: 0, max: 3 },
  icu_bed_delta: { min: -20, max: 20 },
  staff_availability_delta: { min: -1, max: 1 },
};

export function validateScenario(admissionsMultiplier: number, icuBedDelta: number, staffDelta: number): string[] {
  const problems: string[] = [];
  const { admissions_multiplier: am, icu_bed_delta: icu, staff_availability_delta: staff } = SIMULATION_LIMITS;
  if (!Number.isFinite(admissionsMultiplier) || admissionsMultiplier < am.min || admissionsMultiplier > am.max) {
    problems.push(`Admissions multiplier must be between ${am.min} and ${am.max} (1.2 means +20%).`);
  }
  if (!Number.isInteger(icuBedDelta) || icuBedDelta < icu.min || icuBedDelta > icu.max) {
    problems.push(`ICU bed change must be a whole number between ${icu.min} and ${icu.max}.`);
  }
  if (!Number.isFinite(staffDelta) || staffDelta < staff.min || staffDelta > staff.max) {
    problems.push(`Staff availability change must be between ${staff.min * 100}% and +${staff.max * 100}%.`);
  }
  return problems;
}

const SIMULATION_METRICS: Array<[string, string]> = [
  ['overflow_bed_days', 'Overflow bed-days'],
  ['icu_overflow_bed_days', 'ICU overflow bed-days'],
  ['overflow_due_to_staffing', 'Overflow due to staffing'],
  ['overflow_due_to_bed_constraints', 'Overflow due to bed constraints'],
  ['unmet_isolation_bed_days', 'Unmet isolation bed-days'],
  ['staffing_constraint_violations', 'Staffing constraint violations'],
  ['emergency_icu_reserve_violations', 'Emergency/ICU reserve violations'],
  ['allocated_bed_days', 'Allocated bed-days'],
];

export function SimulatorPage() {
  const [admissionsPercent, setAdmissionsPercent] = useState('20');
  const [icuBedDelta, setIcuBedDelta] = useState('-3');
  const [staffPercent, setStaffPercent] = useState('-10');
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const multiplier = 1 + Number(admissionsPercent) / 100;
  const icuDelta = Number(icuBedDelta);
  const staffDelta = Number(staffPercent) / 100;
  const problems = [admissionsPercent, icuBedDelta, staffPercent].some((value) => value.trim() === '')
    ? ['Every assumption needs a value (use 0 for no change).']
    : validateScenario(multiplier, icuDelta, staffDelta);

  async function run(event: FormEvent) {
    event.preventDefault();
    if (problems.length) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api<SimulationResult>('/simulation/run', {
        method: 'POST',
        body: JSON.stringify({
          scenario_name: 'dashboard_what_if',
          admissions_multiplier: multiplier,
          icu_bed_delta: icuDelta,
          staff_availability_delta: staffDelta,
        }),
      }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Simulation failed');
    } finally {
      setLoading(false);
    }
  }

  const rows: ComparisonRow[] = result ? SIMULATION_METRICS
    .filter(([key]) => key in result.baseline_metrics)
    .map(([key, name]) => ({
      metric: name,
      baseline: result.baseline_metrics[key],
      scenario: result.simulated_metrics[key],
      lowerIsBetter: key !== 'allocated_bed_days',
    })) : [];

  return (
    <>
      <PageHeader title="What-if simulator" eyebrow="Scenario solver" detail="Applies your assumptions to a copy of the saved forecast and inventory, reruns the OR-Tools department/day allocator and emergency-capacity model, and compares with the saved baseline. Baseline data is never modified." />
      <Panel title="Scenario assumptions">
        <div className="prompt-list" aria-label="Preset scenarios">
          {demoConfig.scenarioPresets.map((preset) => (
            <button key={preset.label} type="button" className="prompt" onClick={() => {
              setAdmissionsPercent(String(preset.admissionsPercent));
              setIcuBedDelta(String(preset.icuBedDelta));
              setStaffPercent(String(preset.staffPercent));
            }}>{preset.label}</button>
          ))}
        </div>
        <form className="scenario-form" onSubmit={run} noValidate>
          <label>Admissions change (%)
            <input type="number" step="5" value={admissionsPercent} onChange={(event) => setAdmissionsPercent(event.target.value)} aria-describedby="scenario-help" />
          </label>
          <label>ICU beds (+/−)
            <input type="number" step="1" value={icuBedDelta} onChange={(event) => setIcuBedDelta(event.target.value)} />
          </label>
          <label>Staff availability change (%)
            <input type="number" step="5" value={staffPercent} onChange={(event) => setStaffPercent(event.target.value)} />
          </label>
          <button className="button" type="submit" disabled={loading || problems.length > 0}>{loading ? 'Running solver…' : 'Run scenario'}</button>
        </form>
        <p id="scenario-help" className="muted">Example: +20% admissions, −3 ICU beds, −10% staff availability.</p>
        {problems.length > 0 && <ul className="form-errors" role="alert">{problems.map((problem) => <li key={problem}>{problem}</li>)}</ul>}
        {error && <RequestError message={error} />}
      </Panel>
      {result && (
        <Panel title="Baseline vs scenario" detail={`${result.solver} · baseline run ${result.source_run_id} · saved as ${result.simulation_id}`}>
          <ComparisonTable rows={rows} baselineLabel="Saved baseline" scenarioLabel="Scenario" />
          <div className="metric-grid compact">
            <Metric label="Scenario forecast admissions" value={formatCount(result.simulated_metrics.projected_admissions)} note="Scaled from the saved forecast, not retrained" />
            <Metric label="Scenario peak occupied beds" value={formatCount(result.simulated_metrics.peak_projected_occupied_beds)} />
            <Metric label="Scenario unfilled staff positions" value={formatCount(result.simulated_metrics.unfilled_staff_positions)} />
          </div>
          <p className="recommendation"><strong>Solver-based recommendation:</strong> {result.operational_recommendation}</p>
          <p className="muted"><strong>Rerun:</strong> department/day bed allocator and emergency-capacity model. <strong>Not rerun:</strong> forecasting models and the request-level replay. {result.scope_note}</p>
          <p className="muted">Saved {result.timestamp}</p>
        </Panel>
      )}
    </>
  );
}

export type AssistantAnswer = {
  answer: string;
  intent: string;
  evidence: Array<{ label: string; value: unknown; kind: string; source: string }>;
  limitations: string[];
  sources: string[];
  source_run_id: string;
  evidence_run_id: string | null;
  assistant_provider: string;
  llm_configured: boolean;
  fallback_reason: string | null;
  simulation_ran: boolean;
};

const EXAMPLE_QUESTIONS: string[] = demoConfig.exampleQuestions;

const KIND_LABELS: Record<string, string> = {
  observed: 'Observed (synthetic)',
  forecast: 'Forecast',
  model_metric: 'Model metric',
  simulated: 'Simulated',
};

export function AssistantAnswerCard({ answer }: { answer: AssistantAnswer }) {
  return (
    <div className="answer">
      <h3>Answer · run {answer.evidence_run_id ?? answer.source_run_id}</h3>
      <p>{answer.answer}</p>
      {answer.fallback_reason && (
        <p className="fallback-note" role="note">The configured language model was not used: {answer.fallback_reason} This answer comes from the deterministic explainer.</p>
      )}
      {answer.evidence.length > 0 && (
        <div className="table-wrap">
          <table>
            <caption className="sr-only">Evidence</caption>
            <thead><tr><th scope="col">Evidence</th><th scope="col">Value</th><th scope="col">Type</th><th scope="col">Source</th></tr></thead>
            <tbody>{answer.evidence.map((item) => (
              <tr key={`${item.label}-${item.source}`}>
                <th scope="row">{item.label}</th>
                <td>{typeof item.value === 'number' && item.kind !== 'model_metric' && Number.isInteger(item.value) ? formatCount(item.value) : String(item.value)}</td>
                <td><span className={`kind kind-${item.kind}`}>{KIND_LABELS[item.kind] ?? item.kind}</span></td>
                <td><code>{item.source}</code></td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}
      {answer.limitations.length > 0 && <><h4>Limitations</h4><ul>{answer.limitations.map((item) => <li key={item}>{item}</li>)}</ul></>}
      <p className="muted">Sources: {answer.sources.join(', ')}{answer.simulation_ran ? ' · a new what-if simulation was run for this question' : ''}</p>
      <p className="muted">Explainer: {answer.assistant_provider} · intent {answer.intent}</p>
    </div>
  );
}

export function AssistantPage() {
  const [question, setQuestion] = useState(EXAMPLE_QUESTIONS[0]);
  const [answer, setAnswer] = useState<AssistantAnswer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setAnswer(null);
    try {
      setAnswer(await api<AssistantAnswer>('/assistant/ask', { method: 'POST', body: JSON.stringify({ question }) }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Assistant request failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <PageHeader title="CareOps assistant" eyebrow="Grounded explanations" detail="Answers are calculated from the saved forecasts, optimization runs and simulations, with the evidence and sources shown. An optional language model may rephrase them; no API key is required." />
      <Panel title="Ask about this run">
        <div className="prompt-list">{EXAMPLE_QUESTIONS.map((prompt) => <button key={prompt} type="button" className="prompt" onClick={() => setQuestion(prompt)}>{prompt}</button>)}</div>
        <form className="assistant-form" onSubmit={ask}>
          <label htmlFor="assistant-question">Question</label>
          <textarea id="assistant-question" value={question} maxLength={1000} onChange={(event) => setQuestion(event.target.value)} required />
          <button className="button" type="submit" disabled={loading || !question.trim()}>{loading ? 'Working…' : 'Ask'}</button>
        </form>
        {error && <RequestError message={error} />}
        {answer && <AssistantAnswerCard answer={answer} />}
        <p className="muted">Decision support on synthetic data only. The assistant does not make clinical decisions or take operational actions.</p>
      </Panel>
    </>
  );
}

function AuditPage() {
  const runs = useApi<Row[]>('/optimization/runs');
  const checks = useApi<Row[]>('/audit/runs');
  const assistantAudit = useApi<Row[]>('/audit/assistant');
  const simulations = useApi<Row[]>('/simulation/runs');
  const all = [runs, checks, assistantAudit, simulations];
  if (all.some((item) => item.loading)) return <Loading />;
  const error = all.find((item) => item.error)?.error;
  if (error) return <RequestError message={error} />;
  const latest = (runs.data ?? [])[0];
  return (
    <>
      <PageHeader title="Audit & run history" eyebrow="Reproducibility" detail="Pipeline manifest, data-quality checks, assistant requests and persisted what-if simulations from this environment's database." />
      <Panel title="Pipeline runs">
        {latest ? <JsonBlock value={latest} /> : <Empty>No completed pipeline runs are recorded.</Empty>}
      </Panel>
      <Panel title="Data-quality audit"><DataTable rows={checks.data ?? []} /></Panel>
      <Panel title="Assistant request audit"><DataTable rows={assistantAudit.data ?? []} /></Panel>
      <Panel title="What-if simulation history">
        <DataTable rows={(simulations.data ?? []).map((run) => ({
          simulation_id: run.simulation_id, timestamp: run.timestamp, scenario: run.scenario_name,
          ...asObject(run.assumptions),
          scenario_overflow_bed_days: asObject(run.simulated_metrics).overflow_bed_days,
          scenario_unfilled_staff: asObject(run.simulated_metrics).unfilled_staff_positions,
        }))} />
      </Panel>
    </>
  );
}

function AboutPage() {
  return (
    <>
      <PageHeader title="About CareOps AI" eyebrow="How it works" detail="A reproducible hospital resource-planning project on synthetic data." />
      <Panel title="Data-to-decision flow">
        <ol className="flow-list">
          <li>Seeded synthetic patients, admissions, beds, staff, appointments, shifts and daily operations</li>
          <li>Admission, occupancy and ICU-admission forecasts; LOS and no-show models; each backtested against naive baselines</li>
          <li>Bed availability forecast from admissions, predicted LOS and the current census</li>
          <li>OR-Tools CP-SAT: department/day bed plan, request-level bed allocation replay, staff and appointment scheduling, emergency capacity</li>
          <li>FastAPI serves the saved results; this dashboard shows those API responses</li>
          <li>The assistant answers from calculated results with evidence and sources; an optional LLM may only rephrase</li>
        </ol>
      </Panel>
      <Panel title="Limitations">
        <ul>
          <li>All data and results are synthetic and do not describe a real hospital.</li>
          <li>Daily admission and ICU counts are close to Poisson, with small weekday and seasonal effects; no learned model beats the moving-average baseline for them, so the baseline is published.</li>
          <li>The ICU model forecasts admissions, not ICU census.</li>
          <li>The synthetic shift roster is not tied to ward census; rostered staffing is reported, not enforced, in the allocation replay.</li>
          <li>Surge beds and staff in the capacity plan are scenario assumptions.</li>
          <li>Never use these outputs for clinical decisions or real placement or staffing orders.</li>
        </ul>
      </Panel>
      <Panel title="Service links"><p><a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">FastAPI documentation</a> · <a href={`${API_URL}/health`} target="_blank" rel="noreferrer">Backend health</a></p></Panel>
    </>
  );
}

// Admission forecasts include an "All Departments" total row per day; sum departments only.
function departmentRows(rows: Row[]): Row[] {
  return rows.filter((row) => String(row.department ?? '').toLowerCase() !== 'all departments');
}

function sumByDate(rows: Row[], field: string) {
  const totals = new Map<string, number>();
  rows.forEach((row) => {
    const date = String(row.date ?? '');
    const value = toNumber(row[field]);
    if (!date || value === null) return;
    totals.set(date, (totals.get(date) ?? 0) + value);
  });
  const labels = [...totals.keys()].sort();
  return { labels, values: labels.map((date) => totals.get(date) ?? 0) };
}

function mean(values: Array<number | null>): number | null {
  const valid = values.filter((value): value is number => value !== null && Number.isFinite(value));
  return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : null;
}
