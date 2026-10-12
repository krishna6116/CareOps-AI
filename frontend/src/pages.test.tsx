import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  AssistantAnswerCard,
  BedAssignmentSection,
  OverviewPage,
  SimulatorPage,
  validateScenario,
  type AssistantAnswer,
} from './App';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function mockApi(routes: Record<string, unknown>, failing: string[] = []) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const path = Object.keys(routes).find((route) => url.endsWith(route) || url.includes(`${route}?`));
    if (failing.some((route) => url.includes(route))) {
      return new Response(JSON.stringify({ detail: 'Backend failed for test' }), { status: 500 });
    }
    if (!path) return new Response(JSON.stringify({ detail: `No mock for ${url}` }), { status: 404 });
    return new Response(JSON.stringify(routes[path]), { status: 200 });
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

const OVERVIEW = {
  total_beds: 230, occupied_beds: 144, available_beds: 77, unavailable_beds: 9, utilization: 62.61,
  predicted_admissions: 338.4, predicted_occupancy: 140.6, icu_total_beds: 20, icu_occupied_beds: 17,
  icu_available_beds: 2, emergency_total_beds: 10, emergency_available_beds: 4, alerts: 1,
  optimization_status: 'ready_for_next_process', scenario: '2026-10-05', latest_forecast_date: '2026-11-04',
  forecast_horizon_days: 30, latest_optimization_run: '2026-10-05', operational_recommendations: [],
};

const ASSIGNMENT = {
  window: ['2026-09-08', '2026-10-05'],
  baseline_policy: 'Arrival-order first fit into the home ward.',
  solver_statuses: { OPTIMAL: 28 },
  reason_definitions: { protected_reserve: 'Held for critical requests.' },
  kpis: [
    { policy: 'baseline', requests: 315, assigned: 288, unassigned: 27, assignment_rate: 0.9143, high_severity_assignment_rate: 0.9244, off_service_placements: 0, compatibility_violations: 0, staffing_exceedances: 263, unassigned_protected_reserve: 5 },
    { policy: 'optimized', requests: 315, assigned: 311, unassigned: 4, assignment_rate: 0.9873, high_severity_assignment_rate: 0.9916, off_service_placements: 22, compatibility_violations: 0, staffing_exceedances: 279, unassigned_protected_reserve: 2 },
  ],
  sensitivity: [],
};

describe('OverviewPage', () => {
  it('shows a loading state, then whole-number counts from the API', async () => {
    mockApi({
      '/dashboard/overview': OVERVIEW,
      '/forecast/admissions': [{ date: '2026-10-06', department: 'ICU', predicted_admissions: 1 }],
      '/forecast/occupancy': [{ date: '2026-10-06', forecast_occupied_beds: 144 }],
      '/forecast/icu': [{ date: '2026-10-06', predicted_icu_admissions: 1 }],
      '/alerts': [],
      '/optimization/bed-assignments': ASSIGNMENT,
    });
    render(<OverviewPage />);
    expect(screen.getByRole('status')).toBeTruthy();
    expect(await screen.findByText('338')).toBeTruthy();
    expect(screen.getByText('141')).toBeTruthy();
    expect(screen.getByText('62.6%')).toBeTruthy();
    expect(await screen.findByText('311 / 315')).toBeTruthy();
    expect(screen.queryByText(/338\.4|140\.6/)).toBeNull();
  });

  it('shows the API error instead of figures when the backend fails', async () => {
    mockApi({ '/dashboard/overview': OVERVIEW }, ['/forecast/admissions']);
    render(<OverviewPage />);
    expect((await screen.findByRole('alert')).textContent).toContain('Backend failed for test');
  });
});

describe('BedAssignmentSection', () => {
  it('compares baseline and optimized and handles no unassigned rows', async () => {
    mockApi({
      '/optimization/bed-assignments': ASSIGNMENT,
      '/optimization/bed-assignments/requests': { total: 0, offset: 0, limit: 500, rows: [] },
    });
    render(<BedAssignmentSection />);
    expect(await screen.findByText('+23')).toBeTruthy();
    expect(screen.getByText('98.7%')).toBeTruthy();
    expect(await screen.findByText('Every request was placed under this policy.')).toBeTruthy();
  });
});

describe('SimulatorPage', () => {
  it('blocks out-of-range assumptions before calling the API', () => {
    const fetchMock = mockApi({});
    render(<SimulatorPage />);
    fireEvent.change(screen.getByLabelText('Admissions change (%)'), { target: { value: '500' } });
    fireEvent.change(screen.getByLabelText('ICU beds (+/−)'), { target: { value: '1.5' } });
    expect(screen.getByRole('alert').textContent).toContain('Admissions multiplier must be between 0 and 3');
    expect(screen.getByRole('alert').textContent).toContain('ICU bed change must be a whole number');
    const button = screen.getByRole('button', { name: 'Run scenario' }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    fireEvent.click(button);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('submits the converted assumptions and shows whole-number results', async () => {
    const fetchMock = mockApi({
      '/simulation/run': {
        simulation_id: 'sim-1', scenario_name: 'dashboard_what_if',
        assumptions: { admissions_multiplier: 1.2, icu_bed_delta: -3, staff_availability_delta: -0.1 },
        baseline_metrics: { overflow_bed_days: 0, allocated_bed_days: 4174 },
        simulated_metrics: { overflow_bed_days: 12, allocated_bed_days: 4980, projected_admissions: 406, peak_projected_occupied_beds: 171, unfilled_staff_positions: 5 },
        operational_recommendation: 'Review overflow.', timestamp: '2026-10-12T00:00:00Z',
        source_run_id: '2026-10-05', solver: 'OR-Tools CP-SAT', scope_note: 'Synthetic.',
      },
    });
    render(<SimulatorPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Run scenario' }));
    expect(await screen.findByText('+12')).toBeTruthy();
    const body = JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body));
    expect(body).toMatchObject({ admissions_multiplier: 1.2, icu_bed_delta: -3, staff_availability_delta: -0.1 });
    expect(screen.getByText('406')).toBeTruthy();
  });

  it('validates assumptions with the backend limits', () => {
    expect(validateScenario(1.2, -3, -0.1)).toEqual([]);
    expect(validateScenario(3.5, 0, 0)).toHaveLength(1);
    expect(validateScenario(1, 25, 0)).toHaveLength(1);
    expect(validateScenario(1, 0, -1.5)).toHaveLength(1);
  });
});

describe('AssistantAnswerCard', () => {
  const answer: AssistantAnswer = {
    answer: 'For 2026-10-12 the forecast leaves 90 available beds.',
    intent: 'bed_availability',
    evidence: [
      { label: 'forecast available beds', value: 90, kind: 'forecast', source: 'bed_availability_forecast.csv' },
      { label: 'Occupied beds MAE', value: '8.72', kind: 'model_metric', source: 'docs/MODEL_METRICS.md' },
    ],
    limitations: ['Synthetic portfolio data; not real hospital performance or clinical advice.'],
    sources: ['bed_availability_forecast.csv'],
    source_run_id: '2026-10-05', evidence_run_id: '2026-10-05',
    assistant_provider: 'deterministic', llm_configured: false,
    fallback_reason: 'OPENAI_API_KEY is not configured.', simulation_ran: false,
  };

  it('shows evidence types, sources and the fallback reason', () => {
    render(<AssistantAnswerCard answer={answer} />);
    expect(screen.getByText('Forecast')).toBeTruthy();
    expect(screen.getByText('Model metric')).toBeTruthy();
    expect(screen.getByText('8.72')).toBeTruthy();
    expect(screen.getByRole('note').textContent).toContain('OPENAI_API_KEY is not configured.');
    expect(screen.getByText(/Sources: bed_availability_forecast.csv/)).toBeTruthy();
  });

  it('reports assistant API errors', async () => {
    mockApi({}, ['/assistant/ask']);
    const { AssistantPage } = await import('./App');
    render(<AssistantPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('Backend failed for test'));
  });
});
