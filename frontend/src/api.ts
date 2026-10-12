import { useEffect, useState } from 'react';

export type Row = Record<string, unknown>;

export const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

// Static demo (GitHub Pages): responses were exported from the real API by
// deploy/export_demo_snapshot.py; there is no live backend.
export const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'snapshot';
const DEMO_ROOT = `${import.meta.env.BASE_URL}demo/`;

/** Path to snapshot file name; must match ``slug`` in deploy/export_demo_snapshot.py. */
export function snapshotFile(path: string): string {
  return `${path.replace(/^\//, '').replace(/[^A-Za-z0-9]+/g, '_').replace(/_+$/, '')}.json`;
}

/** Must match ``scenario_key`` in deploy/export_demo_snapshot.py. */
export function scenarioKey(admissionsMultiplier: number, icuBedDelta: number, staffDelta: number): string {
  return `a${admissionsMultiplier.toFixed(2)}_i${Math.trunc(icuBedDelta)}_s${staffDelta.toFixed(2)}`;
}

async function readSnapshot<T>(file: string): Promise<T> {
  const response = await fetch(`${DEMO_ROOT}${file}`);
  if (!response.ok) throw new Error('This view is not included in the static demo. Run the project locally for the live API.');
  return response.json() as Promise<T>;
}

async function demoApi<T>(path: string, init?: RequestInit): Promise<T> {
  if (!init?.method || init.method === 'GET') return readSnapshot<T>(snapshotFile(path));
  const body = JSON.parse(String(init.body ?? '{}')) as Record<string, unknown>;
  if (path === '/assistant/ask') {
    const answers = await readSnapshot<Record<string, T>>('assistant.json');
    const question = String(body.question ?? '').trim().toLowerCase();
    const match = Object.entries(answers).find(([text]) => text.toLowerCase() === question);
    if (match) return match[1];
    throw new Error('The static demo answers the example questions above. Run the project locally to ask anything else.');
  }
  if (path === '/simulation/run') {
    const runs = await readSnapshot<Record<string, T>>('simulations.json');
    const key = scenarioKey(Number(body.admissions_multiplier), Number(body.icu_bed_delta), Number(body.staff_availability_delta));
    if (runs[key]) return runs[key];
    throw new Error('The static demo includes the preset scenarios only. Run the project locally to solve any scenario.');
  }
  throw new Error('This action needs the live API.');
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  if (DEMO_MODE) return demoApi<T>(path, init);
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: {
        ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
        ...init?.headers,
      },
    });
  } catch {
    throw new Error(`The CareOps API at ${API_URL} could not be reached.`);
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => null) as { detail?: unknown } | null;
    const message = typeof detail?.detail === 'string'
      ? detail.detail
      : Array.isArray(detail?.detail)
        ? 'The request was rejected: check the input values.'
        : `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function useApi<T>(path: string | null) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(path !== null);

  useEffect(() => {
    if (path === null) return undefined;
    let active = true;
    setLoading(true);
    setError(null);
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

export function asRows(value: unknown): Row[] {
  return Array.isArray(value) ? value.filter((row): row is Row => typeof row === 'object' && row !== null) : [];
}

export function asObject(value: unknown): Row {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Row : {};
}
