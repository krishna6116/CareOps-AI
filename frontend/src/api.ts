import { useEffect, useState } from 'react';

export type Row = Record<string, unknown>;

export const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
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
