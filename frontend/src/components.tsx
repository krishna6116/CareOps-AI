import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';

import { API_URL, DEMO_MODE, type Row } from './api';
import { display, formatCount, formatSigned, label, toNumber } from './format';

export type Series = { name: string; values: Array<number | null> };

export function PageHeader({ title, eyebrow, detail }: { title: string; eyebrow: string; detail?: string }) {
  return (
    <header className="page-header">
      <div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1>{detail && <p className="page-detail">{detail}</p>}</div>
      <span className="synthetic-badge">Synthetic data</span>
    </header>
  );
}

export function Panel({ title, detail, wide = false, children }: { title: string; detail?: string; wide?: boolean; children: ReactNode }) {
  return <section className={wide ? 'panel panel-wide' : 'panel'}><div className="panel-heading"><h2>{title}</h2>{detail && <p>{detail}</p>}</div>{children}</section>;
}

export function Metric({ label: name, value, note }: { label: string; value: string | number | null; note?: string }) {
  return <article className="metric-card"><span>{name}</span><strong>{value ?? 'Unavailable'}</strong>{note && <small>{note}</small>}</article>;
}

// Line chart for daily series. ``counts`` renders axis ticks and point
// tooltips as whole numbers.
export function TrendChart({ labels, series, counts = true }: { labels: string[]; series: Series[]; counts?: boolean }) {
  const width = 760;
  const height = 230;
  const pad = 44;
  const allValues = series.flatMap((line) => line.values.filter((value): value is number => value !== null && Number.isFinite(value)));
  if (!labels.length || !allValues.length) return <Empty>No chart data is available.</Empty>;
  const min = Math.min(0, ...allValues);
  const rawMax = Math.max(...allValues, min + 1);
  const max = counts ? Math.ceil(rawMax / 3) * 3 : rawMax;
  const colors = ['#65a5ff', '#47d7ae', '#ffc66e'];
  const x = (index: number) => pad + (labels.length <= 1 ? 0 : index * (width - pad * 2) / (labels.length - 1));
  const y = (value: number) => height - pad - (value - min) * (height - pad * 2) / (max - min || 1);
  const tick = (value: number) => (counts ? formatCount(value) : value.toFixed(2));
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
        {[0, 1, 2, 3].map((step) => {
          const value = min + (max - min) * step / 3;
          const lineY = y(value);
          return <g key={step}><line x1={pad} x2={width - pad} y1={lineY} y2={lineY} className="chart-grid" /><text x="2" y={lineY + 6} className="chart-label">{tick(value)}</text></g>;
        })}
        {paths.map((path, index) => <path key={series[index].name} d={path} fill="none" stroke={colors[index % colors.length]} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />)}
        {series.flatMap((line) => line.values.map((value, index) => (
          value === null || !Number.isFinite(value) ? null : (
            <circle key={`${line.name}-${index}`} cx={x(index)} cy={y(value)} r="6" className="chart-hit">
              <title>{`${labels[index]} · ${line.name}: ${tick(value)}`}</title>
            </circle>
          )
        )))}
      </svg>
      <div className="chart-legend">{series.map((line, index) => <span key={line.name}><i style={{ backgroundColor: colors[index % colors.length] }} />{line.name}</span>)}</div>
      <div className="chart-range"><span>{labels[0]}</span><span>{labels[labels.length - 1]}</span></div>
    </div>
  );
}

export function DataTable({ rows, caption }: { rows: Row[]; caption?: string }) {
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
          <div className="table-wrap"><table>{caption && <caption className="sr-only">{caption}</caption>}<thead><tr>{columns.map((column) => <th key={column} scope="col">{label(column)}</th>)}</tr></thead><tbody>
            {visibleRows.map((row, index) => <tr key={`${String(row.id ?? row.admission_id ?? row.appointment_id ?? row.date ?? 'row')}-${currentPage * pageSize + index}`}>{columns.map((column) => {
              const text = display(row[column]);
              return <td key={column} className={text.length > 48 ? 'cell-long' : undefined}>{text}</td>;
            })}</tr>)}
          </tbody></table></div>
          {pageCount > 1 && (
            <div className="table-pagination">
              <button type="button" className="button button-secondary" onClick={() => setPage(currentPage - 1)} disabled={currentPage === 0}>Previous</button>
              <span>Page {currentPage + 1} of {pageCount}</span>
              <button type="button" className="button button-secondary" onClick={() => setPage(currentPage + 1)} disabled={currentPage + 1 >= pageCount}>Next</button>
            </div>
          )}
        </>
      ) : <Empty>No records match this search.</Empty>}
    </>
  );
}

export type ComparisonRow = { metric: string; baseline: unknown; scenario: unknown; format?: (value: unknown) => string; lowerIsBetter?: boolean };

// Baseline vs scenario/optimized table with a signed change column.
export function ComparisonTable({ rows, baselineLabel = 'Baseline', scenarioLabel = 'Scenario' }: { rows: ComparisonRow[]; baselineLabel?: string; scenarioLabel?: string }) {
  return (
    <div className="table-wrap">
      <table className="comparison-table">
        <thead><tr><th scope="col">Metric</th><th scope="col">{baselineLabel}</th><th scope="col">{scenarioLabel}</th><th scope="col">Change</th></tr></thead>
        <tbody>
          {rows.map((row) => {
            const formatter = row.format ?? formatCount;
            const base = toNumber(row.baseline);
            const scen = toNumber(row.scenario);
            const delta = base === null || scen === null ? null : scen - base;
            const tone = delta === null || delta === 0 || row.lowerIsBetter === undefined
              ? '' : (delta < 0) === row.lowerIsBetter ? 'delta-better' : 'delta-worse';
            return (
              <tr key={row.metric}>
                <th scope="row">{row.metric}</th>
                <td>{formatter(row.baseline)}</td>
                <td>{formatter(row.scenario)}</td>
                <td className={tone}>{formatSigned(delta, formatter)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>;
}

export function Loading() {
  return <div className="state-card" role="status">Loading CareOps data…</div>;
}

export function RequestError({ message }: { message: string }) {
  return (
    <div className="state-card error-state" role="alert">
      <strong>Could not load this data</strong>
      <p>{message}</p>
      <small>{DEMO_MODE ? 'This is the static demo; some actions need the live API.' : `Check that the FastAPI backend is running and reachable at ${API_URL}.`}</small>
      <p><button className="button" type="button" onClick={() => window.location.reload()}>Retry</button></p>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="empty-state">{children}</p>;
}
