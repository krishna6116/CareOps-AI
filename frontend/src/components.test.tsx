import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { ComparisonTable, DataTable, Metric, TrendChart } from './components';
import { formatPercent } from './format';

afterEach(cleanup);

describe('Metric', () => {
  it('shows Unavailable instead of an empty value', () => {
    render(<Metric label="Forecast occupied beds" value={null} />);
    expect(screen.getByText('Unavailable')).toBeTruthy();
  });
});

describe('ComparisonTable', () => {
  it('formats counts as whole numbers and signs the change', () => {
    render(<ComparisonTable rows={[
      { metric: 'Assigned', baseline: 288, scenario: 311.4, lowerIsBetter: false },
      { metric: 'Assignment rate', baseline: 0.9143, scenario: 0.9873, format: (value) => formatPercent(value) },
    ]} />);
    const assigned = screen.getByText('Assigned').closest('tr') as HTMLElement;
    expect(within(assigned).getByText('311')).toBeTruthy();
    expect(within(assigned).getByText('+23').className).toBe('delta-better');
    expect(screen.getByText('98.7%')).toBeTruthy();
  });
});

describe('TrendChart', () => {
  it('labels count axes and tooltips without decimals', () => {
    const { container } = render(
      <TrendChart labels={['2026-10-06', '2026-10-07']} series={[{ name: 'Forecast admissions', values: [10.6, 11.2] }]} />,
    );
    const labels = Array.from(container.querySelectorAll('text')).map((node) => node.textContent ?? '');
    const tooltips = Array.from(container.querySelectorAll('title')).map((node) => node.textContent ?? '');
    expect(labels.length).toBeGreaterThan(0);
    for (const text of [...labels, ...tooltips.map((tip) => tip.split(': ')[1])]) {
      expect(text).toMatch(/^\d[\d,]*$/);
    }
  });

  it('shows an empty state without data', () => {
    render(<TrendChart labels={[]} series={[]} />);
    expect(screen.getByText('No chart data is available.')).toBeTruthy();
  });
});

describe('DataTable', () => {
  it('shows an empty state for no rows', () => {
    render(<DataTable rows={[]} />);
    expect(screen.getByText('No rows are available.')).toBeTruthy();
  });

  it('filters and paginates', () => {
    const rows = Array.from({ length: 30 }, (_, index) => ({ id: index, department: index % 2 ? 'ICU' : 'Surgery' }));
    render(<DataTable rows={rows} />);
    expect(screen.getByText('Showing 1–25 of 30 records')).toBeTruthy();
    fireEvent.click(screen.getByText('Next'));
    expect(screen.getByText('Page 2 of 2')).toBeTruthy();
    fireEvent.change(screen.getByPlaceholderText('Search any column'), { target: { value: 'icu' } });
    expect(screen.getByText('Showing 1–15 of 15 records')).toBeTruthy();
  });
});
