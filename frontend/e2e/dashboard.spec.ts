import { expect, test, type Page } from '@playwright/test';

const demo = process.env.E2E_MODE === 'demo';
const shots = process.env.E2E_SCREENSHOTS ?? 'e2e-screenshots';

// The static demo uses hash routes; the live dashboard uses paths.
const url = (route: string) => (demo ? `./#${route}` : `.${route}`);

const PAGES: Array<[string, string, string]> = [
  ['/', 'Executive overview', 'overview'],
  ['/forecasting', 'Forecasting & model evaluation', 'forecasting'],
  ['/beds', 'Bed management', 'beds'],
  ['/staff', 'Staff scheduling', 'staff'],
  ['/appointments', 'Appointments', 'appointments'],
  ['/optimization', 'OR-Tools optimization', 'optimization'],
  ['/alerts', 'Alerts', 'alerts'],
  ['/simulator', 'What-if simulator', 'simulator'],
  ['/assistant', 'CareOps assistant', 'assistant'],
  ['/audit', 'Audit & run history', 'audit'],
  ['/about', 'About CareOps AI', 'about'],
];

function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  return errors;
}

async function settle(page: Page) {
  await expect(page.getByRole('status')).toHaveCount(0);
}

for (const [route, heading, name] of PAGES) {
  test(`${name} page renders real data without errors`, async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await page.goto(url(route));
    await expect(page.getByRole('heading', { level: 1, name: heading })).toBeVisible();
    await settle(page);
    await expect(page.locator('.error-state')).toHaveCount(0);
    if (demo) await expect(page.locator('.demo-banner')).toContainText('Static demo');
    await page.screenshot({ path: `${shots}/${demo ? 'demo' : 'live'}-${name}.png`, fullPage: true });
    expect(errors, errors.join('\n')).toEqual([]);
  });
}

test('overview counts are whole numbers', async ({ page }) => {
  await page.goto(url('/'));
  await settle(page);
  const values = await page.locator('.metric-card strong').allTextContents();
  expect(values.length).toBeGreaterThan(5);
  for (const value of values) {
    // Percentages may have a decimal; counts and "x / y" pairs may not.
    if (!value.includes('%')) expect(value, value).not.toMatch(/\d\.\d/);
  }
});

test('assistant answers an example question with evidence', async ({ page }) => {
  await page.goto(url('/assistant'));
  await page.getByRole('button', { name: 'Why were some bed requests left unassigned?' }).click();
  await page.getByRole('button', { name: 'Ask' }).click();
  const answer = page.locator('.answer');
  await expect(answer).toContainText('requests unassigned');
  await expect(answer.getByRole('table')).toContainText('Simulated');
  await expect(answer).toContainText('bed_assignment_summary.json');
  await page.screenshot({ path: `${shots}/${demo ? 'demo' : 'live'}-assistant-answer.png`, fullPage: true });
});

test('simulator runs a preset scenario and compares it with the baseline', async ({ page }) => {
  await page.goto(url('/simulator'));
  await page.getByRole('button', { name: 'Admissions +20%' }).click();
  await expect(page.getByLabel('Admissions change (%)')).toHaveValue('20');
  await page.getByRole('button', { name: 'Run scenario' }).click();
  const comparison = page.locator('.comparison-table');
  await expect(comparison).toContainText('Overflow bed-days', { timeout: 120_000 });
  await expect(page.getByText('Not rerun:')).toBeVisible();
  await page.screenshot({ path: `${shots}/${demo ? 'demo' : 'live'}-simulator-result.png`, fullPage: true });
});

test('bed allocation lists unassigned requests for both policies', async ({ page }) => {
  await page.goto(url('/beds'));
  await expect(page.getByText('CP-SAT optimized').first()).toBeVisible();
  await page.getByLabel('Policy').selectOption('baseline');
  await expect(page.getByText('No compatible bed in inventory').first()).toBeVisible();
});

test('invalid simulator input is blocked before any request', async ({ page }) => {
  await page.goto(url('/simulator'));
  await page.getByLabel('Admissions change (%)').fill('500');
  await expect(page.getByRole('alert')).toContainText('Admissions multiplier must be between 0 and 3');
  await expect(page.getByRole('button', { name: 'Run scenario' })).toBeDisabled();
});
