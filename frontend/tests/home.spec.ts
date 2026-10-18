import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Basal Area Growth Rate Tool demo', () => {
  test('homepage loads with main title and tabs', async ({ page }) => {
    await page.goto('/');

    await expect(
      page.getByRole('heading', { name: /basal area growth rate tool/i })
    ).toBeVisible();

    await expect(
      page.getByText(/forest growth & yield analytics system/i)
    ).toBeVisible();

    await expect(page.getByRole('tab', { name: /data entry/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /calculations/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /model coefficients/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /model benchmarks/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /visualization/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /reports/i })).toBeVisible();
  });

  test('language toggle works', async ({ page }) => {
    await page.goto('/');

    const languageButton = page.getByRole('button', { name: /français/i });
    await expect(languageButton).toBeVisible();
    await languageButton.click();

    await expect(
      page.getByRole('heading', {
        name: /outil de taux de croissance de surface terrière/i,
      })
    ).toBeVisible();

    await expect(
      page.getByText(/système d'analyse de croissance et de rendement forestier/i)
    ).toBeVisible();

    await expect(page.getByRole('tab', { name: /saisie de données/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /calculs/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /coefficients du modèle/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /visualisation/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /rapports/i })).toBeVisible();

    await expect(page.getByRole('button', { name: /english/i })).toBeVisible();
  });

  test('theme toggle works', async ({ page }) => {
    await page.goto('/');

    const html = page.locator('html');
    const initialClass = (await html.getAttribute('class')) ?? '';

    const themeButton = page.getByRole('button', {
      name: /switch to dark mode|switch to light mode/i,
    });

    await expect(themeButton).toBeVisible();
    await themeButton.click();

    const newClass = (await html.getAttribute('class')) ?? '';
    expect(newClass).not.toBe(initialClass);
  });

  test('tab navigation works', async ({ page }) => {
    await page.goto('/');

    await page.getByRole('tab', { name: /model coefficients/i }).click();
    await expect(
      page.getByText(/regression model coefficients/i)
    ).toBeVisible();

    await page.getByRole('tab', { name: /data entry/i }).click();
    await expect(page.getByText(/database connection/i)).toBeVisible();

    await page.getByRole('tab', { name: /reports/i }).click();
    await expect(page.getByRole('tab', { name: /reports/i })).toHaveAttribute(
      'aria-selected',
      'true'
    );
  });

  test('model benchmarks tab renders leaderboard or empty state', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('tab', { name: /model benchmarks/i }).click();
    await expect(page.getByRole('heading', { name: /model benchmarks/i })).toBeVisible();
    // Either a report is loaded (leaderboard table) or the empty-state alert is shown.
    const leaderboard = page.getByText(/leaderboard/i).first();
    const emptyState = page.getByText(/no benchmark report yet/i);
    await expect(leaderboard.or(emptyState)).toBeVisible({ timeout: 15000 });
  });

  test('accessibility scan on homepage', async ({ page }) => {
    await page.goto('/');

    const results = await new AxeBuilder({ page }).analyze();

    console.log(
      'Axe violations:',
      JSON.stringify(results.violations, null, 2)
    );

    const seriousOrCritical = results.violations.filter((v) =>
      ['serious', 'critical'].includes(v.impact ?? '')
    );

    expect(seriousOrCritical).toEqual([]);
  });

  test('visual snapshot', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveScreenshot('home.png', { fullPage: true });
  });
});