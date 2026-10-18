import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  use: {
    baseURL: 'http://127.0.0.1:5173',
    headless: true,
    // PW_CHANNEL=chrome lets the suite drive an installed Google Chrome when
    // Playwright's bundled browsers are not downloaded (e.g. restricted networks).
    channel: process.env.PW_CHANNEL || undefined,
  },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 5173',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
});