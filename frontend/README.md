# Frontend

React 19 + TypeScript single-page app for the Basal Area Growth Tool, built
with Vite 7, Tailwind CSS 4, Radix UI primitives, Recharts and Lucide icons.
See the [root README](../README.md) for the full project overview.

## Run

```bash
npm install
npm run dev        # http://localhost:5173, calls the API at VITE_API_BASE_URL
npm run build      # type-check + production bundle in dist/
npm run preview    # serve the production bundle locally
```

Copy `.env.example` to `.env.local` to change the backend URL (default
`http://localhost:8000`).

## Quality

```bash
npm run lint             # ESLint (flat config)
npx playwright test      # 7 end-to-end tests, including an axe accessibility scan
```

Playwright starts the Vite dev server itself. Run the backend on port 8000
beforehand so the live-data tabs have the synthetic demo database to talk to.

## Layout

```
src/
├── App.tsx                 Tab shell, theme and language state
├── components/
│   ├── Header.tsx          Title, demo badge, theme and language toggles
│   ├── DataEntry.tsx       Connection check and plot entry
│   ├── GrowthCalculator.tsx, BAGrowthCalculator.tsx
│   ├── CoefficientDisplay.tsx
│   ├── ModelBenchmarks.tsx Leaderboard, charts, leakage / interval / importance panels
│   ├── Visualization.tsx, Reports.tsx
│   └── ui/                 Radix-based primitives
├── hooks/useBAGrowthModel.ts
└── types/api.ts            API response types
```
