import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Alert, AlertDescription } from "./ui/alert";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import {
  Activity,
  AlertCircle,
  ArrowDown,
  ArrowUp,
  Download,
  FlaskConical,
  Info,
  Play,
  RefreshCw,
  ShieldCheck,
  Trophy,
} from "lucide-react";
import type { Language } from "../App";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types mirroring backend/ml/benchmark.py output
// ---------------------------------------------------------------------------

interface Metrics {
  r2?: number;
  adj_r2?: number;
  rmse?: number;
  mae?: number;
  median_ae?: number;
  p90_ae?: number;
  p95_ae?: number;
  p99_ae?: number;
  bias?: number;
  n?: number;
}

interface VsBaseline {
  rmse_improvement_pct?: number | null;
  p95_improvement_pct?: number | null;
  paired_t_on_fold_rmse?: { p_value?: number | null };
  wilcoxon_on_abs_errors?: { p_value?: number | null };
}

interface ModelEntry {
  family: string;
  description: string;
  linear: boolean;
  rank: number;
  metrics_mean: Metrics;
  metrics_std: Metrics;
  pooled_oof?: Metrics;
  vs_baseline?: VsBaseline | null;
  sign_check?: { wrong_signs: string[]; wrong_sign_count: number } | null;
  coefficients?: Record<string, number>;
  other_strategies?: Record<string, { metrics_mean: Metrics }>;
  bootstrap_ci?: { rmse?: number[]; p95_ae?: number[] };
  wall_seconds?: number;
}

interface IntervalResult {
  method: string;
  model?: string;
  target_coverage?: number;
  coverage?: number;
  mean_width?: number;
  calibrated?: boolean;
}

interface GroupReport {
  group: string;
  dataset: { n_samples: number; n_plots: number; year_range?: number[] };
  cv?: { strategy: string; n_splits: number; plot_overlap_between_train_and_test: number };
  models: Record<string, ModelEntry>;
  ranking?: string[];
  best_model?: string;
  error?: string;
  leakage_check?: {
    median_r2_inflation?: number | null;
    mean_r2_inflation: number | null;
    per_model: Record<string, { r2_grouped: number; r2_naive_kfold: number; r2_inflation: number }>;
  };
  intervals?: Record<string, IntervalResult | { error: string }>;
  importance?: { model: string; features: { feature: string; importance_mean: number; importance_std: number }[] };
}

interface BenchmarkReport {
  generated_at: string;
  elapsed_seconds: number;
  config: { cv_strategy: string; n_splits: number; feature_set: string; alpha: number; models: string[] };
  groups: Record<string, GroupReport>;
  summary: {
    wins_by_model: Record<string, number>;
    mean_rank_by_model: Record<string, number>;
    overall_best: string | null;
    baseline: string;
  };
}

interface RegistryModel {
  name: string;
  family: string;
  description: string;
  available: boolean;
  requires?: string | null;
}

interface Job {
  job_id: string;
  status: string;
  progress: { total: number; done: number; failed: number; current: string | null };
  errors: Record<string, string>;
}

interface Props {
  language: Language;
}

type SortKey = "rank" | "rmse" | "r2" | "mae" | "p95" | "bias" | "vsOls";

const FAMILY_COLOR: Record<string, string> = {
  linear: "#2563eb",
  additive: "#7c3aed",
  kernel: "#0891b2",
  ensemble: "#059669",
  neural: "#d97706",
  mixed: "#db2777",
};

const AXIS_TICK = { fill: "var(--foreground)", fontSize: 12 };
const FAST_MODELS = ["ols", "ridge", "huber", "spline_gam", "random_forest", "hist_gbm", "mixed_effects"];

const fmt = (v: number | null | undefined, d = 5) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(d);
const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(1)}%`;
const pval = (v: number | null | undefined) =>
  v === null || v === undefined ? "–" : v < 0.001 ? "<0.001" : v.toFixed(3);

export function ModelBenchmarks({ language }: Props) {
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [reportFile, setReportFile] = useState<string>("");
  const [registry, setRegistry] = useState<RegistryModel[]>([]);
  const [group, setGroup] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [fastRun, setFastRun] = useState(true);
  const [cvStrategy, setCvStrategy] = useState<"group" | "temporal" | "kfold">("group");
  const [featureSet, setFeatureSet] = useState<"base" | "extended">("base");
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortAsc, setSortAsc] = useState(true);

  const t = useMemo(
    () =>
      language === "fr"
        ? {
            title: "Comparaison des modèles",
            description:
              "Validation croisée par placette (sans fuite) de 17 algorithmes candidats contre l'équation RGR de production.",
            noReport: "Aucun rapport de comparaison. Lancez-en un ci-dessous ou exécutez scripts/benchmark_models.py --all.",
            run: "Lancer",
            runFast: "Modèles rapides (7)",
            runFull: "Tous les modèles (17)",
            running: "En cours…",
            refresh: "Actualiser",
            download: "JSON",
            group: "Groupe de modèle",
            cv: "Validation croisée",
            cvGroup: "Par placette (recommandé)",
            cvTemporal: "Temporelle (avant → après)",
            cvKfold: "K-fold naïve (fuite)",
            features: "Variables",
            featBase: "Équation (7)",
            featExt: "Étendues (11)",
            leaderboard: "Classement",
            rank: "Rang",
            model: "Modèle",
            family: "Famille",
            vsOls: "vs OLS",
            pValue: "p (t apparié)",
            signs: "Signes erronés",
            summary: "Résumé toutes espèces",
            meanRank: "rang moyen",
            wins: "victoires",
            leakage: "Contrôle de fuite",
            leakageText: (v: number) =>
              `La validation K-fold naïve surestime le R² de ${v >= 0 ? "+" : ""}${v.toFixed(3)} (médiane sur les modèles) par rapport à la validation groupée par placette.`,
            intervals: "Intervalles de prédiction",
            method: "Méthode",
            coverage: "Couverture",
            width: "Largeur moy.",
            calibrated: "Calibré",
            importance: "Importance par permutation",
            rmseChart: "Variation du RMSE vs OLS (%) — à droite = mieux",
            scatter: "R² vs erreur p95 — en bas à droite = mieux",
            best: "Meilleur modèle",
            baseline: "Référence",
            dataset: (n: number, p: number) => `${n} intervalles · ${p} placettes`,
            generated: "Généré",
            kpiBest: "Meilleur modèle",
            kpiGain: "Gain RMSE vs OLS",
            kpiR2: "R² (placettes non vues)",
            kpiCoverage: "Couverture IP 90 %",
            kpiLeak: "Inflation R² (fuite)",
            notSignificant: "non significatif",
            significant: "significatif (p < 0,05)",
            loading: "Chargement du rapport…",
            noOverlap: "aucune placette partagée entre entraînement et test",
          }
        : {
            title: "Model Benchmarks",
            description:
              "Leakage-free, plot-grouped cross-validation of 17 candidate algorithms against the production RGR equation.",
            noReport: "No benchmark report yet. Run one below or execute scripts/benchmark_models.py --all.",
            run: "Run benchmark",
            runFast: "Fast models (7)",
            runFull: "All models (17)",
            running: "Running…",
            refresh: "Refresh",
            download: "JSON",
            group: "Model group",
            cv: "Cross-validation",
            cvGroup: "Plot-grouped (recommended)",
            cvTemporal: "Temporal (train past → test future)",
            cvKfold: "Naive K-fold (leaky)",
            features: "Features",
            featBase: "Equation (7)",
            featExt: "Extended (11)",
            leaderboard: "Leaderboard",
            rank: "Rank",
            model: "Model",
            family: "Family",
            vsOls: "vs OLS",
            pValue: "p (paired t)",
            signs: "Wrong signs",
            summary: "Summary across groups",
            meanRank: "mean rank",
            wins: "wins",
            leakage: "Leakage check",
            leakageText: (v: number) =>
              `Naive K-fold inflates R² by ${v >= 0 ? "+" : ""}${v.toFixed(3)} (median across models) versus plot-grouped CV.`,
            intervals: "Prediction intervals",
            method: "Method",
            coverage: "Coverage",
            width: "Mean width",
            calibrated: "Calibrated",
            importance: "Permutation importance",
            rmseChart: "RMSE change vs OLS (%) — right is better",
            scatter: "R² vs p95 absolute error — bottom-right is better",
            best: "Best model",
            baseline: "Baseline",
            dataset: (n: number, p: number) => `${n} intervals · ${p} plots`,
            generated: "Generated",
            kpiBest: "Best model",
            kpiGain: "RMSE gain vs OLS",
            kpiR2: "R² on unseen plots",
            kpiCoverage: "90% interval coverage",
            kpiLeak: "R² inflation (leakage)",
            notSignificant: "not significant",
            significant: "significant (p < 0.05)",
            loading: "Loading report…",
            noOverlap: "no plot shared between train and test folds",
          },
    [language],
  );

  const loadReport = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/benchmarks`, { cache: "no-store" });
      if (res.status === 404) {
        setReport(null);
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = await res.json();
      setReport(body.report);
      setReportFile(body.report_file);
      const keys = Object.keys(body.report.groups ?? {});
      setGroup((g) => (g && keys.includes(g) ? g : keys[0] ?? ""));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadReport();
    fetch(`${API_BASE}/api/benchmarks/models`)
      .then((r) => (r.ok ? r.json() : null))
      .then((b) => b && setRegistry(b.models))
      .catch(() => undefined);
  }, [loadReport]);

  // poll a running job
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const id = window.setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/models/training-status/${job.job_id}`);
        if (!res.ok) return;
        const next: Job = await res.json();
        setJob(next);
        if (next.status !== "running") {
          window.clearInterval(id);
          await loadReport();
        }
      } catch {
        /* keep polling */
      }
    }, 1500);
    return () => window.clearInterval(id);
  }, [job, loadReport]);

  const startRun = async () => {
    setError(null);
    const available = new Set(registry.filter((m) => m.available).map((m) => m.name));
    const payload: Record<string, unknown> = { cv_strategy: cvStrategy, feature_set: featureSet };
    if (fastRun) payload.models = FAST_MODELS.filter((m) => available.size === 0 || available.has(m));
    try {
      const res = await fetch(`${API_BASE}/api/benchmarks/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = await res.json();
      setJob({ job_id: body.job_id, status: "running", progress: { total: 0, done: 0, failed: 0, current: null }, errors: {} });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const downloadJson = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = reportFile || "benchmark.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc((v) => !v);
    else {
      setSortKey(key);
      setSortAsc(key === "r2" || key === "vsOls" ? false : true);
    }
  };

  const section = report && group ? report.groups[group] : null;
  const ranking = section?.ranking ?? [];
  const rows = ranking.map((name) => {
    const e = section!.models[name];
    const vb = e.vs_baseline ?? undefined;
    return {
      model: name,
      family: e.family,
      rank: e.rank,
      rmse: e.metrics_mean.rmse ?? 0,
      rmseErr: e.metrics_std.rmse ?? 0,
      r2: e.metrics_mean.r2 ?? 0,
      mae: e.metrics_mean.mae ?? 0,
      p95: e.metrics_mean.p95_ae ?? 0,
      bias: e.metrics_mean.bias ?? 0,
      vsOls: vb?.rmse_improvement_pct ?? null,
      p: vb?.paired_t_on_fold_rmse?.p_value ?? null,
      signs: e.sign_check ?? null,
      description: e.description,
    };
  });
  const sortedRows = [...rows].sort((a, b) => {
    const av = a[sortKey] ?? Number.NEGATIVE_INFINITY;
    const bv = b[sortKey] ?? Number.NEGATIVE_INFINITY;
    return sortAsc ? Number(av) - Number(bv) : Number(bv) - Number(av);
  });
  const deltaRows = rows.map((r) => ({ model: r.model, delta: r.vsOls ?? 0 }));
  const bestEntry = section?.best_model ? section.models[section.best_model] : null;
  const bestVs = bestEntry?.vs_baseline ?? undefined;
  const bestP = bestVs?.paired_t_on_fold_rmse?.p_value ?? null;
  const intervalRows: IntervalResult[] = Object.entries(section?.intervals ?? {})
    .filter(([, v]) => v && typeof v === "object" && "coverage" in v)
    .map(([k, v]) => ({ ...(v as IntervalResult), method: k }));
  const coverageBest = intervalRows.find((r) => r.method === "split_conformal_best")?.coverage;
  const leakMedian = section?.leakage_check?.median_r2_inflation ?? section?.leakage_check?.mean_r2_inflation ?? null;

  const SortHead = ({ k, label, className = "" }: { k: SortKey; label: string; className?: string }) => (
    <TableHead className={`cursor-pointer select-none ${className}`} onClick={() => toggleSort(k)}>
      <span className="inline-flex items-center gap-1">
        {label}
        {sortKey === k && (sortAsc ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />)}
      </span>
    </TableHead>
  );

  const progressPct =
    job && job.progress.total > 0 ? Math.min(100, Math.round(((job.progress.done + job.progress.failed) / job.progress.total) * 100)) : 0;

  return (
    <div className="space-y-6">
      {/* ---------------------------------------------------------------- header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="flex items-center gap-2 text-2xl font-semibold">
            <FlaskConical className="h-6 w-6 text-green-700 dark:text-green-400" />
            {t.title}
          </h2>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{t.description}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="outline" size="sm" onClick={loadReport} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            <span className="ml-2">{t.refresh}</span>
          </Button>
          <Button variant="outline" size="sm" onClick={downloadJson} disabled={!report}>
            <Download className="h-4 w-4" />
            <span className="ml-2">{t.download}</span>
          </Button>
        </div>
      </div>

      {/* ---------------------------------------------------------------- run controls */}
      <Card>
        <CardContent className="flex flex-wrap items-end gap-4 pt-6">
          <div className="space-y-1">
            <div className="text-xs font-medium text-muted-foreground">{t.model}s</div>
            <div className="flex gap-1 rounded-lg bg-muted p-1">
              <Button size="sm" variant={fastRun ? "default" : "ghost"} onClick={() => setFastRun(true)}>
                {t.runFast}
              </Button>
              <Button size="sm" variant={!fastRun ? "default" : "ghost"} onClick={() => setFastRun(false)}>
                {t.runFull}
              </Button>
            </div>
          </div>
          <div className="space-y-1">
            <div className="text-xs font-medium text-muted-foreground">{t.cv}</div>
            <Select value={cvStrategy} onValueChange={(v) => setCvStrategy(v as typeof cvStrategy)}>
              <SelectTrigger className="w-64">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="group">{t.cvGroup}</SelectItem>
                <SelectItem value="temporal">{t.cvTemporal}</SelectItem>
                <SelectItem value="kfold">{t.cvKfold}</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1">
            <div className="text-xs font-medium text-muted-foreground">{t.features}</div>
            <Select value={featureSet} onValueChange={(v) => setFeatureSet(v as typeof featureSet)}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="base">{t.featBase}</SelectItem>
                <SelectItem value="extended">{t.featExt}</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <Button onClick={startRun} disabled={job?.status === "running"} className="ml-auto">
            <Play className="h-4 w-4" />
            <span className="ml-2">{job?.status === "running" ? t.running : t.run}</span>
          </Button>
          {job && job.status === "running" && (
            <div className="basis-full space-y-1">
              <Progress value={progressPct} />
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Activity className="h-3 w-3 animate-pulse" />
                {job.progress.current ?? t.running}
                {job.progress.total ? ` · ${job.progress.done + job.progress.failed}/${job.progress.total}` : ""}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {job && job.status === "failed" && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{JSON.stringify(job.errors)}</AlertDescription>
        </Alert>
      )}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {loading && !report && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <RefreshCw className="h-4 w-4 animate-spin" /> {t.loading}
        </div>
      )}

      {!report && !loading && !error && (
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>{t.noReport}</AlertDescription>
        </Alert>
      )}

      {report && (
        <>
          {/* ------------------------------------------------------------ report meta + group picker */}
          <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
            <span>
              {t.generated}: {new Date(report.generated_at).toLocaleString()} · {reportFile}
            </span>
            <Badge variant="secondary">
              CV: {report.config.cv_strategy} × {report.config.n_splits}
            </Badge>
            <Badge variant="secondary">features: {report.config.feature_set}</Badge>
            <Badge variant="secondary">{report.elapsed_seconds}s</Badge>
            <div className="ml-auto flex items-center gap-2">
              <span>{t.group}</span>
              <Select value={group} onValueChange={setGroup}>
                <SelectTrigger className="w-52">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.keys(report.groups).map((k) => (
                    <SelectItem key={k} value={k}>
                      {k} {report.groups[k].best_model ? `· ${report.groups[k].best_model}` : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {section?.error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{section.error}</AlertDescription>
            </Alert>
          )}

          {section && !section.error && bestEntry && (
            <>
              {/* ---------------------------------------------------------- KPI tiles */}
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                <Card className="border-green-200 bg-green-50/60 dark:border-green-900 dark:bg-green-950/30">
                  <CardContent className="pt-5">
                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                      <Trophy className="h-3 w-3" /> {t.kpiBest}
                    </div>
                    <div className="mt-1 truncate text-xl font-semibold">{section.best_model}</div>
                    <div className="text-xs text-muted-foreground">{bestEntry.family}</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-5">
                    <div className="text-xs text-muted-foreground">{t.kpiGain}</div>
                    <div className="mt-1 text-xl font-semibold tabular-nums">
                      {section.best_model === report.summary.baseline ? "—" : pct(bestVs?.rmse_improvement_pct)}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {section.best_model === report.summary.baseline
                        ? t.baseline
                        : bestP !== null && bestP < 0.05
                          ? t.significant
                          : `${t.notSignificant} (p ${pval(bestP)})`}
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-5">
                    <div className="text-xs text-muted-foreground">{t.kpiR2}</div>
                    <div className="mt-1 text-xl font-semibold tabular-nums">{fmt(bestEntry.metrics_mean.r2, 3)}</div>
                    <div className="text-xs text-muted-foreground">OLS {fmt(section.models.ols?.metrics_mean.r2, 3)}</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-5">
                    <div className="text-xs text-muted-foreground">{t.kpiCoverage}</div>
                    <div className="mt-1 text-xl font-semibold tabular-nums">
                      {coverageBest === undefined ? "–" : `${(coverageBest * 100).toFixed(1)}%`}
                    </div>
                    <div className="text-xs text-muted-foreground">split conformal · target 90%</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-5">
                    <div className="text-xs text-muted-foreground">{t.kpiLeak}</div>
                    <div className="mt-1 text-xl font-semibold tabular-nums">
                      {leakMedian === null ? "–" : `${leakMedian >= 0 ? "+" : ""}${leakMedian.toFixed(3)}`}
                    </div>
                    <div className="text-xs text-muted-foreground">naive K-fold − grouped</div>
                  </CardContent>
                </Card>
              </div>

              {/* ---------------------------------------------------------- leaderboard */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    {t.leaderboard} — {group}
                  </CardTitle>
                  <CardDescription>
                    {t.dataset(section.dataset.n_samples, section.dataset.n_plots)}
                    {section.dataset.year_range ? ` · ${section.dataset.year_range[0]}–${section.dataset.year_range[1]}` : ""}
                    {section.cv ? ` · ${section.cv.strategy} CV × ${section.cv.n_splits}` : ""}
                    {section.cv && section.cv.plot_overlap_between_train_and_test === 0 ? ` · ${t.noOverlap}` : ""}
                  </CardDescription>
                </CardHeader>
                <CardContent className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <SortHead k="rank" label={t.rank} />
                        <TableHead>{t.model}</TableHead>
                        <TableHead>{t.family}</TableHead>
                        <SortHead k="rmse" label="RMSE" className="text-right" />
                        <SortHead k="r2" label="R²" className="text-right" />
                        <SortHead k="mae" label="MAE" className="text-right" />
                        <SortHead k="p95" label="p95 |AE|" className="text-right" />
                        <SortHead k="bias" label="Bias" className="text-right" />
                        <SortHead k="vsOls" label={t.vsOls} className="text-right" />
                        <TableHead className="text-right">{t.pValue}</TableHead>
                        <TableHead className="text-right">{t.signs}</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sortedRows.map((r) => {
                        const better = (r.vsOls ?? 0) > 0;
                        return (
                          <TableRow
                            key={r.model}
                            title={r.description}
                            className={r.model === section.best_model ? "bg-green-50 dark:bg-green-950/30" : ""}
                          >
                            <TableCell>{r.rank}</TableCell>
                            <TableCell className="font-medium">
                              {r.model}
                              {r.model === report.summary.baseline && (
                                <Badge variant="outline" className="ml-2">
                                  {t.baseline}
                                </Badge>
                              )}
                            </TableCell>
                            <TableCell>
                              <span className="inline-flex items-center gap-1">
                                <span className="inline-block h-2 w-2 rounded-full" style={{ background: FAMILY_COLOR[r.family] }} />
                                {r.family}
                              </span>
                            </TableCell>
                            <TableCell className="text-right tabular-nums">{fmt(r.rmse)}</TableCell>
                            <TableCell className="text-right tabular-nums">{fmt(r.r2, 3)}</TableCell>
                            <TableCell className="text-right tabular-nums">{fmt(r.mae)}</TableCell>
                            <TableCell className="text-right tabular-nums">{fmt(r.p95)}</TableCell>
                            <TableCell className="text-right tabular-nums">{fmt(r.bias)}</TableCell>
                            <TableCell
                              className={`text-right tabular-nums ${
                                r.vsOls === null ? "" : better ? "text-green-700 dark:text-green-400" : "text-red-600 dark:text-red-400"
                              }`}
                            >
                              {r.vsOls === null ? "–" : pct(r.vsOls)}
                            </TableCell>
                            <TableCell className="text-right tabular-nums">
                              {pval(r.p)}
                              {r.p !== null && r.p < 0.05 && <span title="significant at 5%"> *</span>}
                            </TableCell>
                            <TableCell className="text-right">
                              {r.signs ? (
                                r.signs.wrong_sign_count === 0 ? (
                                  <ShieldCheck className="ml-auto h-4 w-4 text-green-600" />
                                ) : (
                                  <span title={r.signs.wrong_signs.join(", ")}>{r.signs.wrong_sign_count}</span>
                                )
                              ) : (
                                "–"
                              )}
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              {/* ---------------------------------------------------------- charts */}
              <div className="grid gap-6 lg:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">{t.rmseChart}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={Math.max(240, rows.length * 26)}>
                      <BarChart data={deltaRows} layout="vertical" margin={{ left: 24, right: 32 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis tick={AXIS_TICK}
                          type="number"
                          domain={[(min: number) => Math.min(-1, Math.floor(min)), (max: number) => Math.max(1, Math.ceil(max))]}
                          tickFormatter={(v) => `${Number(v) > 0 ? "+" : ""}${Number(v).toFixed(0)}%`}
                        />
                        <YAxis type="category" dataKey="model" width={110} interval={0} tick={AXIS_TICK} />
                        <Tooltip
                          formatter={(v) => [`${Number(v) > 0 ? "+" : ""}${Number(v).toFixed(2)}%`, t.vsOls]}
                          labelFormatter={(label) => {
                            const r = rows.find((x) => x.model === label);
                            return r ? `${label} · RMSE ${fmt(r.rmse)} ± ${fmt(r.rmseErr)}` : String(label);
                          }}
                        />
                        <ReferenceLine x={0} stroke="#2563eb" strokeDasharray="4 4" label={{ value: "OLS", position: "top", fontSize: 11 }} />
                        <Bar dataKey="delta" isAnimationActive={false}>
                          {deltaRows.map((r) => (
                            <Cell key={r.model} fill={r.delta >= 0 ? "#059669" : "#dc2626"} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">{t.scatter}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 26)}>
                      <ScatterChart margin={{ left: 12, right: 24, bottom: 12 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis tick={AXIS_TICK}
                          type="number"
                          dataKey="r2"
                          name="R²"
                          domain={["auto", "auto"]}
                          tickFormatter={(v) => Number(v).toFixed(2)}
                          label={{ value: "R²", position: "insideBottom", offset: -6 }}
                        />
                        <YAxis tick={AXIS_TICK}
                          type="number"
                          dataKey="p95"
                          name="p95 |AE|"
                          domain={["auto", "auto"]}
                          tickFormatter={(v) => Number(v).toFixed(4)}
                        />
                        <Tooltip
                          cursor={{ strokeDasharray: "3 3" }}
                          content={({ payload }) =>
                            payload && payload.length ? (
                              <div className="rounded border bg-background p-2 text-xs shadow">
                                <div className="font-medium">{(payload[0].payload as { model: string }).model}</div>
                                <div>R² {Number((payload[0].payload as { r2: number }).r2).toFixed(3)}</div>
                                <div>p95 {Number((payload[0].payload as { p95: number }).p95).toFixed(5)}</div>
                              </div>
                            ) : null
                          }
                        />
                        <Legend />
                        {Object.keys(FAMILY_COLOR).map((fam) => (
                          <Scatter
                            key={fam}
                            name={fam}
                            data={rows.filter((r) => r.family === fam)}
                            fill={FAMILY_COLOR[fam]}
                            isAnimationActive={false}
                          />
                        ))}
                      </ScatterChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>

              {/* ---------------------------------------------------------- diagnostics */}
              <div className="grid gap-6 lg:grid-cols-3">
                {section.leakage_check && leakMedian !== null && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">{t.leakage}</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <p>{t.leakageText(leakMedian)}</p>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>{t.model}</TableHead>
                            <TableHead className="text-right">R² grouped</TableHead>
                            <TableHead className="text-right">R² naive</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {Object.entries(section.leakage_check.per_model)
                            .sort((a, b) => b[1].r2_inflation - a[1].r2_inflation)
                            .slice(0, 8)
                            .map(([name, v]) => (
                              <TableRow key={name}>
                                <TableCell>{name}</TableCell>
                                <TableCell className="text-right tabular-nums">{fmt(v.r2_grouped, 3)}</TableCell>
                                <TableCell className="text-right tabular-nums">{fmt(v.r2_naive_kfold, 3)}</TableCell>
                              </TableRow>
                            ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                )}

                {intervalRows.length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">
                        {t.intervals} ({Math.round((1 - report.config.alpha) * 100)}%)
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>{t.method}</TableHead>
                            <TableHead className="text-right">{t.coverage}</TableHead>
                            <TableHead className="text-right">{t.width}</TableHead>
                            <TableHead className="text-right" title={t.calibrated}>
                              OK
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {intervalRows.map((r) => (
                            <TableRow key={r.method}>
                              <TableCell>
                                <div>{r.method}</div>
                                <div className="text-xs text-muted-foreground">{r.model}</div>
                              </TableCell>
                              <TableCell className="text-right tabular-nums">{fmt(r.coverage, 3)}</TableCell>
                              <TableCell className="text-right tabular-nums">{fmt(r.mean_width)}</TableCell>
                              <TableCell className="text-right">{r.calibrated ? "✓" : "✗"}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                )}

                {section.importance?.features && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">
                        {t.importance} · {section.importance.model}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={section.importance.features} layout="vertical" margin={{ left: 12, right: 16 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis tick={AXIS_TICK} type="number" tickFormatter={(v) => Number(v).toFixed(4)} />
                          <YAxis type="category" dataKey="feature" width={70} interval={0} tick={AXIS_TICK} />
                          <Tooltip formatter={(v) => Number(v).toFixed(5)} />
                          <Bar dataKey="importance_mean" fill="#7c3aed" isAnimationActive={false}>
                            <ErrorBar dataKey="importance_std" width={4} strokeWidth={1} stroke="#334155" direction="x" />
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                )}
              </div>

              {/* ---------------------------------------------------------- summary across groups */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">{t.summary}</CardTitle>
                  <CardDescription>
                    {t.best}: <strong>{report.summary.overall_best}</strong> · {t.baseline}: {report.summary.baseline}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(report.summary.mean_rank_by_model).map(([name, rank]) => (
                      <Badge
                        key={name}
                        variant="outline"
                        style={{ borderColor: FAMILY_COLOR[section.models[name]?.family ?? ""] ?? undefined }}
                      >
                        {name} · {t.meanRank} {rank.toFixed(2)} · {report.summary.wins_by_model[name] ?? 0} {t.wins}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </>
      )}
    </div>
  );
}
