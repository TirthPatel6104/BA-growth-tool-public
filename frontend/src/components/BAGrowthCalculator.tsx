import { useEffect, useRef, useState } from "react";
import { Button } from "./ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Alert, AlertDescription } from "./ui/alert";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import {
  AlertCircle,
  Loader,
  TrendingUp,
  Download,
  RefreshCw,
} from "lucide-react";
import type { Language } from "../App";

interface BAgrowthResult {
  species_group: string;
  mean_BAgrowth: number;
  mean_RGR: number;
  std_BAgrowth: number;
  ci_lower: number;
  ci_upper: number;
  n_plots: number;
  mean_BA_current: number;
  per_plot_results: Array<{
    PlotKey: string;
    BA_current: number;
    BAgrowth_predicted: number;
    RGR_predicted: number;
  }>;
  yearly_projection?: Array<{
    year: number;
    BA_predicted: number;
    BAgrowth_predicted: number;
    RGR: number;
  }>;
  model_info: {
    coefficient_id?: number | null;
    intercept: number;
    coefficients: Record<string, number>;
    cv_metrics: {
      rmse_mean: number | null;
      rmse_std: number | null;
      r2_mean: number | null;
      r2_std: number | null;
    };
    n_training_rows: number;
    fitted_at: string;
  };
  pipeline_metrics: {
    training_skipped: boolean;
    cache_reason: string;
    elapsed_seconds: number;
  };
}

interface SummaryStats {
  species_group: string;
  plots_analyzed: number;
  measurements: number;
  excluded_measurements_below_dq_threshold?: number;
  variables: {
    BA?: { mean: number; min: number; max: number } | null;
    BA2?: { mean: number; min: number; max: number } | null;
    AGSfrac?: { mean: number; min: number; max: number } | null;
    TCR?: { mean: number; min: number; max: number } | null;
    DQ?: { mean: number; min: number; max: number } | null;
    q?: { mean: number; min: number; max: number } | null;
    BAgrowth?: { mean: number; min: number; max: number } | null;
  };
}

interface FormulaCalculation {
  species_group: string;
  formula: string;
  formula_explanation: string;
  inputs: {
    mean_BA: number;
    mean_BA2: number;
    mean_AGSfrac: number;
    mean_TCR: number;
    mean_DQ: number;
    mean_q: number;
  };
  coefficients: {
    intercept: number;
    BA: number;
    BA2: number;
    AGSfrac: number;
    TCR: number;
    DQ: number;
    q: number;
    Cut: number;
  };
  scenarios: {
    without_cut: {
      cut_value: number;
      label: string;
      description: string;
      calculated_bagrowth: number;
    };
    with_cut: {
      cut_value: number;
      label: string;
      description: string;
      calculated_bagrowth: number;
    };
  };
  measurements_used: number;
  plots_used: number;
  excluded_measurements_below_dq_threshold?: number;
}

interface SpeciesListResponse {
  species_groups: string[];
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const RESPONSE_CACHE_VERSION = "v4_cut_positive_fallback";
const RESPONSE_CACHE_TTL_MS = 5 * 60 * 1000;
const BAGROWTH_UI_STATE_KEY = "bagt_bagrowth_ui_state_v1";

interface CachedResponse {
  expiresAt: number;
  data: unknown;
}

interface PersistedBAGrowthUIState {
  species?: string;
  currentYear?: string;
  result?: BAgrowthResult | null;
  summaryStats?: SummaryStats | null;
  formulaCalc?: FormulaCalculation | null;
}

interface BAGrowthCalculatorProps {
  language: Language;
  selectedModelGroup?: string;
  onModelGroupChange?: (modelGroup: string) => void;
}

export function BAGrowthCalculator({
  language,
  selectedModelGroup,
  onModelGroupChange,
}: BAGrowthCalculatorProps) {
  const responseCacheRef = useRef<Map<string, CachedResponse>>(new Map());
  const [species, setSpecies] = useState<string>(
    selectedModelGroup || "HardMaple",
  );
  const [speciesList, setSpeciesList] = useState<string[]>([]);
  const [currentYear, setCurrentYear] = useState<string>("2026");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BAgrowthResult | null>(null);
  const [summaryStats, setSummaryStats] = useState<SummaryStats | null>(null);
  const [formulaCalc, setFormulaCalc] = useState<FormulaCalculation | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [uiStateHydrated, setUiStateHydrated] = useState(false);

  const getCached = <T,>(key: string): T | null => {
    const entry = responseCacheRef.current.get(key);
    if (!entry) return null;
    if (Date.now() >= entry.expiresAt) {
      responseCacheRef.current.delete(key);
      return null;
    }
    return entry.data as T;
  };

  const setCached = (key: string, data: unknown) => {
    responseCacheRef.current.set(key, {
      data,
      expiresAt: Date.now() + RESPONSE_CACHE_TTL_MS,
    });
  };

  const setSpeciesAndSync = (nextSpecies: string) => {
    setSpecies(nextSpecies);
    onModelGroupChange?.(nextSpecies);
  };

  const versionedCacheKey = (key: string) => `${RESPONSE_CACHE_VERSION}:${key}`;

  useEffect(() => {
    if (!selectedModelGroup) return;
    if (selectedModelGroup !== species) {
      setSpecies(selectedModelGroup);
    }
  }, [selectedModelGroup, species]);

  useEffect(() => {
    if (typeof window === "undefined") {
      setUiStateHydrated(true);
      return;
    }

    try {
      const raw = window.localStorage.getItem(BAGROWTH_UI_STATE_KEY);
      if (!raw) {
        setUiStateHydrated(true);
        return;
      }

      const parsed = JSON.parse(raw) as PersistedBAGrowthUIState;
      if (parsed.species && typeof parsed.species === "string") {
        setSpecies(parsed.species);
        onModelGroupChange?.(parsed.species);
      }
      if (parsed.currentYear && typeof parsed.currentYear === "string") {
        setCurrentYear(parsed.currentYear);
      }
      if (parsed.result !== undefined) {
        setResult(parsed.result ?? null);
      }
      if (parsed.summaryStats !== undefined) {
        setSummaryStats(parsed.summaryStats ?? null);
      }
      if (parsed.formulaCalc !== undefined) {
        setFormulaCalc(parsed.formulaCalc ?? null);
      }
    } catch (storageErr) {
      console.warn("Failed to restore BA growth UI state:", storageErr);
    } finally {
      setUiStateHydrated(true);
    }
  }, [onModelGroupChange]);

  useEffect(() => {
    if (typeof window === "undefined" || !uiStateHydrated) return;

    const payload: PersistedBAGrowthUIState = {
      species,
      currentYear,
      result,
      summaryStats,
      formulaCalc,
    };

    try {
      window.localStorage.setItem(
        BAGROWTH_UI_STATE_KEY,
        JSON.stringify(payload),
      );
    } catch (storageErr) {
      console.warn("Failed to persist BA growth UI state:", storageErr);
    }
  }, [
    uiStateHydrated,
    species,
    currentYear,
    result,
    summaryStats,
    formulaCalc,
  ]);

  const t =
    language === "en"
      ? {
          title: "BA Growth Calculator",
          description:
            "Calculate average BA growth prediction for forest species (database-driven)",
          species: "Select Model Group",
          yearLabel: "Projection Year",
          calculate: "Calculate BA Growth",
          calculating: "Calculating...",
          summary: "Growth Summary",
          ciLabel: "95% Confidence Interval",
          rmse: "Model RMSE (CV)",
          plots: "Plots Analyzed",
          yearlyProjection: "10-Year Projection",
          perPlotResults: "Per-Plot Results",
          datasetStats: "Dataset Statistics",
          meanFormulaTitle: "Formula Using Mean Inputs",
          cutHelp:
            "Showing both scenarios because Cut can be 0 (no harvest) or 1 (harvest/treatment).",
          dqThresholdNote: "Rows with DQ < 9 cm are excluded.",
          year: "Year",
          ba: "BA",
          growth: "BAgrowth",
          rgr: "RGR",
          download: "Download Results",
          currentAvgBA: "Current Avg BA",
          noData: "Run calculation to see results",
          cache: "Cache Status",
          measurements: "Measurements",
          mean: "Mean",
          min: "Min",
          max: "Max",
        }
      : {
          title: "Calculateur de Croissance ST",
          description:
            "Calculer la prédiction moyenne de croissance ST pour les espèces forestières (basée sur la base de données)",
          species: "Sélectionner Groupe de Modèle",
          yearLabel: "Année de Projection",
          calculate: "Calculer Croissance ST",
          calculating: "Calcul en cours...",
          summary: "Résumé de Croissance",
          ciLabel: "Intervalle de Confiance 95%",
          rmse: "RMSE Modèle (CV)",
          plots: "Parcelles Analysées",
          yearlyProjection: "Projection 10 Ans",
          perPlotResults: "Résultats par Parcelle",
          datasetStats: "Statistiques de Dataset",
          meanFormulaTitle: "Formule avec Moyennes des Variables",
          cutHelp:
            "Les deux scénarios sont affichés parce que Cut peut être 0 (sans récolte) ou 1 (récolte/traitement).",
          dqThresholdNote: "Les lignes avec DQ < 9 cm sont exclues.",
          year: "Année",
          ba: "ST",
          growth: "Croissance ST",
          rgr: "TCR",
          download: "Télécharger Résultats",
          currentAvgBA: "ST Moyen Actuel",
          noData: "Exécuter le calcul pour voir les résultats",
          cache: "État du Cache",
          measurements: "Mesures",
          mean: "Moyenne",
          min: "Min",
          max: "Max",
        };

  useEffect(() => {
    const fetchSpecies = async () => {
      try {
        const cacheKey = versionedCacheKey("species-list");
        const cached = getCached<SpeciesListResponse>(cacheKey);
        if (cached) {
          const loadedSpecies = cached.species_groups || [];
          setSpeciesList(loadedSpecies);
          if (loadedSpecies.length > 0) {
            setSpecies((previousSpecies) => {
              const nextSpecies = loadedSpecies.includes(previousSpecies)
                ? previousSpecies
                : loadedSpecies[0];
              onModelGroupChange?.(nextSpecies);
              return nextSpecies;
            });
          }
          return;
        }

        const res = await fetch(`${API_BASE}/api/species-list`);
        if (!res.ok) {
          return;
        }
        const data = (await res.json()) as SpeciesListResponse;
        setCached(cacheKey, data);
        const loadedSpecies = data.species_groups || [];
        setSpeciesList(loadedSpecies);
        if (loadedSpecies.length > 0) {
          setSpecies((previousSpecies) => {
            const nextSpecies = loadedSpecies.includes(previousSpecies)
              ? previousSpecies
              : loadedSpecies[0];
            onModelGroupChange?.(nextSpecies);
            return nextSpecies;
          });
        }
      } catch (fetchErr) {
        console.error("Failed to fetch species list:", fetchErr);
      }
    };

    fetchSpecies();
  }, [onModelGroupChange]);

  useEffect(() => {
    const fetchSummaryAndFormula = async () => {
      try {
        const requestTs = Date.now();
        const [summaryRes, formulaRes] = await Promise.all([
          fetch(
            `${API_BASE}/api/bagrowth/summary-stats?species_group=${encodeURIComponent(species)}&_ts=${requestTs}`,
            { cache: "no-store" },
          ),
          fetch(
            `${API_BASE}/api/bagrowth/formula-calculation?species_group=${encodeURIComponent(species)}&_ts=${requestTs}`,
            { cache: "no-store" },
          ),
        ]);

        if (summaryRes.ok) {
          const data = (await summaryRes.json()) as SummaryStats;
          setSummaryStats(data);
        } else {
          setSummaryStats(null);
        }

        if (formulaRes.ok) {
          const data = (await formulaRes.json()) as FormulaCalculation;
          setFormulaCalc(data);
        } else {
          setFormulaCalc(null);
        }
      } catch (fetchErr) {
        console.error("Failed to fetch summary stats/formula:", fetchErr);
        setSummaryStats(null);
        setFormulaCalc(null);
      }
    };

    if (species) {
      fetchSummaryAndFormula();
    }
  }, [species]);

  const handleCalculate = async () => {
    setLoading(true);
    setError(null);

    try {
      const baseUrl = `${API_BASE}/api/bagrowth/calculate?species_group=${encodeURIComponent(species)}&current_year=${encodeURIComponent(currentYear)}`;
      const latestUrl = `${baseUrl}&refresh=true&_ts=${Date.now()}`;
      const res = await fetch(latestUrl, { cache: "no-store" });

      if (!res.ok) {
        const text = await res.text();
        const lower = text.toLowerCase();
        const isDbConnectionError =
          lower.includes("database connection failed") ||
          lower.includes("login timeout expired");

        if (isDbConnectionError) {
          const fallbackRes = await fetch(`${baseUrl}&_ts=${Date.now()}`, {
            cache: "no-store",
          });
          if (fallbackRes.ok) {
            const data = (await fallbackRes.json()) as BAgrowthResult;
            setResult(data);
            return;
          }
        }

        let message = "Calculation failed";
        try {
          const json = JSON.parse(text);
          message = json.detail || json.message || text || message;
        } catch {
          message = text || message;
        }
        throw new Error(message);
      }

      const data = (await res.json()) as BAgrowthResult;
      setResult(data);
    } catch (calculationErr) {
      setError(
        calculationErr instanceof Error
          ? calculationErr.message
          : "Unknown error",
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!result) return;
    const json = JSON.stringify(result, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const year =
      result.yearly_projection?.[0]?.year || new Date().getFullYear();
    a.download = `ba_growth_${species}_${year}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            {t.title}
          </CardTitle>
          <CardDescription>{t.description}</CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="flex gap-3 items-end">
            <div className="flex-1">
              <label className="text-sm font-medium">{t.species}</label>
              <Select
                value={species}
                onValueChange={setSpeciesAndSync}
                disabled={loading}
              >
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {speciesList.map((speciesName) => (
                    <SelectItem key={speciesName} value={speciesName}>
                      {speciesName}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="w-32">
              <label className="text-sm font-medium">{t.yearLabel}</label>
              <input
                type="number"
                value={currentYear}
                onChange={(event) => setCurrentYear(event.target.value)}
                disabled={loading}
                className="mt-1 w-full px-3 py-2 border border-gray-300 rounded-md dark:bg-slate-800 dark:border-slate-600"
              />
            </div>

            <Button onClick={handleCalculate} disabled={loading} size="sm">
              {loading ? (
                <>
                  <Loader className="h-4 w-4 mr-2 animate-spin" />
                  {t.calculating}
                </>
              ) : (
                <>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  {t.calculate}
                </>
              )}
            </Button>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {result && (
            <>
              <div className="space-y-4 border-t pt-6">
                <h3 className="font-semibold">{t.summary}</h3>

                <div className="grid grid-cols-4 gap-3">
                  <Card className="p-3">
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      {t.ciLabel}
                    </p>
                    <p className="text-sm font-mono">
                      [{result.ci_lower.toFixed(4)},{" "}
                      {result.ci_upper.toFixed(4)}]
                    </p>
                  </Card>

                  <Card className="p-3">
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      {t.plots}
                    </p>
                    <p className="text-xl font-bold">{result.n_plots}</p>
                  </Card>

                  <Card className="p-3">
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      {t.currentAvgBA}
                    </p>
                    <p className="text-lg font-bold">
                      {result.mean_BA_current.toFixed(2)}
                    </p>
                  </Card>

                  <Card className="p-3">
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      {t.rmse}
                    </p>
                    <p className="text-sm font-mono">
                      {result.model_info.cv_metrics.rmse_mean != null
                        ? result.model_info.cv_metrics.rmse_mean.toFixed(4)
                        : "N/A"}
                    </p>
                  </Card>
                </div>

                <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                  <p>
                    Active coefficient ID:{" "}
                    <span className="font-mono font-semibold text-gray-800 dark:text-gray-200">
                      {result.model_info.coefficient_id ?? "N/A"}
                    </span>
                  </p>
                  <p>
                    Trained at:{" "}
                    <span className="font-mono font-semibold text-gray-800 dark:text-gray-200">
                      {result.model_info.fitted_at
                        ? new Date(result.model_info.fitted_at).toLocaleString(
                            language === "en" ? "en-US" : "fr-FR",
                          )
                        : "N/A"}
                    </span>
                  </p>
                </div>

                {formulaCalc && (
                  <div className="border-t pt-6 space-y-4">
                    <div>
                      <h3 className="font-semibold mb-1">
                        {t.meanFormulaTitle}
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {formulaCalc.formula}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {t.cutHelp}
                      </p>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <Card className="p-4 border-blue-200 dark:border-blue-900 bg-blue-50/70 dark:bg-blue-950/20">
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          Mean BA Growth (Without Cut)
                        </p>
                        <p className="text-2xl font-bold text-blue-600 mt-2">
                          {formulaCalc.scenarios.without_cut.calculated_bagrowth.toFixed(
                            6,
                          )}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">m²/ha/year</p>
                      </Card>

                      <Card className="p-4 border-amber-200 dark:border-amber-900 bg-amber-50/70 dark:bg-amber-950/20">
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          Mean BA Growth (With Cut)
                        </p>
                        <p className="text-2xl font-bold text-amber-600 mt-2">
                          {formulaCalc.scenarios.with_cut.calculated_bagrowth.toFixed(
                            6,
                          )}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">m²/ha/year</p>
                      </Card>
                    </div>
                  </div>
                )}

                {summaryStats && (
                  <div className="border-t pt-6">
                    <h3 className="font-semibold mb-3">{t.datasetStats}</h3>
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      <Card className="p-3 bg-blue-50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-900">
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          {t.plots}
                        </p>
                        <p className="text-xl font-bold text-blue-600">
                          {summaryStats.plots_analyzed}
                        </p>
                      </Card>
                      <Card className="p-3 bg-green-50 dark:bg-green-950/20 border-green-200 dark:border-green-900">
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          {t.measurements}
                        </p>
                        <p className="text-xl font-bold text-green-600">
                          {summaryStats.measurements}
                        </p>
                      </Card>
                    </div>

                    {(summaryStats.excluded_measurements_below_dq_threshold ??
                      0) > 0 && (
                      <Alert className="mb-4">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>
                          {t.dqThresholdNote} Excluded:{" "}
                          {
                            summaryStats.excluded_measurements_below_dq_threshold
                          }
                        </AlertDescription>
                      </Alert>
                    )}

                    <div className="rounded-lg border overflow-hidden">
                      <Table className="text-sm">
                        <TableHeader className="bg-gray-50 dark:bg-slate-800/50">
                          <TableRow>
                            <TableHead>Variable</TableHead>
                            <TableHead className="text-right">
                              {t.mean}
                            </TableHead>
                            <TableHead className="text-right">
                              {t.min}
                            </TableHead>
                            <TableHead className="text-right">
                              {t.max}
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {summaryStats.variables.BA && (
                            <TableRow className="hover:bg-gray-50 dark:hover:bg-slate-800/40">
                              <TableCell className="font-medium">
                                BA (m²/ha)
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.BA.mean.toFixed(2)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.BA.min.toFixed(2)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.BA.max.toFixed(2)}
                              </TableCell>
                            </TableRow>
                          )}
                          {summaryStats.variables.BA2 && (
                            <TableRow className="hover:bg-gray-50 dark:hover:bg-slate-800/40">
                              <TableCell className="font-medium">
                                BA² ((m²/ha)²)
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.BA2.mean.toFixed(2)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.BA2.min.toFixed(2)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.BA2.max.toFixed(2)}
                              </TableCell>
                            </TableRow>
                          )}
                          {summaryStats.variables.AGSfrac && (
                            <TableRow className="hover:bg-gray-50 dark:hover:bg-slate-800/40">
                              <TableCell className="font-medium">
                                AGSfrac
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.AGSfrac.mean.toFixed(4)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.AGSfrac.min.toFixed(4)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.AGSfrac.max.toFixed(4)}
                              </TableCell>
                            </TableRow>
                          )}
                          {summaryStats.variables.TCR && (
                            <TableRow className="hover:bg-gray-50 dark:hover:bg-slate-800/40">
                              <TableCell className="font-medium">TCR</TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.TCR.mean.toFixed(2)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.TCR.min.toFixed(0)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.TCR.max.toFixed(0)}
                              </TableCell>
                            </TableRow>
                          )}
                          {summaryStats.variables.DQ && (
                            <TableRow className="hover:bg-gray-50 dark:hover:bg-slate-800/40">
                              <TableCell className="font-medium">
                                DQ (cm)
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.DQ.mean.toFixed(2)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.DQ.min.toFixed(2)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.DQ.max.toFixed(2)}
                              </TableCell>
                            </TableRow>
                          )}
                          {summaryStats.variables.q && (
                            <TableRow className="hover:bg-gray-50 dark:hover:bg-slate-800/40">
                              <TableCell className="font-medium">
                                q (DBH ratio)
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.q.mean.toFixed(2)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.q.min.toFixed(2)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {summaryStats.variables.q.max.toFixed(2)}
                              </TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                )}

                <Tabs defaultValue="yearly" className="pt-6 border-t">
                  <TabsList>
                    <TabsTrigger value="yearly">
                      {t.yearlyProjection}
                    </TabsTrigger>
                    <TabsTrigger value="plots">{t.perPlotResults}</TabsTrigger>
                  </TabsList>

                  <TabsContent value="yearly" className="space-y-3">
                    {result.yearly_projection &&
                    result.yearly_projection.length > 0 ? (
                      <div className="rounded-lg border overflow-hidden">
                        <Table>
                          <TableHeader className="bg-gray-50 dark:bg-slate-800/50">
                            <TableRow>
                              <TableHead>{t.year}</TableHead>
                              <TableHead className="text-right">
                                {t.ba} (m²/ha)
                              </TableHead>
                              <TableHead className="text-right">
                                {t.growth}
                              </TableHead>
                              <TableHead className="text-right">
                                {t.rgr}
                              </TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {result.yearly_projection.map((projection) => (
                              <TableRow
                                key={projection.year}
                                className="hover:bg-gray-50 dark:hover:bg-slate-800/40"
                              >
                                <TableCell className="font-medium">
                                  {projection.year}
                                </TableCell>
                                <TableCell className="text-right font-mono">
                                  {projection.BA_predicted.toFixed(4)}
                                </TableCell>
                                <TableCell className="text-right font-mono text-green-600">
                                  +{projection.BAgrowth_predicted.toFixed(4)}
                                </TableCell>
                                <TableCell className="text-right font-mono">
                                  {projection.RGR.toFixed(6)}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    ) : (
                      <Alert>
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>
                          10-year projection data unavailable. Currently showing
                          current-year metrics only.
                        </AlertDescription>
                      </Alert>
                    )}
                  </TabsContent>

                  <TabsContent value="plots" className="space-y-3">
                    <div className="text-xs text-gray-500">
                      Showing first 20 of {result.per_plot_results.length} plots
                    </div>
                    <div className="rounded-lg border overflow-hidden max-h-96 overflow-y-auto">
                      <Table>
                        <TableHeader className="bg-gray-50 dark:bg-slate-800/50 sticky top-0">
                          <TableRow>
                            <TableHead>Plot</TableHead>
                            <TableHead className="text-right">
                              {t.ba} Current
                            </TableHead>
                            <TableHead className="text-right">
                              {t.growth} Pred
                            </TableHead>
                            <TableHead className="text-right">
                              {t.rgr} Pred
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {result.per_plot_results.slice(0, 20).map((plot) => (
                            <TableRow
                              key={plot.PlotKey}
                              className="text-xs hover:bg-gray-50 dark:hover:bg-slate-800/40"
                            >
                              <TableCell className="font-mono">
                                {plot.PlotKey}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {plot.BA_current.toFixed(2)}
                              </TableCell>
                              <TableCell className="text-right font-mono text-green-600">
                                {plot.BAgrowth_predicted > 0 ? "+" : ""}
                                {plot.BAgrowth_predicted.toFixed(4)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {plot.RGR_predicted.toFixed(6)}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </TabsContent>
                </Tabs>

                <Button
                  onClick={handleDownload}
                  variant="outline"
                  size="sm"
                  className="w-full"
                >
                  <Download className="h-4 w-4 mr-2" />
                  {t.download}
                </Button>

                {result.pipeline_metrics && (
                  <div className="text-xs text-gray-500 space-y-1">
                    <p>
                      {t.cache}:{" "}
                      {result.pipeline_metrics.training_skipped
                        ? "✓ Cached"
                        : "Fresh"}
                    </p>
                    <p>Time: {result.pipeline_metrics.elapsed_seconds}s</p>
                  </div>
                )}
              </div>
            </>
          )}

          {!result && !loading && (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{t.noData}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
