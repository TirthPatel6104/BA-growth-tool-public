import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Alert, AlertDescription } from "./ui/alert";
import { Badge } from "./ui/badge";
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
  TrendingUp,
  AlertCircle,
  RefreshCw,
  Download,
  Check,
} from "lucide-react";
import { Button } from "./ui/button";
import type { Language } from "../App";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

interface Coefficient {
  name: string;
  value: number;
  label: string;
}

interface ModelInfo {
  n_samples?: number;
  n_features?: number;
  coefficient_id?: number;
  model_type: string;
  training_date?: string;
  r2_score?: number;
  rmse?: number;
  status?: string;
}

interface CoefficientDisplayProps {
  language: Language;
  selectedModelGroup?: string;
  onModelGroupChange?: (modelGroup: string) => void;
}

interface SpeciesListResponse {
  species_groups: string[];
}

export function CoefficientDisplay({
  language,
  selectedModelGroup,
  onModelGroupChange,
}: CoefficientDisplayProps) {
  const [modelGroup, setModelGroup] = useState<string>(
    selectedModelGroup || "HardMaple",
  );
  const [modelGroups, setModelGroups] = useState<string[]>([]);
  const [coefficients, setCoefficients] = useState<Coefficient[]>([]);
  const [intercept, setIntercept] = useState<number>(0);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const [success, setSuccess] = useState(false);

  const translations = {
    en: {
      title: "Regression Model Coefficients",
      description: "Linear regression coefficients for BA growth prediction",
      intercept: "Intercept (β₀)",
      coefficients: "Model Variables",
      noData: "No coefficients loaded. Please train the model first.",
      loading: "Loading coefficients...",
      error: "Error loading coefficients",
      refresh: "Refresh",
      modelGroup: "Model Group",
      download: "Download",
      variable: "Variable",
      coefficient: "Coefficient",
      interpretation: "Effect",
      positive: "Positive effect",
      negative: "Negative effect",
      lastUpdated: "Last updated",
      modelInfo: "Model Performance & Configuration",
      positiveEffect: "↑ Increases growth",
      negativeEffect: "↓ Decreases growth",
      minimalEffect: "Minimal effect",
      modelType: "Model Type",
      rSquared: "R² Score",
      rmse: "RMSE",
      samples: "Training Samples",
      features: "Features",
      trainingDate: "Training Date",
      modelStatus: "Status",
      coefficientId: "Coefficient ID",
      trained: "Trained",
      status: "Status",
    },
    fr: {
      title: "Coefficients du Modèle de Régression",
      description:
        "Coefficients de régression linéaire pour la prédiction de la croissance ST",
      intercept: "Intersectiony (β₀)",
      coefficients: "Variables du Modèle",
      noData: "Aucun coefficient chargé. Veuillez d'abord entraîner le modèle.",
      loading: "Chargement des coefficients...",
      error: "Erreur lors du chargement des coefficients",
      refresh: "Actualiser",
      modelGroup: "Groupe de Modèle",
      download: "Télécharger",
      variable: "Variable",
      coefficient: "Coefficient",
      interpretation: "Effet",
      positive: "Effet positif",
      negative: "Effet négatif",
      lastUpdated: "Dernière mise à jour",
      modelInfo: "Performance et Configuration du Modèle",
      positiveEffect: "↑ Augmente la croissance",
      negativeEffect: "↓ Diminue la croissance",
      minimalEffect: "Effet minimal",
      modelType: "Type de Modèle",
      rSquared: "Score R²",
      rmse: "RMSE",
      samples: "Échantillons d'Entraînement",
      features: "Caractéristiques",
      trainingDate: "Date d'Entraînement",
      modelStatus: "Statut",
      coefficientId: "ID Coefficient",
      trained: "Entraîné",
      status: "Statut",
    },
  };

  const t = translations[language];

  useEffect(() => {
    if (!selectedModelGroup) return;
    if (selectedModelGroup !== modelGroup) {
      setModelGroup(selectedModelGroup);
    }
  }, [selectedModelGroup, modelGroup]);

  const fetchModelGroups = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/species-list`);
      if (!response.ok) return;

      const data = (await response.json()) as SpeciesListResponse;
      const groups = data.species_groups || [];
      setModelGroups(groups);

      if (groups.length === 0) return;
      const next = groups.includes(modelGroup)
        ? modelGroup
        : selectedModelGroup && groups.includes(selectedModelGroup)
          ? selectedModelGroup
          : groups[0];

      if (next !== modelGroup) {
        setModelGroup(next);
      }
      onModelGroupChange?.(next);
    } catch (fetchErr) {
      console.error("Error fetching model groups:", fetchErr);
    }
  };

  // Fetch coefficients from backend
  const fetchCoefficients = async () => {
    setLoading(true);
    setError(null);
    setSuccess(false);
    try {
      const groupQuery = modelGroup
        ? `?species_group=${encodeURIComponent(modelGroup)}`
        : "";
      const cacheBustedQuery = groupQuery
        ? `${groupQuery}&_ts=${Date.now()}`
        : `?_ts=${Date.now()}`;
      const response = await fetch(
        `${API_BASE}/api/model-coefficients${cacheBustedQuery}`,
        { cache: "no-store" },
      );

      if (!response.ok) {
        const text = await response.text();
        let message = `HTTP error! status: ${response.status}`;
        if (text) {
          try {
            const json = JSON.parse(text);
            message = json.detail || json.message || text || message;
          } catch {
            message = text;
          }
        }
        throw new Error(message);
      }

      const data = await response.json();

      if (!data?.coefficients) {
        throw new Error("No coefficients returned by backend");
      }

      setIntercept(Number(data.intercept || 0));
      const nextModelInfo = (data.model_info || null) as ModelInfo | null;
      setModelInfo(nextModelInfo);

      // Map coefficients in the correct order
      const correctOrder = ["BA", "BA2", "AGSfrac", "TCR", "DQ", "q", "Cut"];
      setCoefficients(
        correctOrder.map((varName) => ({
          name: varName,
          value: Number(data.coefficients[varName] || 0),
          label: varName === "BA2" ? "BA²" : varName.toUpperCase(),
        })),
      );

      const trainedAt = nextModelInfo?.training_date
        ? new Date(nextModelInfo.training_date).toLocaleString(
            language === "en" ? "en-US" : "fr-FR",
          )
        : new Date().toLocaleString(language === "en" ? "en-US" : "fr-FR");
      setLastUpdated(trainedAt);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error occurred");
      console.error("Error fetching coefficients:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModelGroups();
  }, []);

  useEffect(() => {
    fetchCoefficients();
  }, [language, modelGroup]);

  // Download coefficients as JSON
  const downloadCoefficients = () => {
    const data = {
      model_group: modelGroup,
      intercept,
      coefficients: coefficients.reduce(
        (acc, coef) => {
          acc[coef.name] = coef.value;
          return acc;
        },
        {} as Record<string, number>,
      ),
      model_info: modelInfo,
      timestamp: new Date().toISOString(),
    };

    const element = document.createElement("a");
    element.href =
      "data:text/json;charset=utf-8," +
      encodeURIComponent(JSON.stringify(data, null, 2));
    element.download = `model_coefficients_${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const getInterpretation = (
    value: number,
  ): { text: string; color: string } => {
    if (value > 0.01) {
      return {
        text: t.positiveEffect,
        color: "text-green-600 dark:text-green-400",
      };
    } else if (value < -0.01) {
      return {
        text: t.negativeEffect,
        color: "text-red-600 dark:text-red-400",
      };
    } else {
      return {
        text: t.minimalEffect,
        color: "text-gray-500 dark:text-slate-400",
      };
    }
  };

  return (
    <div className="w-full space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              <div>
                <CardTitle>{t.title}</CardTitle>
                <CardDescription>{t.description}</CardDescription>
              </div>
            </div>
            <div className="flex gap-2 items-center">
              <div className="w-64">
                <Select
                  value={modelGroup}
                  onValueChange={(value) => {
                    setModelGroup(value);
                    onModelGroupChange?.(value);
                  }}
                  disabled={loading || modelGroups.length === 0}
                >
                  <SelectTrigger>
                    <SelectValue placeholder={t.modelGroup} />
                  </SelectTrigger>
                  <SelectContent>
                    {modelGroups.map((group) => (
                      <SelectItem key={group} value={group}>
                        {group}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {success && (
                <div className="flex items-center gap-1 text-green-600 text-sm">
                  <Check className="h-4 w-4" />
                  Loaded
                </div>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={fetchCoefficients}
                disabled={loading}
              >
                <RefreshCw
                  className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`}
                />
                {t.refresh}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={downloadCoefficients}
                disabled={coefficients.length === 0}
              >
                <Download className="h-4 w-4 mr-2" />
                {t.download}
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Error State */}
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="dark:text-red-400">
                {error}
              </AlertDescription>
            </Alert>
          )}

          {/* Loading State */}
          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="text-center">
                <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2 text-blue-600" />
                <p>{t.loading}</p>
              </div>
            </div>
          )}

          {/* Model Info Display */}
          {!loading && modelInfo && (
            <div className="grid grid-cols-2 gap-3">
              {modelInfo.n_samples && (
                <div className="rounded-lg bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-500/20 dark:to-pink-500/10 p-3 border border-purple-200 dark:border-purple-700/50">
                  <p className="text-xs font-medium text-gray-600 dark:text-purple-300/80">
                    {t.samples}
                  </p>
                  <p className="text-lg font-bold text-purple-600 dark:text-purple-400">
                    {modelInfo.n_samples}
                  </p>
                </div>
              )}
              {modelInfo.n_features && (
                <div className="rounded-lg bg-gradient-to-br from-orange-50 to-amber-50 dark:from-amber-800/20 dark:to-orange-700/35 p-3 border border-orange-200 dark:border-amber-600/30">
                  <p className="text-xs font-medium text-gray-600 dark:text-amber-300/90">
                    {t.features}
                  </p>
                  <p className="text-lg font-bold text-orange-600 dark:text-amber-500">
                    {modelInfo.n_features}
                  </p>
                </div>
              )}
              {modelInfo.coefficient_id != null && (
                <div className="rounded-lg bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 p-3 border border-blue-200 dark:border-blue-700/40">
                  <p className="text-xs font-medium text-gray-600 dark:text-blue-300/80">
                    {t.coefficientId}
                  </p>
                  <p className="text-lg font-bold text-blue-600 dark:text-blue-400 font-mono">
                    {modelInfo.coefficient_id}
                  </p>
                </div>
              )}
              {modelInfo.training_date && (
                <div className="rounded-lg bg-gradient-to-br from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 p-3 border border-emerald-200 dark:border-emerald-700/40">
                  <p className="text-xs font-medium text-gray-600 dark:text-emerald-300/80">
                    {t.trainingDate}
                  </p>
                  <p className="text-sm font-bold text-emerald-700 dark:text-emerald-300 font-mono break-all">
                    {new Date(modelInfo.training_date).toLocaleString(
                      language === "en" ? "en-US" : "fr-FR",
                    )}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Intercept Display */}
          {!loading && coefficients.length > 0 && (
            <>
              <div className="rounded-lg border border-gray-200 dark:border-slate-800 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-indigo-950/80">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-700 dark:text-slate-400">
                      {t.intercept}
                    </p>
                    <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                      {intercept.toFixed(6)}
                    </p>
                  </div>
                  <Badge variant="secondary" className="text-lg px-3 py-1">
                    β₀
                  </Badge>
                </div>
              </div>

              {/* Coefficients Table */}
              <div className="space-y-2">
                <p className="font-semibold text-sm">{t.coefficients}</p>
                <div className="rounded-lg border overflow-hidden">
                  <Table>
                    <TableHeader className="bg-gray-50 dark:bg-slate-800/50">
                      <TableRow>
                        <TableHead className="font-semibold dark:text-slate-300">
                          {t.variable}
                        </TableHead>
                        <TableHead className="text-right font-semibold">
                          {t.coefficient}
                        </TableHead>
                        <TableHead className="font-semibold">
                          {t.interpretation}
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {coefficients.map((coef, index) => {
                        const interpretation = getInterpretation(coef.value);
                        return (
                          <TableRow
                            key={index}
                            className="hover:bg-gray-50 dark:hover:bg-slate-800/40 border-b dark:border-slate-800 transition-colors"
                          >
                            <TableCell className="font-medium">
                              <div className="flex items-center gap-2">
                                <Badge variant="outline">β{index + 1}</Badge>
                                <span>{coef.label}</span>
                              </div>
                            </TableCell>
                            <TableCell className="text-right font-mono font-semibold">
                              <span
                                className={
                                  coef.value > 0
                                    ? "text-green-600"
                                    : coef.value < 0
                                      ? "text-red-600 dark:text-red-400"
                                      : "text-gray-600"
                                }
                              >
                                {coef.value > 0 ? "+" : ""}
                                {coef.value.toFixed(6)}
                              </span>
                            </TableCell>
                            <TableCell>
                              <span className={interpretation.color}>
                                {interpretation.text}
                              </span>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
              </div>

              {/* Last Updated Section */}
              <div className="flex items-center justify-between text-xs pt-2 border-t border-gray-200 dark:border-slate-800">
                <span className="text-gray-500 dark:text-amber-400 font-medium">
                  {t.lastUpdated}:
                </span>
                <span className="text-gray-600 dark:text-amber-400 font-mono font-semibold">
                  {lastUpdated}
                </span>
              </div>
            </>
          )}

          {/* Empty State */}
          {!loading && coefficients.length === 0 && !error && (
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
