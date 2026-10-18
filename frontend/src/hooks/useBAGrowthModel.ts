import { useCallback, useState } from "react";

/**
 * Thin typed client for the growth-model endpoints.
 *
 *   GET /api/model-coefficients?model_group=…   fitted coefficients + quality metrics
 *   GET /api/bagrowth/calculate?species_group=…  per-plot predictions + yearly projection
 *   GET /api/benchmarks                          latest model-comparison leaderboard
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface CoefficientsResponse {
  source: string;
  model_group?: string;
  intercept: number;
  coefficients: Record<string, number>;
  model_info: Record<string, unknown>;
}

export interface GrowthCalculationParams {
  speciesGroup: string;
  currentYear: number;
  projectionYears?: number;
  modelVersion?: string;
  refresh?: boolean;
}

export interface YearlyProjection {
  year: number;
  BA_predicted: number;
  BAgrowth_predicted: number;
  RGR: number;
}

export interface GrowthCalculationResult {
  species_group: string;
  mean_BAgrowth: number;
  mean_RGR: number;
  std_BAgrowth: number;
  ci_lower: number;
  ci_upper: number;
  n_plots: number;
  mean_BA_current: number;
  yearly_projection: YearlyProjection[];
  model_info: Record<string, unknown>;
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* body not JSON */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export function useBAGrowthModel() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [coefficients, setCoefficients] = useState<CoefficientsResponse | null>(null);
  const [lastResult, setLastResult] = useState<GrowthCalculationResult | null>(null);

  const fetchCoefficients = useCallback(async (modelGroup?: string) => {
    setLoading(true);
    setError(null);
    try {
      const query = modelGroup ? `?model_group=${encodeURIComponent(modelGroup)}` : "";
      const data = await getJson<CoefficientsResponse>(`${API_BASE}/api/model-coefficients${query}`);
      setCoefficients(data);
      return data;
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to fetch model coefficients");
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const calculateGrowth = useCallback(async (params: GrowthCalculationParams) => {
    setLoading(true);
    setError(null);
    try {
      const search = new URLSearchParams({
        species_group: params.speciesGroup,
        current_year: String(params.currentYear),
        projection_years: String(params.projectionYears ?? 10),
        model_version: params.modelVersion ?? "1.0",
      });
      if (params.refresh) search.set("refresh", "true");
      const data = await getJson<GrowthCalculationResult>(`${API_BASE}/api/bagrowth/calculate?${search}`);
      setLastResult(data);
      return data;
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Calculation request failed");
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { loading, error, coefficients, lastResult, fetchCoefficients, calculateGrowth };
}
