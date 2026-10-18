/**
 * Frontend DTO type contracts.
 *
 * The growth-model types live next to the hook that uses them
 * (`hooks/useBAGrowthModel.ts`); this module re-exports them and adds the
 * shapes shared by several tabs.
 */

export type {
  CoefficientsResponse,
  GrowthCalculationParams,
  GrowthCalculationResult,
  YearlyProjection,
} from "../hooks/useBAGrowthModel";

export interface SpeciesListResponse {
  species_groups: string[];
  forest_units: string[];
  mapping: Record<string, string>;
}

export interface BackgroundJob {
  job_id: string;
  kind: "training" | "benchmark";
  status: "running" | "completed" | "failed";
  started_at: string;
  finished_at: string | null;
  progress: { total: number; done: number; failed: number; current: string | null };
  results: Record<string, unknown>;
  errors: Record<string, string>;
}

export interface LeaderboardRow {
  group: string;
  model: string;
  family: string;
  rank: number;
  rmse: number | null;
  rmse_std: number | null;
  r2: number | null;
  mae: number | null;
  p95_ae: number | null;
  bias: number | null;
  rmse_improvement_pct: number | null;
  p_value_vs_ols: number | null;
  wrong_sign_count: number | null;
}
