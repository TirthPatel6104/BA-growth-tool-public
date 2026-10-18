import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Alert, AlertDescription } from "./ui/alert";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Info, TrendingUp, BarChart3 } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
  LineChart,
  Line,
  ReferenceLine,
  ScatterChart,
  Scatter,
} from "recharts";
import type { PlotData, GrowthResult, Language } from "../App";

interface VisualizationProps {
  language: Language;
  plotData: PlotData[];
  growthResults: GrowthResult[];
}

interface GroupCalculationResponse {
  species_group?: string;
  mean_BAgrowth?: number;
  mean_RGR?: number;
  std_BAgrowth?: number;
  ci_lower?: number;
  ci_upper?: number;
  n_plots?: number;
  mean_BA_current?: number;
  per_plot_results?: Array<Record<string, unknown>>;
  latest_rows_data?: Array<Record<string, unknown>>;
  filtered_rows_data?: Array<Record<string, unknown>>;
  outlier_analysis?: {
    method?: string;
    BA?:       { sample_size?: number; mean?: number; std_dev?: number; lower_bound?: number; upper_bound?: number; outside_count?: number; outside_percent?: number };
    AGSfrac?:  { sample_size?: number; mean?: number; std_dev?: number; lower_bound?: number; upper_bound?: number; outside_count?: number; outside_percent?: number };
    TCR?:      { sample_size?: number; mean?: number; std_dev?: number; lower_bound?: number; upper_bound?: number; outside_count?: number; outside_percent?: number };
    DQ?:       { sample_size?: number; mean?: number; std_dev?: number; lower_bound?: number; upper_bound?: number; outside_count?: number; outside_percent?: number };
    q?:        { sample_size?: number; mean?: number; std_dev?: number; lower_bound?: number; upper_bound?: number; outside_count?: number; outside_percent?: number };
    BAgrowth?: { sample_size?: number; mean?: number; std_dev?: number; lower_bound?: number; upper_bound?: number; outside_count?: number; outside_percent?: number };
    RGR?:      { sample_size?: number; mean?: number; std_dev?: number; lower_bound?: number; upper_bound?: number; outside_count?: number; outside_percent?: number };
  };
  yearly_projection?: Array<Record<string, unknown>>;
  model_info?: Record<string, unknown>;
}

interface GroupState {
  kind: "species" | "forest_unit";
  open: boolean;
  loading: boolean;
  error: string | null;
  data?: GroupCalculationResponse;
}


const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export function Visualization({
  language,
  plotData,
  growthResults,
}: VisualizationProps) {
  const [groupStates, setGroupStates] = useState<Record<string, GroupState>>({});
  const [speciesGroupList, setSpeciesGroupList] = useState<string[]>([]);
  const [forestUnitGroupList, setForestUnitGroupList] = useState<string[]>([]);
  const [groupListLoading, setGroupListLoading] = useState(true);
  const [groupListError, setGroupListError] = useState<string | null>(null);

  const translations = {
    en: {
      title: "Growth Analysis Visualization",
      description:
        "Graphical analysis of basal area growth and species / forest unit model groups",
      noData: "No calculation results available. Please calculate growth first.",
      groupSelectorTitle: "Open graphs for species or forest units",
      speciesSection: "Species models",
      forestUnitSection: "Forest unit models",
      openGraph: "Open graph",
      closeGraph: "Close graph",
      loadingGroups: "Loading available groups...",
      loadingGroup: "Loading group data...",
      groupError: "Unable to load group data.",
      noGroupSelected:
        "No graphs are open. Select a species or forest unit to load a chart.",
      actual: "Actual",
      predicted: "Predicted",
      noCut: "No cut",
      withCut: "Cut",
      cut: "Cut",
      tcr: "TCR",
      plot: "Plot",
      bAgrowth: "BA Growth (m²/ha/yr)",
      basalArea: "Basal Area (m²/ha)",
      projectionTitle: "Projected BA Trend",
      year: "Year",
      plotId: "Plot ID",
      chartTitle: "BA vs BA Growth",
      currentVsFuture: "Current vs. Projected Basal Area",
      volumeProjections: "Volume Projections by Plot",
      growthTrends: "Growth Trends",
      volume: "Volume (m³/ha)",
      current: "Current",
      projected: "Projected",
      growth: "Growth",
      outliers: "Outliers",
      twoSigmaRule: "outside ±2SD",
    },
    fr: {
      title: "Visualisation de l'Analyse de Croissance",
      description:
        "Analyse graphique de la croissance de la surface terrière et des modèles par espèce / unité forestière",
      noData:
        "Aucun résultat de calcul disponible. Veuillez d'abord calculer la croissance.",
      groupSelectorTitle:
        "Ouvrez des graphiques pour les espèces ou les unités forestières",
      speciesSection: "Modèles d'espèces",
      forestUnitSection: "Modèles d'unités forestières",
      openGraph: "Ouvrir le graphique",
      closeGraph: "Fermer le graphique",
      loadingGroups: "Chargement des groupes disponibles...",
      loadingGroup: "Chargement des données du groupe...",
      groupError: "Impossible de charger les données du groupe.",
      noGroupSelected:
        "Aucun graphique n'est ouvert. Sélectionnez une espèce ou une unité forestière pour charger un graphique.",
      actual: "Observé",
      predicted: "Prédit",
      noCut: "Sans coupe",
      withCut: "Avec coupe",
      cut: "Coupe",
      tcr: "TCR",
      plot: "Placette",
      bAgrowth: "Croissance ST (m²/ha/yr)",
      basalArea: "Surface Terrière (m²/ha)",
      projectionTitle: "Projection de la Surface Terrière",
      year: "Année",
      plotId: "ID Placette",
      chartTitle: "BA vs Croissance ST",
      currentVsFuture: "Surface Terrière Actuelle vs Projetée",
      volumeProjections: "Projections de Volume par Placette",
      growthTrends: "Tendances de Croissance",
      volume: "Volume (m³/ha)",
      current: "Actuelle",
      projected: "Projetée",
      growth: "Croissance",
      outliers: "Valeurs aberrantes",
      twoSigmaRule: "hors ±2ET",
    },
  };

  const t = translations[language];

  const currentYear = new Date().getFullYear();
  const projectionYears = 10;

  useEffect(() => {
    const fetchGroupList = async () => {
      setGroupListLoading(true);
      setGroupListError(null);

      try {
        const response = await fetch(`${API_BASE}/api/species-list`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        const species = Array.isArray(data.species_groups)
          ? (data.species_groups as string[])
          : [];
        const forestUnits = Array.isArray(data.forest_units)
          ? (data.forest_units as string[])
          : [];

        setSpeciesGroupList(species);
        setForestUnitGroupList(forestUnits);
        setGroupStates((prev) => {
          const next: Record<string, GroupState> = {};

          species.forEach((name: string) => {
            next[name] = prev[name]
              ? { ...prev[name], kind: "species" }
              : { kind: "species", open: false, loading: false, error: null };
          });

          forestUnits.forEach((name: string) => {
            next[name] = prev[name]
              ? { ...prev[name], kind: "forest_unit" }
              : { kind: "forest_unit", open: false, loading: false, error: null };
          });

          return next;
        });
      } catch (error) {
        setGroupListError(
          error instanceof Error ? error.message : "Unknown error",
        );
      } finally {
        setGroupListLoading(false);
      }
    };

    fetchGroupList();
  }, [language]);

  const fetchGroupData = async (groupName: string) => {
    setGroupStates((prev) => ({
      ...prev,
      [groupName]: {
        ...prev[groupName],
        loading: true,
        error: null,
      },
    }));

    try {
      const response = await fetch(
        `${API_BASE}/api/bagrowth/calculate?species_group=${encodeURIComponent(
          groupName,
        )}&current_year=${currentYear}&projection_years=${projectionYears}`,
      );

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `HTTP ${response.status}`);
      }

      const data: GroupCalculationResponse = await response.json();
      setGroupStates((prev) => ({
        ...prev,
        [groupName]: {
          ...prev[groupName],
          loading: false,
          data,
          error: null,
        },
      }));
    } catch (error) {
      setGroupStates((prev) => ({
        ...prev,
        [groupName]: {
          ...prev[groupName],
          loading: false,
          error:
            error instanceof Error ? error.message : "Failed to load data",
        },
      }));
    }
  };

  const toggleGroup = (groupName: string) => {
    const group = groupStates[groupName];
    const opening = !group?.open;
    setGroupStates((prev) => ({
      ...prev,
      [groupName]: {
        ...prev[groupName],
        open: opening,
        error: opening ? null : prev[groupName]?.error ?? null,
      },
    }));

    if (opening && !group?.data) {
      fetchGroupData(groupName);
    }
  };

  interface NormalizedRow {
    plotKey: string;
    BA: number;
    BAgrowth: number;
    Cut: number;
    TCR?: number;
  }

  const normalizeActualRows = (rows: Array<Record<string, unknown>> = []) =>
    rows.map((row) => {
      const BA = Number(
        row["BA"] ?? row["ba"] ?? row["BA_current"] ?? row["ba_current"] ?? 0,
      );
      const rawGrowth =
        row["BAgrowth"] ?? row["BAgrowth_Input"] ?? row["bagrowth"];
      const BAgrowth =
        Number(rawGrowth ?? 0) ||
        Number(row["RGR"] ?? row["rgr"] ?? row["RGR_Input"] ?? row["rgr_input"] ?? 0) * BA ||
        0;
      const Cut = Number(row["Cut"] ?? row["cut"] ?? 0) || 0;
      const TCR = row["TCR"] !== undefined ? Number(row["TCR"]) : undefined;
      return {
        plotKey: String(row["PlotKey"] ?? row["plotKey"] ?? ""),
        BA,
        BAgrowth,
        Cut,
        TCR,
      };
    }) as NormalizedRow[];

  const renderGroupCard = (groupName: string, state: GroupState) => {
    const data = state.data;
    const filteredRows = normalizeActualRows(data?.filtered_rows_data);
    const MODEL_VARS: Array<{ key: keyof NonNullable<GroupCalculationResponse['outlier_analysis']>; label: string }> = [
      { key: 'BA',       label: 'BA (m²/ha)' },
      { key: 'AGSfrac',  label: 'AGSfrac' },
      { key: 'TCR',      label: 'TCR' },
      { key: 'DQ',       label: 'DQ (cm)' },
      { key: 'q',        label: 'q-factor' },
      { key: 'BAgrowth', label: 'BAgrowth (m²/ha/yr)' },
      { key: 'RGR',      label: 'RGR' },
    ];
    const projectionRows = (data?.yearly_projection ?? []).map((row) => ({
      year: Number(row["year"] ?? row["Year"] ?? 0),
      BA: Number(row["BA_predicted"] ?? row["BA"] ?? 0),
      growth: Number(row["BAgrowth_predicted"] ?? row["BAgrowth"] ?? 0),
    }));

    const sortByBA = (rows: NormalizedRow[]) =>
      [...rows].sort((a, b) => a.BA - b.BA);

    const actualCut0 = sortByBA(filteredRows.filter((point) => point.Cut === 0));
    const actualCut1 = sortByBA(filteredRows.filter((point) => point.Cut === 1));

    const actualRows = [...actualCut0, ...actualCut1];
    const getAxisBound = (key: 'BA' | 'BAgrowth') => {
      const values = actualRows
        .map((point) => Number(point[key]))
        .filter((value) => Number.isFinite(value));
      if (values.length === 0) {
        return { min: 0, max: 1 };
      }
      const sorted = values.slice().sort((a, b) => a - b);
      const min = sorted[0];
      const max = sorted[sorted.length - 1];
      const useLower = sorted.length > 10 ? sorted[Math.floor(sorted.length * 0.02)] : min;
      const useUpper = sorted.length > 10 ? sorted[Math.ceil(sorted.length * 0.98) - 1] : max;
      const range = Math.abs(useUpper - useLower);
      const padding = Math.max(range * 0.08, range < 1 ? 0.01 : 1);
      return {
        min:
          useLower > 0
            ? Math.max(useLower * 0.9, useLower - padding)
            : useLower - padding,
        max: useUpper + padding,
      };
    };

    const BAExtent = getAxisBound('BA');
    const BAgrowthExtent = getAxisBound('BAgrowth');

    return (
      <Card key={groupName} className="border-green-200">
        <CardHeader className="bg-slate-50 dark:bg-slate-900">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div className="space-y-2">
              <CardTitle className="text-green-900 dark:text-green-300 flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                {groupName}
              </CardTitle>
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline">
                  {state.kind === "species" ? t.speciesSection : t.forestUnitSection}
                </Badge>
                {data?.n_plots != null && (
                  <Badge variant="secondary">{`${data.n_plots} plots`}</Badge>
                )}
                {data?.mean_BAgrowth != null && (
                  <Badge variant="outline">{`${t.bAgrowth}: ${data.mean_BAgrowth.toFixed(2)}`}</Badge>
                )}
              </div>
            </div>
            <Button
              variant={state.open ? "secondary" : "outline"}
              size="sm"
              onClick={() => toggleGroup(groupName)}
            >
              {state.open ? t.closeGraph : t.openGraph}
            </Button>
          </div>
        </CardHeader>

        <CardContent>
          {state.loading && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-slate-700">
              {t.loadingGroup}
            </div>
          )}
          {state.error && (
            <Alert className="border-red-200 bg-red-50">
              <Info className="h-4 w-4 text-red-600" />
              <AlertDescription className="text-red-800">
                {t.groupError} {state.error}
              </AlertDescription>
            </Alert>
          )}

          {!state.loading && data && (
            <div className="space-y-6">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-200">
                  <p className="text-sm font-semibold text-slate-700">{t.chartTitle}</p>
                  <ResponsiveContainer width="100%" height={280}>
                    <ScatterChart margin={{ top: 16, right: 16, bottom: 24, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        type="number"
                        dataKey="BA"
                        name={t.basalArea}
                        unit="m²/ha"
                        tick={{ fill: "#374151" }}
                        domain={[BAExtent.min, BAExtent.max]}
                        padding={{ left: 10, right: 10 }}
                      />
                      <YAxis
                        type="number"
                        dataKey="BAgrowth"
                        name={t.bAgrowth}
                        unit="m²/ha/yr"
                        tick={{ fill: "#374151" }}
                        domain={[BAgrowthExtent.min, BAgrowthExtent.max]}
                        padding={{ top: 10, bottom: 10 }}
                      />
                      <Tooltip
                        cursor={{ strokeDasharray: "3 3" }}
                        formatter={(value) => [value, ""]}
                        labelFormatter={(_label) => ""}
                      />
                      <Legend />
                      <ReferenceLine y={0} stroke="#64748b" strokeDasharray="4 4" />
                      <Scatter
                        name={`${t.actual} (${t.noCut})`}
                        data={actualCut0}
                        dataKey="BAgrowth"
                        stroke="#16a34a"
                        fill="#16a34a"
                      />
                      <Scatter
                        name={`${t.actual} (${t.withCut})`}
                        data={actualCut1}
                        dataKey="BAgrowth"
                        stroke="#dc2626"
                        fill="#dc2626"
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                <div className="rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-200">
                  <p className="text-sm font-semibold text-slate-700">{t.projectionTitle}</p>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={projectionRows} margin={{ top: 16, right: 16, bottom: 24, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="year"
                        tick={{ fill: "#374151" }}
                        domain={["auto", "auto"]}
                        allowDecimals={false}
                      />
                      <YAxis
                        yAxisId="left"
                        tick={{ fill: "#374151" }}
                        label={{
                          value: t.basalArea,
                          angle: -90,
                          position: "insideLeft",
                        }}
                        domain={["auto", "auto"]}
                        width={60}
                      />
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        tick={{ fill: "#374151" }}
                        label={{
                          value: t.bAgrowth,
                          angle: 90,
                          position: "insideRight",
                        }}
                        domain={["auto", "auto"]}
                        width={60}
                      />
                      <Tooltip />
                      <Legend />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="BA"
                        stroke="#15803d"
                        strokeWidth={2}
                        dot={false}
                        name={t.basalArea}
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="growth"
                        stroke="#f97316"
                        strokeWidth={2}
                        dot={false}
                        name={t.bAgrowth}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {data.outlier_analysis && (
                <div className="rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-200">
                  <p className="text-sm font-semibold text-slate-700 mb-3">
                    {t.outliers} <span className="text-xs font-normal text-slate-500">({t.twoSigmaRule})</span>
                  </p>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-slate-200">
                          <th className="text-left py-1.5 pr-4 font-semibold text-slate-600">{t.plot}</th>
                          <th className="text-right py-1.5 pr-4 font-semibold text-slate-600">n</th>
                          <th className="text-right py-1.5 pr-4 font-semibold text-slate-600">Mean</th>
                          <th className="text-right py-1.5 pr-4 font-semibold text-slate-600">SD</th>
                          <th className="text-right py-1.5 pr-4 font-semibold text-slate-600">Lower (−2SD)</th>
                          <th className="text-right py-1.5 pr-4 font-semibold text-slate-600">Upper (+2SD)</th>
                          <th className="text-right py-1.5 pr-4 font-semibold text-slate-600">Outside</th>
                          <th className="text-right py-1.5 font-semibold text-slate-600">%</th>
                        </tr>
                      </thead>
                      <tbody>
                        {MODEL_VARS.map(({ key, label }) => {
                          const stats = data.outlier_analysis?.[key as keyof typeof data.outlier_analysis] as {
                            sample_size?: number;
                            mean?: number;
                            std_dev?: number;
                            lower_bound?: number;
                            upper_bound?: number;
                            outside_count?: number;
                            outside_percent?: number;
                          } | undefined;
                          if (!stats) return null;
                          const pct = stats.outside_percent ?? 0;
                          const highlight = pct > 5;
                          return (
                            <tr key={key} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                              <td className="py-1.5 pr-4 font-mono text-slate-800">{label}</td>
                              <td className="text-right py-1.5 pr-4 text-slate-600">{stats.sample_size ?? '—'}</td>
                              <td className="text-right py-1.5 pr-4 text-slate-600">{stats.mean?.toFixed(4) ?? '—'}</td>
                              <td className="text-right py-1.5 pr-4 text-slate-600">{stats.std_dev?.toFixed(4) ?? '—'}</td>
                              <td className="text-right py-1.5 pr-4 text-slate-600">{stats.lower_bound?.toFixed(4) ?? '—'}</td>
                              <td className="text-right py-1.5 pr-4 text-slate-600">{stats.upper_bound?.toFixed(4) ?? '—'}</td>
                              <td className="text-right py-1.5 pr-4 font-semibold text-slate-800">{stats.outside_count ?? 0}</td>
                              <td className={`text-right py-1.5 font-semibold ${highlight ? 'text-red-600' : 'text-slate-700'}`}>
                                {pct.toFixed(1)}%
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const openGroups = Object.entries(groupStates).filter(([, state]) => state.open);

  const showLegacyCharts = growthResults.length > 0;

  const comparisonData = growthResults.map((result) => {
    const plot = plotData.find((p) => p.plotId === result.plotId);
    return {
      plotId: result.plotId,
      current: plot?.basalArea || 0,
      projected: (plot?.basalArea || 0) + result.predictedGrowth,
      growth: result.predictedGrowth,
    };
  });

  const volumeData = growthResults.map((result) => ({
    plotId: result.plotId,
    volume: result.volumeEstimate,
  }));

  const trendData = Array.from(
    { length: growthResults[0]?.projectionYears || 10 },
    (_, i) => {
      const year = i + 1;
      const avgGrowthPerYear =
        growthResults.reduce((sum, r) => sum + r.predictedGrowth, 0) /
        growthResults.length /
        (growthResults[0]?.projectionYears || 1);
      const avgCurrentBA =
        plotData.reduce((sum, p) => sum + p.basalArea, 0) / plotData.length;

      return {
        year: `Year ${year}`,
        basalArea: avgCurrentBA + avgGrowthPerYear * year,
      };
    },
  );

  return (
    <div className="space-y-6">
      <Card className="border-green-200">
        <CardHeader className="bg-green-50">
          <CardTitle className="text-green-900 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            {t.title}
          </CardTitle>
          <CardDescription>{t.description}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
            <div className="space-y-4 rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
              <p className="text-sm font-semibold text-slate-900">{t.groupSelectorTitle}</p>
              {groupListLoading ? (
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-slate-700">
                  {t.loadingGroups}
                </div>
              ) : groupListError ? (
                <Alert className="border-red-200 bg-red-50">
                  <Info className="h-4 w-4 text-red-600" />
                  <AlertDescription className="text-red-800">
                    {groupListError}
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="space-y-4">
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                      {t.speciesSection}
                    </p>
                    <div className="mt-3 space-y-2">
                      {speciesGroupList.map((groupName) => {
                        const state = groupStates[groupName];
                        return (
                          <div
                            key={groupName}
                            className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3 py-2"
                          >
                            <div className="min-w-0 text-sm text-slate-800">
                              <div className="truncate font-medium">{groupName}</div>
                              <div className="text-xs text-slate-500">
                                {state?.loading ? t.loadingGroup : state?.open ? t.closeGraph : t.openGraph}
                              </div>
                            </div>
                            <Button
                              variant={state?.open ? "secondary" : "outline"}
                              size="sm"
                              onClick={() => toggleGroup(groupName)}
                            >
                              {state?.open ? t.closeGraph : t.openGraph}
                            </Button>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                      {t.forestUnitSection}
                    </p>
                    <div className="mt-3 space-y-2">
                      {forestUnitGroupList.map((groupName) => {
                        const state = groupStates[groupName];
                        return (
                          <div
                            key={groupName}
                            className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3 py-2"
                          >
                            <div className="min-w-0 text-sm text-slate-800">
                              <div className="truncate font-medium">{groupName}</div>
                              <div className="text-xs text-slate-500">
                                {state?.loading ? t.loadingGroup : state?.open ? t.closeGraph : t.openGraph}
                              </div>
                            </div>
                            <Button
                              variant={state?.open ? "secondary" : "outline"}
                              size="sm"
                              onClick={() => toggleGroup(groupName)}
                            >
                              {state?.open ? t.closeGraph : t.openGraph}
                            </Button>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-6">
              {openGroups.length === 0 ? (
                <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50 p-8 text-slate-600">
                  {t.noGroupSelected}
                </div>
              ) : (
                openGroups.map(([groupName, state]) => renderGroupCard(groupName, state))
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {showLegacyCharts && (
        <Card className="border-green-200">
          <CardHeader className="bg-green-50">
            <CardTitle className="text-green-900 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              {t.growthTrends}
            </CardTitle>
            <CardDescription>
              {t.description}
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white p-4 rounded-lg border border-green-100">
                <h3 className="text-green-900 mb-4">{t.currentVsFuture}</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="plotId" />
                    <YAxis
                      label={{
                        value: t.basalArea,
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="current" fill="#15803d" name={t.current} />
                    <Bar dataKey="projected" fill="#22c55e" name={t.projected} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white p-4 rounded-lg border border-green-100">
                <h3 className="text-green-900 mb-4">{t.volumeProjections}</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={volumeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="plotId" />
                    <YAxis
                      label={{
                        value: t.volume,
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="volume" fill="#0891b2" name={t.volume} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white p-4 rounded-lg border border-green-100 lg:col-span-2">
                <h3 className="text-green-900 mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5" />
                  {t.growthTrends}
                </h3>
                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={trendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis
                      label={{
                        value: t.basalArea,
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="basalArea"
                      stroke="#15803d"
                      fill="#22c55e"
                      fillOpacity={0.6}
                      name={t.basalArea}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
