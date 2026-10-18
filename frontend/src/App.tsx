import { useEffect, useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Card } from "./components/ui/card";
import {
  FileText,
  BarChart3,
  Database,
  Calculator,
  TrendingUp,
  Zap,
  FlaskConical,
} from "lucide-react";
import { DataEntry } from "./components/DataEntry";
import { GrowthCalculator } from "./components/GrowthCalculator";
import { Visualization } from "./components/Visualization";
import { Reports } from "./components/Reports";
import { CoefficientDisplay } from "./components/CoefficientDisplay";
import { BAGrowthCalculator } from "./components/BAGrowthCalculator";
import { ModelBenchmarks } from "./components/ModelBenchmarks";
import { Header } from "./components/Header";

import "./styles/globals.css";

export type Language = "en" | "fr";
type Theme = "light" | "dark";

const APP_ACTIVE_TAB_KEY = "bagt_active_tab_v1";
const APP_MODEL_GROUP_KEY = "bagt_selected_model_group_v1";
const VALID_TABS = new Set([
  "data-entry",
  "calculations",
  "ba-growth",
  "coefficients",
  "benchmarks",
  "visualization",
  "reports",
]);

export interface PlotData {
  plotId: string;
  location: string;
  measurementDate: string;
  species: string;
  dbh: number; // Diameter at breast height (cm)
  basalArea: number; // m²/ha
  age: number;
  siteIndex: number;
  agsFrac: number; // Proportion of live basal area in AGS trees
  tcr: number; // Timber capability rating
  dq: number; // Quadratic mean DBH of live trees
  qFactor: number; // The q-factor
  cut: number; // 0 = no cutting, 1 = cutting
}

export interface GrowthResult {
  plotId: string;
  predictedGrowth: number;
  volumeEstimate: number;
  projectionYears: number;
  calculationDate: string;
}

export default function App() {
  const [language, setLanguage] = useState<Language>("en");
  const [plotData, setPlotData] = useState<PlotData[]>([]);
  const [growthResults, setGrowthResults] = useState<GrowthResult[]>([]);
  const [activeTab, setActiveTab] = useState(() => {
    if (typeof window === "undefined") return "data-entry";
    const saved = window.localStorage.getItem(APP_ACTIVE_TAB_KEY);
    return saved && VALID_TABS.has(saved) ? saved : "data-entry";
  });
  const [selectedModelGroup, setSelectedModelGroup] = useState(() => {
    if (typeof window === "undefined") return "HardMaple";
    return window.localStorage.getItem(APP_MODEL_GROUP_KEY) || "HardMaple";
  });

  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof window === "undefined") return "light";

    const saved = window.localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;

    const prefersDark = window.matchMedia?.(
      "(prefers-color-scheme: dark)",
    )?.matches;
    return prefersDark ? "dark" : "light";
  });

  useEffect(() => {
    if (typeof document === "undefined") return;
    document.documentElement.classList.toggle("dark", theme === "dark");
    window.localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!VALID_TABS.has(activeTab)) return;
    window.localStorage.setItem(APP_ACTIVE_TAB_KEY, activeTab);
  }, [activeTab]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!selectedModelGroup) return;
    window.localStorage.setItem(APP_MODEL_GROUP_KEY, selectedModelGroup);
  }, [selectedModelGroup]);

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));
  const toggleLanguage = () =>
    setLanguage((prev) => (prev === "en" ? "fr" : "en"));

  const translations = {
    en: {
      title: "Basal Area Growth Rate Tool",
      subtitle: "Forest Growth & Yield Analytics System (Public Demo)",
      dataEntry: "Data Entry",
      calculations: "Calculations",
      visualization: "Visualization",
      reports: "Reports",
      languageToggle: "Français",
      themeToggle: "Theme",
      coefficients: "Model Coefficients",
      baGrowthCalc: "BA Growth Calculation",
      benchmarks: "Model Benchmarks",
    },
    fr: {
      title: "Outil de Taux de Croissance de Surface Terrière",
      subtitle: "Système d'analyse de croissance et de rendement forestier (démo publique)",
      dataEntry: "Saisie de Données",
      calculations: "Calculs",
      visualization: "Visualisation",
      reports: "Rapports",
      languageToggle: "English",
      themeToggle: "Thème",
      coefficients: "Coefficients du Modèle",
      baGrowthCalc: "Calcul de Croissance ST",
      benchmarks: "Comparaison des Modèles",
    },
  } as const;

  const t = translations[language];

  return (
    <div
      className="
        min-h-screen
        bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50
        dark:from-slate-950 dark:via-slate-900 dark:to-slate-950
        text-foreground
      "
    >
      <Header
        language={language}
        toggleLanguage={toggleLanguage}
        title={t.title}
        subtitle={t.subtitle}
        languageToggle={t.languageToggle}
        theme={theme}
        toggleTheme={toggleTheme}
      />

      <main className="container mx-auto px-4 py-8 max-w-7xl">
        <Card className="shadow-xl border-border bg-card text-card-foreground">
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="w-full"
          >
            <TabsList className="grid h-auto w-full grid-cols-2 gap-1 bg-green-100 p-1 sm:grid-cols-4 lg:grid-cols-7 dark:bg-slate-800">
              <TabsTrigger
                value="data-entry"
                className="flex items-center justify-center gap-2 px-2 py-2 text-xs sm:text-sm"
              >
                <Database className="w-4 h-4" />
                {t.dataEntry}
              </TabsTrigger>

              <TabsTrigger
                value="calculations"
                className="flex items-center justify-center gap-2 px-2 py-2 text-xs sm:text-sm"
              >
                <Calculator className="w-4 h-4" />
                {t.calculations}
              </TabsTrigger>

              <TabsTrigger
                value="ba-growth"
                className="flex items-center justify-center gap-2 px-2 py-2 text-xs sm:text-sm"
              >
                <Zap className="w-4 h-4" />
                {t.baGrowthCalc}
              </TabsTrigger>

              <TabsTrigger
                value="coefficients"
                className="flex items-center justify-center gap-2 px-2 py-2 text-xs sm:text-sm"
              >
                <TrendingUp className="w-4 h-4" />
                {t.coefficients}
              </TabsTrigger>
              <TabsTrigger
                value="benchmarks"
                className="flex items-center justify-center gap-2 px-2 py-2 text-xs sm:text-sm"
              >
                <FlaskConical className="w-4 h-4" />
                {t.benchmarks}
              </TabsTrigger>
              <TabsTrigger
                value="visualization"
                className="flex items-center justify-center gap-2 px-2 py-2 text-xs sm:text-sm"
              >
                <BarChart3 className="w-4 h-4" />
                {t.visualization}
              </TabsTrigger>

              <TabsTrigger value="reports" className="flex items-center justify-center gap-2 px-2 py-2 text-xs sm:text-sm">
                <FileText className="w-4 h-4" />
                {t.reports}
              </TabsTrigger>
            </TabsList>

            <TabsContent value="data-entry" className="p-6" forceMount>
              <DataEntry
                language={language}
                plotData={plotData}
                setPlotData={setPlotData}
              />
            </TabsContent>

            <TabsContent value="calculations" className="p-6" forceMount>
              <GrowthCalculator
                language={language}
                plotData={plotData}
                growthResults={growthResults}
                setGrowthResults={setGrowthResults}
              />
            </TabsContent>

            <TabsContent value="ba-growth" className="p-6" forceMount>
              <BAGrowthCalculator
                language={language}
                selectedModelGroup={selectedModelGroup}
                onModelGroupChange={setSelectedModelGroup}
              />
            </TabsContent>

            <TabsContent value="coefficients" className="p-6" forceMount>
              <CoefficientDisplay
                language={language}
                selectedModelGroup={selectedModelGroup}
                onModelGroupChange={setSelectedModelGroup}
              />
            </TabsContent>

            <TabsContent value="benchmarks" className="p-6">
              <ModelBenchmarks language={language} />
            </TabsContent>

            <TabsContent value="visualization" className="p-6" forceMount>
              <Visualization
                language={language}
                plotData={plotData}
                growthResults={growthResults}
              />
            </TabsContent>

            <TabsContent value="reports" className="p-6" forceMount>
              <Reports
                language={language}
                plotData={plotData}
                growthResults={growthResults}
              />
            </TabsContent>
          </Tabs>
        </Card>
      </main>
    </div>
  );
}
