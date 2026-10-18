import { useState } from "react";
import { Button } from "./ui/button";
import { Database, CheckCircle, XCircle, Loader } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import type { PlotData, Language } from "../App";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

interface DataEntryProps {
  language: Language;
  plotData: PlotData[];
  setPlotData: (data: PlotData[]) => void;
}

export function DataEntry({
  language,
  plotData: _plotData,
  setPlotData: _setPlotData,
}: DataEntryProps) {
  // Database connection state
  const [isConnected, setIsConnected] = useState(false);
  const [connectionData, setConnectionData] = useState<any>(null);
  const [dbLoading, setDbLoading] = useState(false);
  const [dbError, setDbError] = useState<string | null>(null);
  const [tables, setTables] = useState<string[]>([]);

  const translations = {
    en: {
      title: "PSP Data Entry",
      description:
        "Enter Permanent Sample Plot data for shade-tolerant hardwood forests",
      plotId: "Plot ID",
      location: "Location",
      measurementDate: "Measurement Date",
      species: "Species",
      dbh: "DBH (cm)",
      basalArea: "Basal Area (m²/ha)",
      age: "Age (years)",
      siteIndex: "Site Index",
      agsFrac: "AGS Fraction",
      tcr: "TCR",
      dq: "DQ (cm)",
      qFactor: "Q-Factor",
      cut: "Cutting",
      noCutting: "No Cutting",
      cutting: "Cutting",
      addPlot: "Add Plot",
      currentData: "Current Plot Data",
      actions: "Actions",
      delete: "Delete",
      importData: "Import CSV",
      exportData: "Export CSV",
      selectSpecies: "Select species",
      sugarMaple: "Sugar Maple",
      yellowBirch: "Yellow Birch",
      americanBeech: "American Beech",
      redMaple: "Red Maple",
      easternHemlock: "Eastern Hemlock",
      basicInfo: "Basic Information",
      growthParameters: "Growth Model Parameters",
      dbConnection: "Database Connection",
      connectDB: "Connect to Database",
      connecting: "Connecting...",
      connected: "Connected to demo database",
      fetchResults: "Fetch Results",
      fetching: "Fetching...",
      availableTables: "Available Tables",
    },
    fr: {
      title: "Saisie de Données PSP",
      description:
        "Entrez les données des placettes échantillons permanentes pour les forêts de feuillus tolérants à l'ombre",
      plotId: "ID de Placette",
      location: "Emplacement",
      measurementDate: "Date de Mesure",
      species: "Espèce",
      dbh: "DHP (cm)",
      basalArea: "Surface Terrière (m²/ha)",
      age: "Âge (années)",
      siteIndex: "Indice de Site",
      agsFrac: "Fraction AGS",
      tcr: "TCR",
      dq: "DQ (cm)",
      qFactor: "Facteur Q",
      cut: "Coupe",
      noCutting: "Pas de Coupe",
      cutting: "Coupe",
      addPlot: "Ajouter une Placette",
      currentData: "Données Actuelles des Placettes",
      actions: "Actions",
      delete: "Supprimer",
      importData: "Importer CSV",
      exportData: "Exporter CSV",
      selectSpecies: "Sélectionner l'espèce",
      sugarMaple: "Érable à Sucre",
      yellowBirch: "Bouleau Jaune",
      americanBeech: "Hêtre à Grandes Feuilles",
      redMaple: "Érable Rouge",
      easternHemlock: "Pruche de l'Est",
      basicInfo: "Informations de Base",
      growthParameters: "Paramètres du Modèle de Croissance",
      dbConnection: "Connexion à la Base de Données",
      connectDB: "Se Connecter à la Base",
      connecting: "Connexion...",
      connected: "Connecté à la base de démonstration",
      fetchResults: "Récupérer les Résultats",
      fetching: "Récupération...",
      availableTables: "Tables Disponibles",
    },
  };

  const t = translations[language];

  const connectToDatabase = async () => {
    setDbLoading(true);
    setDbError(null);
    try {
      const response = await fetch(`${API_BASE}/test-connection`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const detail =
          errorData && typeof errorData.detail === "string"
            ? errorData.detail
            : "Connection failed";
        throw new Error(detail);
      }
      const data = await response.json();
      setConnectionData(data);
      setIsConnected(true);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Failed to connect to backend or database.";
      setDbError(message);
      setIsConnected(false);
    }
    setDbLoading(false);
  };

  const fetchTables = async () => {
    setDbLoading(true);
    setDbError(null);
    try {
      const response = await fetch(`${API_BASE}/tables`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const detail =
          errorData && typeof errorData.detail === "string"
            ? errorData.detail
            : "Failed to fetch tables";
        throw new Error(detail);
      }
      const data = await response.json();
      setTables(data.tables);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to fetch tables";
      setDbError(message);
      setTables([]);
    }
    setDbLoading(false);
  };

  return (
    <div className="space-y-6">
      {/* Database Connection Card */}
      <Card className="border-border/60 bg-card text-card-foreground shadow-lg shadow-black/5 dark:shadow-black/30">
        <CardHeader className="rounded-t-lg border-b border-border/60 bg-gradient-to-r from-emerald-50/80 via-green-50/80 to-teal-50/80 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800">
          <CardTitle className="text-foreground flex items-center gap-2">
            {t.dbConnection}
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          {!isConnected ? (
            <Button
              onClick={connectToDatabase}
              disabled={dbLoading}
              className="bg-green-700 hover:bg-green-800 dark:bg-green-600 dark:hover:bg-green-500"
            >
              {dbLoading ? (
                <>
                  <Loader className="w-4 h-4 mr-2 animate-spin" />
                  {t.connecting}
                </>
              ) : (
                <>
                  <Database className="w-4 h-4 mr-2" />
                  {t.connectDB}
                </>
              )}
            </Button>
          ) : (
            <div className="space-y-4">
              {/* Success Message */}
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-600" />
                  <span className="font-semibold text-green-800">
                    {connectionData?.message || t.connected}
                  </span>
                </div>
              </div>

              {/* Fetch Results Button */}
              <Button
                onClick={fetchTables}
                disabled={dbLoading}
                className="bg-green-700 hover:bg-green-800 dark:bg-green-600 dark:hover:bg-green-500"
              >
                {dbLoading ? (
                  <>
                    <Loader className="w-4 h-4 mr-2 animate-spin" />
                    {t.fetching}
                  </>
                ) : (
                  t.fetchResults
                )}
              </Button>
            </div>
          )}

          {/* Error Display */}
          {dbError && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-center gap-2">
                <XCircle className="w-5 h-5 text-red-600" />
                <span className="font-semibold text-red-800">{dbError}</span>
              </div>
            </div>
          )}

          {/* Tables Display */}
          {tables.length > 0 && (
            <div className="mt-4">
              <h4 className="text-green-900 dark:text-green-400 font-semibold mb-3">
                {t.availableTables}
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {tables.map((table, index) => (
                  <div
                    key={index}
                    className="p-3 bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-800 rounded-lg"
                  >
                    <span className="text-green-800 dark:text-green-300 font-medium">
                      {table}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
