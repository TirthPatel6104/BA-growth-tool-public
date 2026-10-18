import { useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
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
import { Calculator, Info, PlayCircle, BookOpen } from "lucide-react";
import { Badge } from "./ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import type { PlotData, GrowthResult, Language } from "../App";
//import equationImage from 'figma:asset/8dd2e08d844966df79213c8d56079e32e932da8a.png';

interface GrowthCalculatorProps {
  language: Language;
  plotData: PlotData[];
  growthResults: GrowthResult[];
  setGrowthResults: (results: GrowthResult[]) => void;
}

interface DetailedGrowthResult extends GrowthResult {
  annualBAGrowth: number;
  relativeGrowthRate: number;
  futureBA: number;
}

export function GrowthCalculator({
  language,
  plotData,
  growthResults: _growthResults,
  setGrowthResults,
}: GrowthCalculatorProps) {
  const [projectionYears, setProjectionYears] = useState<number>(10);
  const [isCalculating, setIsCalculating] = useState(false);
  const [detailedResults, setDetailedResults] = useState<
    DetailedGrowthResult[]
  >([]);
  const [selectedModel, setSelectedModel] = useState<"equation4" | "equation9">(
    "equation4",
  );

  const translations = {
    en: {
      title: "Growth Equation Processor",
      description:
        "Apply research-based models for basal area growth estimation",
      projectionYears: "Projection Period (years)",
      calculate: "Calculate Growth",
      calculating: "Calculating...",
      noData: "No plot data available. Please enter plot data first.",
      equation4: "Equation 4: BA Growth",
      equation9: "Equation 9: Relative Growth Rate (RGR)",
      equationDesc4:
        "BAgrowth = β₀ + β₁·BA + β₂·BA² + β₃·AGSfrac + β₄·TCR + β₅·DQ + β₆·q + β₇·Cut",
      equationDesc9:
        "RGR = (0.07292 - 0.00178·BA + 0.01464·AGSfrac + 0.00217·TCR - 0.00093·DQ + 0.00404·q - 0.00974·Cut)·(1/BA)",
      results: "Calculation Results",
      plotId: "Plot ID",
      currentBA: "Current BA",
      annualGrowth: "Annual Growth",
      totalGrowth: "Total Growth",
      futureBA: "Future BA",
      rgr: "RGR",
      volumeEstimate: "Volume (m³/ha)",
      calculationDate: "Calculated",
      modelReference: "Model Reference",
      note: "Note: Coefficients shown are for demonstration. Actual coefficients should be calibrated using PSP data.",
    },
    fr: {
      title: "Processeur d'Équations de Croissance",
      description:
        "Appliquer des modèles basés sur la recherche pour l'estimation de la croissance de la surface terrière",
      projectionYears: "Période de Projection (années)",
      calculate: "Calculer la Croissance",
      calculating: "Calcul en cours...",
      noData:
        "Aucune donnée de placette disponible. Veuillez d'abord entrer les données de placette.",
      equation4: "Équation 4: Croissance ST",
      equation9: "Équation 9: Taux de Croissance Relatif (TCR)",
      equationDesc4:
        "Croissance ST = β₀ + β₁·ST + β₂·ST² + β₃·FractionAGS + β₄·TCR + β₅·DQ + β₆·q + β₇·Coupe",
      equationDesc9:
        "TCR = (0.07292 - 0.00178·ST + 0.01464·FractionAGS + 0.00217·TCR - 0.00093·DQ + 0.00404·q - 0.00974·Coupe)·(1/ST)",
      results: "Résultats de Calcul",
      plotId: "ID Placette",
      currentBA: "ST Actuelle",
      annualGrowth: "Croissance Annuelle",
      totalGrowth: "Croissance Totale",
      futureBA: "ST Future",
      rgr: "TCR",
      volumeEstimate: "Volume (m³/ha)",
      calculationDate: "Calculé",
      modelReference: "Référence du Modèle",
      note: "Note: Les coefficients affichés sont à des fins de démonstration. Les coefficients réels doivent être calibrés à l'aide des données PSP.",
    },
  };

  const t = translations[language];

  // Equation 4 coefficients (example values - should be calibrated with actual PSP data)
  const equation4Coefficients = {
    beta0: 1.8543,
    beta1: -0.0285,
    beta2: 0.00018,
    beta3: 0.5234,
    beta4: 0.0312,
    beta5: 0.0087,
    beta6: 0.0456,
    beta7: -0.1234,
  };

  const calculateEquation4 = (plot: PlotData): number => {
    // Equation 4: BAgrowth = β₀ + β₁·BA + β₂·BA² + β₃·AGSfrac + β₄·TCR + β₅·DQ + β₆·q + β₇·Cut
    const { basalArea, agsFrac, tcr, dq, qFactor, cut } = plot;

    const annualGrowth =
      equation4Coefficients.beta0 +
      equation4Coefficients.beta1 * basalArea +
      equation4Coefficients.beta2 * Math.pow(basalArea, 2) +
      equation4Coefficients.beta3 * agsFrac +
      equation4Coefficients.beta4 * tcr +
      equation4Coefficients.beta5 * dq +
      equation4Coefficients.beta6 * qFactor +
      equation4Coefficients.beta7 * cut;

    return Math.max(0, annualGrowth); // Ensure non-negative growth
  };

  const calculateEquation9 = (plot: PlotData): number => {
    // Equation 9: RGR = (0.07292 - 0.00178·BA + 0.01464·AGSfrac + 0.00217·TCR - 0.00093·DQ + 0.00404·q - 0.00974·Cut)·(1/BA)
    const { basalArea, agsFrac, tcr, dq, qFactor, cut } = plot;

    if (basalArea === 0) return 0; // Avoid division by zero

    const rgr =
      (0.07292 -
        0.00178 * basalArea +
        0.01464 * agsFrac +
        0.00217 * tcr -
        0.00093 * dq +
        0.00404 * qFactor -
        0.00974 * cut) *
      (1 / basalArea);

    // Annual growth from RGR: BAgrowth = RGR × BA
    const annualGrowth = rgr * basalArea;

    return Math.max(0, annualGrowth);
  };

  const calculateGrowth = () => {
    setIsCalculating(true);

    setTimeout(() => {
      const results: DetailedGrowthResult[] = plotData.map((plot) => {
        let annualGrowth: number;
        let rgr: number = 0;

        if (selectedModel === "equation4") {
          annualGrowth = calculateEquation4(plot);
          // Calculate RGR from annual growth for display
          if (plot.basalArea > 0) {
            rgr = annualGrowth / plot.basalArea;
          }
        } else {
          annualGrowth = calculateEquation9(plot);
          // RGR is already calculated in equation 9
          if (plot.basalArea > 0) {
            rgr = annualGrowth / plot.basalArea;
          }
        }

        const totalGrowth = annualGrowth * projectionYears;
        const futureBA = plot.basalArea + totalGrowth;

        // Volume estimation (simplified model based on basal area and site index)
        const volumeEstimate = futureBA * plot.siteIndex * 0.52;

        return {
          plotId: plot.plotId,
          predictedGrowth: totalGrowth,
          volumeEstimate: Math.max(0, volumeEstimate),
          projectionYears,
          calculationDate: new Date().toISOString().split("T")[0],
          annualBAGrowth: annualGrowth,
          relativeGrowthRate: rgr,
          futureBA: futureBA,
        };
      });

      setDetailedResults(results);
      setGrowthResults(results);
      setIsCalculating(false);
    }, 800);
  };

  if (plotData.length === 0) {
    return (
      <Alert className="border-amber-200 bg-amber-50">
        <Info className="h-4 w-4 text-amber-600" />
        <AlertDescription className="text-amber-800">
          {t.noData}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="border-green-200">
        <CardHeader className="bg-green-50">
          <CardTitle className="text-green-900 flex items-center gap-2">
            <Calculator className="w-5 h-5" />
            {t.title}
          </CardTitle>
          <CardDescription>{t.description}</CardDescription>
        </CardHeader>
        <CardContent className="pt-6 space-y-6">
          <Tabs
            value={selectedModel}
            onValueChange={(value) =>
              setSelectedModel(value as "equation4" | "equation9")
            }
          >
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="equation4">{t.equation4}</TabsTrigger>
              <TabsTrigger value="equation9">{t.equation9}</TabsTrigger>
            </TabsList>

            <TabsContent value="equation4" className="space-y-4">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h3 className="text-blue-900 mb-2">{t.equation4}</h3>
                <p className="font-mono text-sm text-blue-800 mb-3">
                  {t.equationDesc4}
                </p>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  <Badge variant="outline" className="bg-white">
                    β₀ = {equation4Coefficients.beta0}
                  </Badge>
                  <Badge variant="outline" className="bg-white">
                    β₁ = {equation4Coefficients.beta1}
                  </Badge>
                  <Badge variant="outline" className="bg-white">
                    β₂ = {equation4Coefficients.beta2}
                  </Badge>
                  <Badge variant="outline" className="bg-white">
                    β₃ = {equation4Coefficients.beta3}
                  </Badge>
                  <Badge variant="outline" className="bg-white">
                    β₄ = {equation4Coefficients.beta4}
                  </Badge>
                  <Badge variant="outline" className="bg-white">
                    β₅ = {equation4Coefficients.beta5}
                  </Badge>
                  <Badge variant="outline" className="bg-white">
                    β₆ = {equation4Coefficients.beta6}
                  </Badge>
                  <Badge variant="outline" className="bg-white">
                    β₇ = {equation4Coefficients.beta7}
                  </Badge>
                </div>

                <p className="text-xs text-blue-700 mt-3 italic">{t.note}</p>
              </div>
            </TabsContent>

            <TabsContent value="equation9" className="space-y-4">
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <h3 className="text-purple-900 mb-2">{t.equation9}</h3>
                <p className="font-mono text-sm text-purple-800 mb-3">
                  {t.equationDesc9}
                </p>

                <div className="bg-white p-3 rounded border border-purple-200">
                  <p className="text-xs text-purple-800">
                    <strong>Fixed Coefficients (OMNR 2000):</strong>
                    <br />
                    0.07292, -0.00178, 0.01464, 0.00217, -0.00093, 0.00404,
                    -0.00974
                  </p>
                </div>

                <p className="text-xs text-purple-700 mt-3 italic">
                  {language === "en"
                    ? "This equation is recommended for typical stand BAs (15-35 m²/ha) based on OMNR hardwood silviculture guide."
                    : "Cette équation est recommandée pour les surfaces terrières de peuplement typiques (15-35 m²/ha) selon le guide de sylviculture des feuillus du MRNFO."}
                </p>
              </div>
            </TabsContent>
          </Tabs>

          <div className="flex items-end gap-4">
            <div className="flex-1 max-w-xs space-y-2">
              <Label htmlFor="years">{t.projectionYears}</Label>
              <Input
                id="years"
                type="number"
                value={projectionYears}
                onChange={(e) =>
                  setProjectionYears(parseInt(e.target.value) || 0)
                }
                min="1"
                max="50"
              />
            </div>
            <Button
              onClick={calculateGrowth}
              disabled={isCalculating}
              className="bg-green-700 hover:bg-green-800"
            >
              <PlayCircle className="w-4 h-4 mr-2" />
              {isCalculating ? t.calculating : t.calculate}
            </Button>
          </div>
        </CardContent>
      </Card>

      {detailedResults.length > 0 && (
        <Card className="border-green-200">
          <CardHeader className="bg-green-50">
            <CardTitle className="text-green-900">{t.results}</CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="rounded-md border overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="bg-green-50">
                    <TableHead>{t.plotId}</TableHead>
                    <TableHead>{t.currentBA}</TableHead>
                    <TableHead>{t.annualGrowth}</TableHead>
                    <TableHead>{t.totalGrowth}</TableHead>
                    <TableHead>{t.futureBA}</TableHead>
                    <TableHead>{t.rgr}</TableHead>
                    <TableHead>{t.volumeEstimate}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {detailedResults.map((result) => {
                    const plot = plotData.find(
                      (p) => p.plotId === result.plotId,
                    );
                    return (
                      <TableRow key={result.plotId}>
                        <TableCell>{result.plotId}</TableCell>
                        <TableCell>
                          {plot?.basalArea.toFixed(2)} m²/ha
                        </TableCell>
                        <TableCell className="text-blue-700">
                          {result.annualBAGrowth.toFixed(3)} m²/ha/yr
                        </TableCell>
                        <TableCell className="text-green-700">
                          +{result.predictedGrowth.toFixed(2)} m²/ha
                        </TableCell>
                        <TableCell>
                          {result.futureBA.toFixed(2)} m²/ha
                        </TableCell>
                        <TableCell>
                          {result.relativeGrowthRate.toFixed(4)}
                        </TableCell>
                        <TableCell>
                          {result.volumeEstimate.toFixed(2)} m³/ha
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Model Reference Image */}
      <Card className="border-blue-200">
        <CardHeader className="bg-blue-50">
          <CardTitle className="text-blue-900 flex items-center gap-2">
            <BookOpen className="w-5 h-5" />
            {t.modelReference}
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-6"></CardContent>
      </Card>
    </div>
  );
}
