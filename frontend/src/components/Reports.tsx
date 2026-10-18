import { useEffect, useState } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Alert, AlertDescription } from './ui/alert';
import { FileText, Download, Info, FileSpreadsheet, Printer, AlertTriangle } from 'lucide-react';
import { Badge } from './ui/badge';
import type { PlotData, GrowthResult, Language } from '../App';

interface ReportsProps {
  language: Language;
  plotData: PlotData[];
  growthResults: GrowthResult[];
}

export function Reports({ language, plotData, growthResults }: ReportsProps) {
  const [trainingResults, setTrainingResults] = useState<Record<string, any> | null>(null);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  const translations = {
    en: {
      title: 'Report Generation',
      description: 'Export analysis results and generate comprehensive reports',
      noData: 'No data available to generate reports. Please enter plot data and calculate growth first.',
      summary: 'Analysis Summary',
      totalPlots: 'Total Plots',
      avgBasalArea: 'Avg Basal Area',
      avgGrowth: 'Avg Growth',
      totalVolume: 'Total Volume',
      exportPDF: 'Export PDF Report',
      exportExcel: 'Export to Excel',
      printReport: 'Print Report',
      reportTypes: 'Available Report Types',
      fullAnalysis: 'Full Analysis Report',
      fullAnalysisDesc: 'Comprehensive report including all plots, calculations, and visualizations',
      summaryReport: 'Summary Report',
      summaryReportDesc: 'Executive summary with key findings and recommendations',
      technicalReport: 'Technical Report',
      technicalReportDesc: 'Detailed technical documentation with equations and coefficients',
      plots: 'plots',
      generated: 'Report generated successfully!',
      trainingReport: 'Training Report',
      trainingReportDesc: 'Latest model training metrics and diagnostics',
      trainingStatus: 'Training Status',
      qualityFlags: 'Quality Flags',
      warnings: 'Warnings',
      model: 'Model',
      rSquared: 'R²',
      rmse: 'RMSE',
      mae: 'MAE',
      samples: 'Samples',
      plotsUsed: 'Plots',
      confidence: 'Confidence',
      productionReady: 'Production Ready',
      trained: 'Trained',
      review: 'Review',
      noTrainingReport: 'No training report available yet.',
      loadingTrainingReport: 'Loading training report...',
    },
    fr: {
      title: 'Génération de Rapports',
      description: 'Exporter les résultats d\'analyse et générer des rapports complets',
      noData: 'Aucune donnée disponible pour générer des rapports. Veuillez d\'abord entrer les données de placette et calculer la croissance.',
      summary: 'Résumé de l\'Analyse',
      totalPlots: 'Total Placettes',
      avgBasalArea: 'ST Moyenne',
      avgGrowth: 'Croissance Moy.',
      totalVolume: 'Volume Total',
      exportPDF: 'Exporter Rapport PDF',
      exportExcel: 'Exporter vers Excel',
      printReport: 'Imprimer Rapport',
      reportTypes: 'Types de Rapports Disponibles',
      fullAnalysis: 'Rapport d\'Analyse Complet',
      fullAnalysisDesc: 'Rapport complet incluant toutes les placettes, calculs et visualisations',
      summaryReport: 'Rapport Sommaire',
      summaryReportDesc: 'Résumé exécutif avec conclusions clés et recommandations',
      technicalReport: 'Rapport Technique',
      technicalReportDesc: 'Documentation technique détaillée avec équations et coefficients',
      plots: 'placettes',
      generated: 'Rapport généré avec succès!',
      trainingReport: 'Rapport d\'Entraînement',
      trainingReportDesc: 'Derniers indicateurs et diagnostics d\'entraînement du modèle',
      trainingStatus: 'Statut d\'Entraînement',
      qualityFlags: 'Indicateurs de Qualité',
      warnings: 'Avertissements',
      model: 'Modèle',
      rSquared: 'R²',
      rmse: 'RMSE',
      mae: 'MAE',
      samples: 'Échantillons',
      plotsUsed: 'Placettes',
      confidence: 'Confiance',
      productionReady: 'Prêt pour production',
      trained: 'Entraîné',
      review: 'Vérifier',
      noTrainingReport: 'Aucun rapport d\'entraînement disponible pour le moment.',
      loadingTrainingReport: 'Chargement du rapport d\'entraînement...',
    }
  };

  const t = translations[language];

  useEffect(() => {
    const fetchTrainingReport = async () => {
      setTrainingLoading(true);
      setTrainingError(null);
      try {
        const response = await fetch(`${API_BASE}/models/training-results`);
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || `HTTP ${response.status}`);
        }
        const data = await response.json();
        setTrainingResults(data.training_results || null);
      } catch (err) {
        setTrainingError(err instanceof Error ? err.message : 'Failed to load training report');
      } finally {
        setTrainingLoading(false);
      }
    };

    fetchTrainingReport();
  }, [API_BASE]);

  const handleExportPDF = () => {
    // Simulate PDF generation
    alert(t.generated);
  };

  const handleExportExcel = () => {
    if (plotData.length === 0) return;

    // Create CSV data
    const headers = ['Plot ID', 'Location', 'Species', 'Current BA', 'Projected Growth', 'Future BA', 'Volume Estimate'];
    const rows = plotData.map(plot => {
      const result = growthResults.find(r => r.plotId === plot.plotId);
      return [
        plot.plotId,
        plot.location,
        plot.species,
        plot.basalArea.toFixed(2),
        result ? result.predictedGrowth.toFixed(2) : 'N/A',
        result ? (plot.basalArea + result.predictedGrowth).toFixed(2) : 'N/A',
        result ? result.volumeEstimate.toFixed(2) : 'N/A'
      ];
    });

    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'growth_analysis_report.csv';
    a.click();
  };

  const handlePrint = () => {
    window.print();
  };

  const noPlotData = plotData.length === 0;

  // Calculate summary statistics
  const totalPlots = plotData.length;
  const avgBasalArea = totalPlots > 0 ? plotData.reduce((sum, p) => sum + p.basalArea, 0) / totalPlots : 0;
  const avgGrowth = growthResults.length > 0 
    ? growthResults.reduce((sum, r) => sum + r.predictedGrowth, 0) / growthResults.length 
    : 0;
  const totalVolume = growthResults.reduce((sum, r) => sum + r.volumeEstimate, 0);

  return (
    <div className="space-y-6">
      {noPlotData && (
        <Alert className="border-amber-200 bg-amber-50">
          <Info className="h-4 w-4 text-amber-600" />
          <AlertDescription className="text-amber-800">
            {t.noData}
          </AlertDescription>
        </Alert>
      )}
      <Card className="border-green-200">
        <CardHeader className="bg-green-50">
          <CardTitle className="text-green-900 flex items-center gap-2">
            <FileText className="w-5 h-5" />
            {t.title}
          </CardTitle>
          <CardDescription>{t.description}</CardDescription>
        </CardHeader>
        <CardContent className="pt-6 space-y-6">
          {/* Summary Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200">
              <p className="text-sm text-green-700">{t.totalPlots}</p>
              <p className="text-green-900 mt-1">
                {totalPlots} {t.plots}
              </p>
            </div>
            <div className="bg-gradient-to-br from-blue-50 to-cyan-50 p-4 rounded-lg border border-blue-200">
              <p className="text-sm text-blue-700">{t.avgBasalArea}</p>
              <p className="text-blue-900 mt-1">
                {avgBasalArea.toFixed(2)} m²/ha
              </p>
            </div>
            <div className="bg-gradient-to-br from-emerald-50 to-teal-50 p-4 rounded-lg border border-emerald-200">
              <p className="text-sm text-emerald-700">{t.avgGrowth}</p>
              <p className="text-emerald-900 mt-1">
                {avgGrowth.toFixed(2)} m²/ha
              </p>
            </div>
            <div className="bg-gradient-to-br from-teal-50 to-cyan-50 p-4 rounded-lg border border-teal-200">
              <p className="text-sm text-teal-700">{t.totalVolume}</p>
              <p className="text-teal-900 mt-1">
                {totalVolume.toFixed(2)} m³/ha
              </p>
            </div>
          </div>

          {/* Export Buttons */}
          <div className="flex flex-wrap gap-3">
            <Button onClick={handleExportPDF} className="bg-green-700 hover:bg-green-800">
              <FileText className="w-4 h-4 mr-2" />
              {t.exportPDF}
            </Button>
            <Button onClick={handleExportExcel} variant="outline" className="border-green-700 text-green-700 hover:bg-green-50">
              <FileSpreadsheet className="w-4 h-4 mr-2" />
              {t.exportExcel}
            </Button>
            <Button onClick={handlePrint} variant="outline" className="border-green-700 text-green-700 hover:bg-green-50">
              <Printer className="w-4 h-4 mr-2" />
              {t.printReport}
            </Button>
          </div>

          {/* Report Types */}
          <div>
            <h3 className="text-green-900 mb-4">{t.reportTypes}</h3>
            <div className="space-y-3">
              <div className="flex items-start gap-3 p-4 bg-white border border-green-200 rounded-lg hover:shadow-md transition-shadow">
                <FileText className="w-5 h-5 text-green-700 mt-1" />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h4 className="text-green-900">{t.fullAnalysis}</h4>
                    <Badge variant="outline" className="bg-green-50 text-green-700 border-green-300">
                      Comprehensive
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{t.fullAnalysisDesc}</p>
                </div>
                <Button size="sm" onClick={handleExportPDF} className="bg-green-700 hover:bg-green-800">
                  <Download className="w-4 h-4" />
                </Button>
              </div>

              <div className="flex items-start gap-3 p-4 bg-white border border-green-200 rounded-lg hover:shadow-md transition-shadow">
                <FileText className="w-5 h-5 text-blue-700 mt-1" />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h4 className="text-green-900">{t.summaryReport}</h4>
                    <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-300">
                      Executive
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{t.summaryReportDesc}</p>
                </div>
                <Button size="sm" onClick={handleExportPDF} className="bg-green-700 hover:bg-green-800">
                  <Download className="w-4 h-4" />
                </Button>
              </div>

              <div className="flex items-start gap-3 p-4 bg-white border border-green-200 rounded-lg hover:shadow-md transition-shadow">
                <FileText className="w-5 h-5 text-purple-700 mt-1" />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h4 className="text-green-900">{t.technicalReport}</h4>
                    <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-300">
                      Technical
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{t.technicalReportDesc}</p>
                </div>
                <Button size="sm" onClick={handleExportPDF} className="bg-green-700 hover:bg-green-800">
                  <Download className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Training Report Preview */}
      <Card className="border-green-200 print:shadow-none">
        <CardHeader className="bg-slate-50">
          <CardTitle className="text-green-900">{t.trainingReport}</CardTitle>
          <CardDescription className="text-sm text-gray-500">
            {t.trainingReportDesc}
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6 space-y-4">
          {trainingLoading && (
            <div className="text-sm text-gray-700">{t.loadingTrainingReport}</div>
          )}

          {trainingError && (
            <Alert className="border-rose-200 bg-rose-50">
              <AlertTriangle className="h-4 w-4 text-rose-600" />
              <AlertDescription className="text-rose-800">
                {trainingError}
              </AlertDescription>
            </Alert>
          )}

          {!trainingLoading && !trainingError && !trainingResults && (
            <Alert className="border-amber-200 bg-amber-50">
              <Info className="h-4 w-4 text-amber-600" />
              <AlertDescription className="text-amber-800">
                {t.noTrainingReport}
              </AlertDescription>
            </Alert>
          )}

          {!trainingLoading && trainingResults && (
            <div className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {Object.entries(trainingResults).map(([modelName, modelData]) => (
                  <div key={modelName} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <p className="text-sm font-semibold text-slate-900">{t.model}: {modelName}</p>
                        <p className="text-xs text-slate-500">{modelData.confidence_level || 'N/A'}</p>
                      </div>
                      <Badge variant={modelData.is_production_ready ? 'secondary' : 'outline'}>
                        {modelData.is_production_ready ? t.trained : t.review}
                      </Badge>
                    </div>

                    <div className="grid gap-2 text-sm text-slate-700">
                      <div className="flex justify-between">
                        <span>{t.rSquared}</span>
                        <span>{modelData.r_squared?.toFixed(3) ?? 'N/A'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>{t.rmse}</span>
                        <span>{modelData.rmse?.toFixed(4) ?? 'N/A'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>{t.mae}</span>
                        <span>{modelData.mae?.toFixed(4) ?? 'N/A'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>{t.samples}</span>
                        <span>{modelData.n_samples ?? 'N/A'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>{t.plotsUsed}</span>
                        <span>{modelData.n_plots ?? 'N/A'}</span>
                      </div>
                    </div>

                    {modelData.warning_message && (
                      <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                        <div className="font-semibold">{t.warnings}</div>
                        <p>{modelData.warning_message}</p>
                      </div>
                    )}

                    {modelData.quality_flags?.length > 0 && (
                      <div className="mt-4 space-y-1">
                        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">{t.qualityFlags}</div>
                        <div className="flex flex-wrap gap-2">
                          {modelData.quality_flags.map((flag: string) => (
                            <Badge key={flag} variant="outline">{flag}</Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Report Preview */}
      {growthResults.length > 0 && (
        <Card className="border-green-200 print:shadow-none">
          <CardHeader className="bg-green-50 print:bg-white">
            <CardTitle className="text-green-900">{t.summary}</CardTitle>
            <CardDescription className="text-sm text-gray-500">
              Forest Growth & Yield Analytics System (Public Demo)
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="space-y-4">
              <div className="prose max-w-none">
                <h3 className="text-green-900">Analysis Overview</h3>
                <p>
                  This report presents basal area growth analysis for {totalPlots} permanent sample plots
                  in shade-tolerant hardwood forests. The analysis uses Equation 9 to estimate growth
                  over a {growthResults[0]?.projectionYears || 0}-year projection period.
                </p>
                
                <h3 className="text-green-900 mt-6">Key Findings</h3>
                <ul>
                  <li>Average current basal area: {avgBasalArea.toFixed(2)} m²/ha</li>
                  <li>Average projected growth: {avgGrowth.toFixed(2)} m²/ha</li>
                  <li>Total estimated volume: {totalVolume.toFixed(2)} m³/ha</li>
                  <li>Analysis date: {new Date().toLocaleDateString()}</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
