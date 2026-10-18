# System Architecture & Technical Design Document

This document outlines the high-level system design, data flow, component breakdown, and database relationships of the **Basal Area Growth Rate Analytics System**.

---

## 1. High-Level C4 Architecture Diagrams

### C4 Level 1: System Context Diagram

```mermaid
graph TD
    User["👤 Forest Planner / Analyst"] -->|Enters Stand Parameters / Views Trajectory Charts| WebApp["🌲 Basal Area Growth Rate Tool (Web System)"]
    WebApp -->|Queries Coefficients / Logs Predictions| Database[("💾 SQL Database (SQLite/MSSQL)")]
```

---

### C4 Level 2: Container Diagram (System Architecture)

```mermaid
graph TB
    subgraph ClientLayer ["Client Layer"]
        UI[React 18 + TypeScript Web App]
        Charts[Recharts Visualization Engine]
    end

    subgraph APILayer ["API Layer - FastAPI"]
        Router[FastAPI REST API Router]
        Middleware[Telemetry & Request Timing Middleware]
    end

    subgraph BusinessLogic ["Business Logic Layer"]
        Predictor[RGR Prediction Engine]
        Trainer[RGR Model Training Pipeline]
        Store[Coefficient Store Manager]
    end

    subgraph PersistenceLayer ["Persistence Layer"]
        DB[(SQLite / MSSQL Database)]
        CSV[(Synthetic Data Fixtures)]
    end

    UI -->|JSON / REST| Router
    Router --> Middleware
    Middleware --> Predictor
    Router --> Store
    Predictor --> Store
    Store --> DB
    Trainer --> CSV
    Trainer --> Store
```

---

## 2. API Sequence Diagram (Prediction Flow)

```mermaid
sequenceDiagram
    autonumber
    actor User as Forest Analyst
    participant UI as React Frontend
    participant API as FastAPI Router
    participant Engine as Predictor Engine
    participant Store as Coefficient Store
    participant DB as SQL Database

    User->>UI: Input stand parameters (BA=25.0, AGS=0.75, Species=Sugar Maple)
    UI->>API: POST /api/calculate-growth {species: "Mh", BA: 25.0, ...}
    API->>Store: get_coefficients("Mh")
    Store->>DB: SELECT Intercept, Coef_BA, Coef_AGSfrac... FROM tblRGRCoefficients
    DB-->>Store: Return coefficient payload
    Store-->>API: Coefficients loaded
    API->>Engine: compute_rgr(coefficients, inputs)
    Engine-->>API: Return RGR = 0.0385, BA_future = 27.2
    API->>Store: log_prediction(inputs, outputs)
    Store->>DB: INSERT INTO tblRGRPredictions
    API-->>UI: 200 OK {success: true, RGR: 0.0385, trajectory: [...]}
    UI->>User: Render growth trajectory graph & summary statistics
```

---

## 3. Database Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    tblRGRCoefficients ||--o{ tblRGRTrainingHistory : "has audit history"
    tblRGRCoefficients ||--o{ tblRGRPredictions : "used for"
    tblRGRCoefficients ||--o{ tblRGRCoefficientHistory : "tracks versions"

    tblRGRCoefficients {
        int CoefficientID PK
        int SpecCode
        string ForestUnit
        string SpeciesName
        float Intercept
        float Coef_BA
        float Coef_BA2
        float Coef_AGSfrac
        float Coef_TCR
        float Coef_DQ
        float Coef_q
        float Coef_Cut
        float R_Squared
        float RMSE
        float MAE
        int N_Samples
        datetime TrainedDate
        string ModelVersion
    }

    tblRGRTrainingHistory {
        int TrainingID PK
        int CoefficientID FK
        datetime TrainingDate
        int N_Plots
        int N_GrowthIntervals
        float AvgBA
        string DataQualityFlags
    }

    tblRGRPredictions {
        int PredictionID PK
        int CoefficientID FK
        int PlotKey
        int MeasureYear
        float BA_Input
        float AGSfrac_Input
        float RGR_Predicted
        float BAgrowth_Predicted
        datetime PredictionDate
    }
```

---

## 4. Architectural Design Principles

1. **Separation of Concerns**: Clean isolation between UI visualization, FastAPI routing, domain prediction engines, and database abstraction layers.
2. **Environment Interoperability**: Automatic fallback between lightweight local SQLite storage (for zero-dependency development) and enterprise MSSQL servers.
3. **Auditability & Traceability**: Immutable prediction logs (`tblRGRPredictions`) and training run histories (`tblRGRTrainingHistory`) ensure complete computational reproducibility.
4. **Performance Optimization**: Request timing middleware (`X-Process-Time`) and response caching in frontend components minimize latency.

## 5. Model Benchmark Subsystem (`backend/ml`)

```mermaid
graph LR
    CLI[scripts/benchmark_models.py] --> B[ml.benchmark.benchmark_all]
    API[POST /api/benchmarks/run · background job] --> B
    B --> F[ml.features · base / extended / spatial design matrix]
    B --> R[ml.registry · 17 candidate estimators]
    B --> E[ml.evaluation · grouped / temporal / naive CV · metrics]
    B --> C[ml.conformal · split-conformal + quantile GBM intervals]
    B --> S[ml.stats · paired t · Wilcoxon · bootstrap CI · ranking]
    B --> W[ml.reports · benchmark_<ts>.json + LEADERBOARD.md]
    W --> G[GET /api/benchmarks · /api/benchmarks/{group}]
    G --> UI[Model Benchmarks tab · React + Recharts]
```

Design decisions:

- **Plot-grouped CV is the primary split.** Permanent sample plots are
  remeasured; treating intervals as independent rows leaks plot identity into
  the test fold. The naive split is still computed, purely to quantify that
  inflation.
- **The production OLS is always in the comparison** (it is inserted if the
  caller omits it) so every leaderboard answers "is anything better than what we
  ship?", with paired tests on the same folds.
- **Optional back-ends degrade gracefully.** Missing `xgboost`, `lightgbm` or
  `statsmodels` removes those rows from the leaderboard rather than failing the
  run; `/api/benchmarks/models` reports availability.
- **Reports are plain JSON on disk.** The API serves the newest file; the job
  registry shared with training exposes progress. No database schema change was
  needed for the first iteration (a `tblBenchmarkRuns` table is on the roadmap).
