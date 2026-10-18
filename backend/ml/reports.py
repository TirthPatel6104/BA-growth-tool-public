"""
ml/reports.py
=============
Persist benchmark output as JSON (consumed by the API/frontend) and a
human-readable Markdown leaderboard (for the docs / README).
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(_BACKEND_DIR, "artifacts", "benchmarks")


def _clean(value: Any) -> Any:
    """Replace NaN/inf with None so the JSON is strict and the frontend can parse it."""
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def write_json(report: Dict, path: Optional[str] = None, out_dir: str = BENCHMARK_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"benchmark_{stamp}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean(report), fh, indent=2, default=str)
    return path


def latest_json(out_dir: str = BENCHMARK_DIR) -> Optional[str]:
    if not os.path.isdir(out_dir):
        return None
    candidates = [
        os.path.join(out_dir, f) for f in os.listdir(out_dir)
        if f.startswith("benchmark_") and f.endswith(".json")
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], digits: int = 5) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "–"
    return f"{value:.{digits}f}"


def _pct(value: Optional[float]) -> str:
    return "–" if value is None else f"{value:+.1f}%"


def _p(value: Optional[float]) -> str:
    if value is None:
        return "–"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def to_markdown(report: Dict) -> str:
    lines: List[str] = []
    cfg = report.get("config", {})
    lines.append("# Model Benchmark Leaderboard")
    lines.append("")
    lines.append(f"Generated: `{report.get('generated_at', '')}`  ")
    lines.append(
        f"Primary CV: **{cfg.get('cv_strategy')}** ({cfg.get('n_splits')} folds, seed {cfg.get('seed')}), "
        f"feature set **{cfg.get('feature_set')}**, interval level **{int((1 - cfg.get('alpha', 0.1)) * 100)}%**"
    )
    lines.append("")

    summary = report.get("summary", {})
    lines.append("## Summary across model groups")
    lines.append("")
    lines.append("| Model | Mean rank | Wins |")
    lines.append("| :--- | ---: | ---: |")
    for name, rank in summary.get("mean_rank_by_model", {}).items():
        lines.append(f"| {name} | {rank:.2f} | {summary.get('wins_by_model', {}).get(name, 0)} |")
    lines.append("")

    for key, rep in report.get("groups", {}).items():
        ds = rep.get("dataset", {})
        lines.append(f"## {key}")
        lines.append("")
        if rep.get("error"):
            lines.append(f"> Skipped: {rep['error']}")
            lines.append("")
            continue
        yr = ds.get("year_range")
        lines.append(
            f"{ds.get('n_samples')} intervals from {ds.get('n_plots')} plots"
            + (f", {yr[0]}–{yr[1]}" if yr else "") + f". Best model: **{rep.get('best_model')}**."
        )
        lines.append("")
        lines.append(
            "| Rank | Model | Family | RMSE | ±sd | R² | MAE | p95 |AE| | Bias | vs OLS | "
            "p (paired t) | Wrong signs |"
        )
        lines.append("| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for name in rep.get("ranking", []):
            e = rep["models"][name]
            m, s = e["metrics_mean"], e["metrics_std"]
            vb = e.get("vs_baseline") or {}
            p_value = (vb.get("paired_t_on_fold_rmse") or {}).get("p_value")
            wrong = (e.get("sign_check") or {}).get("wrong_sign_count", "–")
            lines.append(
                f"| {e['rank']} | {name} | {e['family']} | {_fmt(m.get('rmse'))} | {_fmt(s.get('rmse'))} | "
                f"{_fmt(m.get('r2'), 3)} | {_fmt(m.get('mae'))} | {_fmt(m.get('p95_ae'))} | "
                f"{_fmt(m.get('bias'))} | {_pct(vb.get('rmse_improvement_pct'))} | {_p(p_value)} | {wrong} |"
            )
        lines.append("")

        leak = rep.get("leakage_check")
        if leak and leak.get("median_r2_inflation") is not None:
            lines.append(
                f"Leakage check: naive K-fold inflates R² by **{leak['median_r2_inflation']:+.3f}** (median across "
                f"models; mean {leak['mean_r2_inflation']:+.3f}) versus plot-grouped CV."
            )
            lines.append("")

        intervals = rep.get("intervals") or {}
        rows: List[Tuple[str, Dict]] = [(k, v) for k, v in intervals.items() if isinstance(v, dict) and "coverage" in v]
        if rows:
            lines.append("| Interval method | Model | Target | Coverage | Mean width | Calibrated |")
            lines.append("| :--- | :--- | ---: | ---: | ---: | :---: |")
            for k, v in rows:
                lines.append(
                    f"| {k} | {v.get('model')} | {_fmt(v.get('target_coverage'), 2)} | {_fmt(v.get('coverage'), 3)} | "
                    f"{_fmt(v.get('mean_width'))} | {'yes' if v.get('calibrated') else 'no'} |"
                )
            lines.append("")

        imp = rep.get("importance") or {}
        if imp.get("features"):
            top = ", ".join(f"{f['feature']} ({f['importance_mean']:.5f})" for f in imp["features"][:5])
            lines.append(f"Permutation importance ({imp.get('model')}): {top}")
            lines.append("")

    return "\n".join(lines)


def write_markdown(report: Dict, path: Optional[str] = None, out_dir: str = BENCHMARK_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if path is None:
        path = os.path.join(out_dir, "LEADERBOARD.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(report))
    return path
