"""
scripts/test_all_species_queries.py
=====================================
Verifies that sp_GetBAGrowthData (and sp_GetBAGrowthData_Cedar) return
results for all configured model definitions (8 species + 5 forest-unit).

Run from backend/:
    python scripts/test_all_species_queries.py

Optional flags:
    --min-samples N    Override the minimum sample threshold (default 30)
    --verbose          Print per-row column info for every result set
"""

from __future__ import annotations

import argparse
import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from DatabaseLinker.main import get_db_connection
from data.extractors import (
    BAGrowthDataExtractor,
    SPECIES_CONFIG,
    FOREST_UNIT_CONFIG,
    VALID_COMBINATIONS,
)

_DIV = "=" * 70


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test all configured species/FU model queries.")
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def main() -> None:
    args       = _parse_args()
    connection = get_db_connection()
    extractor  = BAGrowthDataExtractor(connection, populate_tcr=False)  # skip TCR for speed

    results: dict = {}

    print(_DIV)
    print("TESTING ALL MODEL QUERIES")
    print(f"Total models: {len(VALID_COMBINATIONS)}  "
          f"(8 species models  +  5 forest-unit models = 13)")
    print(_DIV)

    for species_abbrev, forest_unit in VALID_COMBINATIONS:
        is_fu = species_abbrev is None
        if is_fu:
            cfg   = FOREST_UNIT_CONFIG[forest_unit]
            key   = f"FU_{forest_unit}"
            label = f"FU {forest_unit} - {cfg['name']}"
        else:
            cfg   = SPECIES_CONFIG[species_abbrev]
            key   = species_abbrev
            label = f"{cfg['name']} ({species_abbrev}) - all FUs"

        print(f"\n>>  {label}")
        print(f"   Procedure : {cfg['procedure']}")
        if not is_fu:
            print(f"   SpecCode  : {cfg['code']}")

        try:
            df = extractor.extract_training_data(species_abbrev, forest_unit)
            n_rows  = len(df)
            n_plots = int(df["PlotKey"].nunique()) if n_rows else 0
            date_rng = (
                f"{int(df['StartYear'].min())}-{int(df['EndYear'].max())}"
                if n_rows else "N/A"
            )

            results[key] = {
                "status":    "SUCCESS",
                "n_rows":    n_rows,
                "n_plots":   n_plots,
                "date_range": date_rng,
                "sufficient": n_rows >= args.min_samples,
            }

            flag = "OK " if n_rows >= args.min_samples else "LOW"
            print(f"   {flag} {n_rows:>5} growth intervals | {n_plots:>4} plots | {date_rng}")

            if args.verbose and n_rows:
                print(f"   Columns: {list(df.columns)}")
                print(df.head(2).to_string(index=False))

        except Exception as exc:  # noqa: BLE001
            results[key] = {"status": "FAILED", "error": str(exc)}
            print(f"   ERR  FAILED: {exc}")

    connection.close()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{_DIV}")
    print("SUMMARY")
    print(_DIV)

    succeeded    = [k for k, v in results.items() if v["status"] == "SUCCESS"]
    failed       = [k for k, v in results.items() if v["status"] == "FAILED"]
    insufficient = [k for k, v in results.items()
                    if v["status"] == "SUCCESS" and not v.get("sufficient", True)]

    n_sp_ok = sum(1 for k in succeeded if not k.startswith("FU_"))
    n_fu_ok = sum(1 for k in succeeded if k.startswith("FU_"))

    print(f"  Total models          : {len(results)}  (8 species + 5 FU)")
    print(f"   Species models OK   : {n_sp_ok} / 8")
    print(f"   FU models OK        : {n_fu_ok} / 5")
    print(f"   Failed              : {len(failed)}")
    print(f"   Insufficient data (< {args.min_samples} samples): {len(insufficient)}")

    if succeeded:
        print(f"\n  Successful combinations:")
        for k in succeeded:
            r = results[k]
            tag = "" if r["sufficient"] else "  LOW LOW DATA"
            print(f" {k:<22}  {r['n_rows']:>5} rows  {r['n_plots']:>4} plots  {r['date_range']}{tag}")

    if failed:
        print(f"\n  Failed combinations:")
        for k in failed:
            print(f"{k:<22}  {results[k]['error']}")

    print()
    if not failed:
        print("All queries executed successfully.")
    else:
        print(f"{len(failed)} combination(s) failed - check stored procedures and database.")
        sys.exit(1)


if __name__ == "__main__":
    main()
