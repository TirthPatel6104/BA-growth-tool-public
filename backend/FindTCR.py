"""
FindTCR.py
==========
Stand-alone utility: enrich a growth-interval CSV with the Timber Capability
Rating (TCR) of the polygon each plot falls in.

Inputs (all env-configurable, see .env.example)
    BAGT_TCR_TRAIN_CSV   CSV with at least a ``PlotKey`` column
    BAGT_TCR_COORDS_CSV  optional CSV with PlotKey, Easting, Northing and
                         optional Datum / Zone columns.  When omitted the
                         coordinates are read from the training CSV itself.
    BAGT_TCR_SHAPEFILE   polygon shapefile carrying a TCR attribute column
    BAGT_TCR_OUT_CSV     output path

No database access is performed – coordinates come from CSV files – so the
script contains nothing specific to any organisation's inventory schema.

    python FindTCR.py
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, Optional, Tuple

import pandas as pd

TRAIN_CSV = os.getenv("BAGT_TCR_TRAIN_CSV", "training_data.csv")
COORDS_CSV = os.getenv("BAGT_TCR_COORDS_CSV", "")
OUT_CSV = os.getenv("BAGT_TCR_OUT_CSV", "ba_growth_train_with_TCR.csv")
SHAPEFILE = os.getenv("BAGT_TCR_SHAPEFILE", "tcr_polygons.shp")
TCR_COLUMNS = ("TIMBER_CAP", "OLI_TIMBER_CAP", "TCR", "TIMBERCAP")

# Datum / UTM zone codes → source CRS.  Extend for other regions.
_CRS_BY_CODE = {
    15: "EPSG:26915",   # NAD83 / UTM 15N
    16: "EPSG:26916",   # NAD83 / UTM 16N
    17: "EPSG:26917",   # NAD83 / UTM 17N
    83: "EPSG:26917",   # datum code 83 → NAD83 / UTM 17N
    18: "EPSG:26918",   # NAD83 / UTM 18N
    27: "EPSG:26717",   # NAD27 / UTM 17N
}


def choose_one_tcr(tcr_list) -> Optional[int]:
    """Pick a single TCR when polygons overlap: the mode, ties → smallest value."""
    if not tcr_list:
        return None
    counts = Counter(tcr_list)
    top = max(counts.values())
    return sorted(k for k, v in counts.items() if v == top)[0]


class TCRLookup:
    """Point-in-polygon TCR lookup with datum / zone aware re-projection."""

    def __init__(self, shapefile: str = SHAPEFILE):
        import geopandas as gpd  # heavy import kept local

        self.gdf = gpd.read_file(shapefile)
        self.target_crs = self.gdf.crs
        self.column = next((c for c in TCR_COLUMNS if c in self.gdf.columns), None)
        if self.column is None:
            raise ValueError(f"No TCR column in shapefile; expected one of {TCR_COLUMNS}")
        self._transformers: Dict[int, object] = {}

    def _transformer(self, code: Optional[int]):
        if code is None:
            return None
        if code not in self._transformers:
            from pyproj import Transformer

            src = _CRS_BY_CODE.get(int(code))
            self._transformers[code] = (
                Transformer.from_crs(src, self.target_crs, always_xy=True) if src and self.target_crs else None
            )
        return self._transformers[code]

    def lookup(self, easting: float, northing: float, code: Optional[int]) -> Tuple[Optional[int], int]:
        from shapely.geometry import Point

        x, y = float(easting), float(northing)
        transformer = self._transformer(code)
        if transformer is not None:
            x, y = transformer.transform(x, y)
        matches = self.gdf[self.gdf.geometry.covers(Point(x, y))]
        if matches.empty:
            return None, 0
        values = matches[self.column].tolist()
        chosen = choose_one_tcr(values)
        return (int(chosen) if chosen is not None else None), len(values)


def enrich(train: pd.DataFrame, coords: pd.DataFrame, lookup: TCRLookup) -> pd.DataFrame:
    required = {"PlotKey", "Easting", "Northing"}
    missing = required - set(coords.columns)
    if missing:
        raise ValueError(f"Coordinates frame is missing columns: {sorted(missing)}")

    rows = []
    for row in coords.drop_duplicates("PlotKey").itertuples(index=False):
        zone = getattr(row, "Zone", None)
        datum = getattr(row, "Datum", None)
        code = zone if zone is not None and not pd.isna(zone) else datum
        code = int(code) if code is not None and not pd.isna(code) else None
        if pd.isna(row.Easting) or pd.isna(row.Northing):
            rows.append((row.PlotKey, None, 0))
            continue
        tcr, n_matches = lookup.lookup(row.Easting, row.Northing, code)
        rows.append((row.PlotKey, tcr, n_matches))

    tcr_df = pd.DataFrame(rows, columns=["PlotKey", "TCR", "TCR_matches"])
    print(f"TCR coverage: {tcr_df['TCR'].notna().sum()} / {len(tcr_df)} plots")
    merged = train.drop(columns=[c for c in ("TCR",) if c in train.columns])
    return merged.merge(tcr_df[["PlotKey", "TCR"]], on="PlotKey", how="left")


def main() -> None:
    train = pd.read_csv(TRAIN_CSV)
    if "PlotKey" not in train.columns:
        raise ValueError("PlotKey column not found in training CSV.")
    coords = pd.read_csv(COORDS_CSV) if COORDS_CSV else train
    print(f"Loaded {len(train)} rows, {train['PlotKey'].nunique()} plots")
    lookup = TCRLookup(SHAPEFILE)
    print(f"Shapefile CRS: {lookup.target_crs}  ({len(lookup.gdf)} polygons, column {lookup.column})")
    final = enrich(train, coords, lookup)
    final.to_csv(OUT_CSV, index=False)
    print(f"Saved merged dataset -> {OUT_CSV}")


if __name__ == "__main__":
    main()
