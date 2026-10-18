"""
data/extractors.py
==================
Extracts BA growth training data from the configured database.

Two data sources:

  * Default (every dialect) – the portable ``tblBAGrowthData`` table
    (``db.models.BAGrowthInterval``): one row per plot × growth interval.
  * Optional (SQL Server only, ``BAGT_USE_STORED_PROCEDURES=true``) –
    site-specific stored procedures that compute the same row shape on the fly
    from an inventory database.  The procedures are **not** part of this
    repository (they encode a particular organisation's schema); see
    ``sql/README.md`` for the result-set contract they must satisfy.

Model structure
  Species models  (8 total): one set of coefficients per species across ALL FUs
  Forest-unit models (5 total): one set of coefficients per FU, stand-level BA/DQ/q

Integrates with the SQLAlchemy session factory in
  DatabaseLinker/main.py  ->  get_db_connection()
TCR enrichment (optional) uses a polygon shapefile pointed to by
``BAGT_TCR_SHAPEFILE``; without it the stored TCR values are kept.
"""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd
from shapely.geometry import Point
from shapely.ops import transform as shapely_transform
from sqlalchemy import and_, func, select, text
from sqlalchemy.orm import Session

try:
    from pyproj import Transformer
except Exception:  # noqa: BLE001
    Transformer = None

try:
    import geopandas as gpd
except Exception:  # noqa: BLE001
    gpd = None

# ---------------------------------------------------------------------------
# Resolve paths so imports work regardless of working directory
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from DatabaseLinker.main import get_db_connection  # SQLAlchemy session factory
from db.models import GROWTH_DATA_COLUMNS, BAGrowthInterval

# Re-export FOREST_UNIT_CONFIG for use by pipeline and other modules

# ---------------------------------------------------------------------------
# Lazy TCR shapefile load (no DB/CSV side effects).
# ---------------------------------------------------------------------------
_TCR_GDF = None
_TCR_COL = None
_TCR_TARGET_CRS = None
_TCR_TRANSFORMERS: Dict[int, Optional[Transformer]] = {}
_TCR_LOADED = False

logger = logging.getLogger(__name__)


def _ensure_tcr_lookup_loaded() -> None:
    """Load TCR polygon shapefile once, on demand."""
    global _TCR_GDF, _TCR_COL, _TCR_TARGET_CRS, _TCR_LOADED

    if _TCR_LOADED:
        return

    _TCR_LOADED = True

    if gpd is None:
        logger.warning("TCR spatial lookup unavailable (geopandas missing). Defaulting TCR=5.")
        return

    shp_path = os.getenv("BAGT_TCR_SHAPEFILE", os.path.join(_BACKEND_DIR, "spatial", "tcr_polygons.shp"))
    if not os.path.exists(shp_path):
        logger.warning("TCR shapefile not found at %s. Defaulting TCR=5.", shp_path)
        return

    try:
        gdf = gpd.read_file(shp_path)
        for candidate in ("TIMBER_CAP", "OLI_TIMBER_CAP", "TCR", "TIMBERCAP"):
            if candidate in gdf.columns:
                _TCR_GDF = gdf
                _TCR_COL = candidate
                _TCR_TARGET_CRS = gdf.crs
                _TCR_TRANSFORMERS.clear()

                if Transformer is None:
                    logger.warning("pyproj unavailable; datum/zone-aware TCR transforms disabled.")
                elif gdf.crs is None:
                    logger.warning("TCR shapefile has no CRS metadata; datum/zone-aware transforms disabled.")
                else:
                    logger.info("TCR lookup coordinate transforms target CRS: %s", gdf.crs)

                logger.info("Loaded TCR shapefile (%d polygons, column=%s).", len(gdf), candidate)
                return

        logger.warning(
            "TCR shapefile loaded but no TCR column found in %s. Defaulting TCR=5.",
            list(gdf.columns),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("TCR spatial lookup unavailable (%s). Defaulting TCR=5.", exc)

# ---------------------------------------------------------------------------
# Species model configuration
# One set of coefficients per species, plots from ALL qualifying forest units.
# BA, DQ, q calculated for the TARGET SPECIES ONLY.
# ---------------------------------------------------------------------------
SPECIES_CONFIG: Dict[str, Dict] = {
    # SpecCode 30 – Hard Maple
    "Mh":    {"code": 30,  "name": "Hard Maple",   "procedure": "sp_GetBAGrowthData_Species"},
    # SpecCode 38 – Red Oak
    "Or":    {"code": 38,  "name": "Red Oak",       "procedure": "sp_GetBAGrowthData_Species"},
    # SpecCode 37 – Yellow Birch
    "By":    {"code": 37,  "name": "Yellow Birch",  "procedure": "sp_GetBAGrowthData_Species"},
    # SpecCode 19 – Hemlock
    "He":    {"code": 19,  "name": "Hemlock",       "procedure": "sp_GetBAGrowthData_Species"},
    # SpecCode 45 – Black Ash
    "Ab":    {"code": 45,  "name": "Black Ash",     "procedure": "sp_GetBAGrowthData_Species"},
    # SpecCode 32 – Red Maple
    "Mr":    {"code": 32,  "name": "Red Maple",     "procedure": "sp_GetBAGrowthData_Species"},
    # SpecCodes 22 + 181 – Cedar combined group (Cw + CE)
    "Cedar": {"code": 999, "name": "Cedar (Cw+CE)", "procedure": "sp_GetBAGrowthData_Cedar"},
    # SpecCode 44 – Beech
    "Be":    {"code": 44,  "name": "Beech",         "procedure": "sp_GetBAGrowthData_Species"},
}

# ---------------------------------------------------------------------------
# Forest unit model configuration
# One set of coefficients per forest unit, stand-level BA/DQ/q across ALL species.
# ---------------------------------------------------------------------------
FOREST_UNIT_CONFIG: Dict[str, Dict] = {
    "TH1":   {"code": None, "name": "TH1 (NER)",    "procedure": "sp_GetBAGrowthData_ForestUnit", "require_tolerant_species": 1},
    "HDSL":  {"code": None, "name": "HDSL (GLSL)",  "procedure": "sp_GetBAGrowthData_ForestUnit", "require_tolerant_species": 0},  # relaxed
    "HDSL2": {"code": None, "name": "HDSL2 (GLSL)", "procedure": "sp_GetBAGrowthData_ForestUnit", "require_tolerant_species": 1},
    "CE1":   {"code": None, "name": "CE1 (GLSL)",   "procedure": "sp_GetBAGrowthData_ForestUnit", "require_tolerant_species": 1},
    "HE1":   {"code": None, "name": "HE1 (GLSL)",   "procedure": "sp_GetBAGrowthData_ForestUnit", "require_tolerant_species": 1},
}

# Flat list of all valid forest unit codes
FOREST_UNITS = list(FOREST_UNIT_CONFIG.keys())

# ---------------------------------------------------------------------------
# All 13 valid model combinations:
#   8 species models  → (species_abbrev, None)   plots from ALL forest units
#   5 FU models       → (None, forest_unit)      stand-level, all species
# ---------------------------------------------------------------------------
VALID_COMBINATIONS = (
    [(sp, None) for sp in SPECIES_CONFIG]       +  # 8 species models
    [(None, fu) for fu in FOREST_UNIT_CONFIG]      # 5 forest unit models
)

MIN_SAMPLES = 30  # Minimum growth intervals needed to train a model
# Read from env vars so operators can tune without code changes (SQL Server only).
# QUERY_TIMEOUT: applied at connection level by db/engine.py (0 = no timeout).
# LOCK_TIMEOUT : -1 = no limit; READ UNCOMMITTED in SPs prevents contention.
SQL_QUERY_TIMEOUT_SECONDS = int(os.getenv("BAGT_SQL_QUERY_TIMEOUT", "0"))
SQL_LOCK_TIMEOUT_MS = int(os.getenv("BAGT_SQL_LOCK_TIMEOUT_MS", "-1"))

# ---------------------------------------------------------------------------
# TCR helper
# ---------------------------------------------------------------------------

def _lookup_tcr(easting: Optional[float], northing: Optional[float]) -> Optional[int]:
    """
    Return TCR from spatial lookup, or None when unavailable/invalid.
    """
    _ensure_tcr_lookup_loaded()

    if _TCR_GDF is None or _TCR_COL is None:
        return None

    if easting is None or northing is None or pd.isna(easting) or pd.isna(northing):
        return None

    try:
        point = Point(float(easting), float(northing))
        matches = _TCR_GDF[_TCR_GDF.geometry.covers(point)]
        if not matches.empty:
            tcr_value = matches.iloc[0][_TCR_COL]
            return int(tcr_value) if tcr_value is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("TCR lookup failed for (%.1f, %.1f): %s", easting, northing, exc)

    return None


def _make_transformer(data: int) -> Optional[Transformer]:
    if Transformer is None or _TCR_TARGET_CRS is None:
        return None
    if data == 15:
        return Transformer.from_crs("EPSG:26915", _TCR_TARGET_CRS, always_xy=True)
    if data == 16:
        return Transformer.from_crs("EPSG:26916", _TCR_TARGET_CRS, always_xy=True)
    if data == 18:
        return Transformer.from_crs("EPSG:26918", _TCR_TARGET_CRS, always_xy=True)
    if data == 83 or data == 17:
        return Transformer.from_crs("EPSG:26917", _TCR_TARGET_CRS, always_xy=True)
    if data == 27:
        return Transformer.from_crs("EPSG:26717", _TCR_TARGET_CRS, always_xy=True)
    return None


def _get_transformer(data: int) -> Optional[Transformer]:
    if data not in _TCR_TRANSFORMERS:
        _TCR_TRANSFORMERS[data] = _make_transformer(data)
    return _TCR_TRANSFORMERS[data]


def _choose_one_tcr(tcr_list):
    if not tcr_list:
        return None
    counts = Counter(tcr_list)
    top = max(counts.values())
    candidates = sorted([key for key, value in counts.items() if value == top])
    return candidates[0]


def _lookup_tcr_with_data(
    easting: Optional[float],
    northing: Optional[float],
    data: Optional[int],
) -> Optional[int]:
    """Find TCR using datum/zone-aware transforms from updated FindTCR logic."""
    _ensure_tcr_lookup_loaded()

    if _TCR_GDF is None or _TCR_COL is None:
        return None
    if easting is None or northing is None or pd.isna(easting) or pd.isna(northing):
        return None

    try:
        point = Point(float(easting), float(northing))
        if data is not None:
            transformer = _get_transformer(int(data))
            if transformer is None:
                return None
            point = shapely_transform(transformer.transform, point)

        matches = _TCR_GDF[_TCR_GDF.geometry.covers(point)]
        if matches.empty:
            return None

        tcr_vals = matches[_TCR_COL].tolist()
        chosen = _choose_one_tcr(tcr_vals)
        return int(chosen) if chosen is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("TCR lookup with datum/zone failed for (%.1f, %.1f, data=%s): %s", easting, northing, data, exc)
        return None


def _get_tcr(easting: Optional[float], northing: Optional[float]) -> int:
    """
    Backward-compatible TCR helper for call sites that still require an int.
    Returns 5 when spatial lookup is unavailable.
    """
    value = _lookup_tcr(easting, northing)
    return value if value is not None else 5


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

class BAGrowthDataExtractor:
    """
    Extract BA growth training data from the database.

    Parameters
    ----------
    connection : sqlalchemy.orm.Session, optional
        An open session.  If None a new one is created via
        get_db_connection() from DatabaseLinker/main.py.
    populate_tcr : bool
        When True, the TCR column is overridden with values from the
        spatial shapefile lookup (FindTCR.py).  Defaults to True.
    """

    def __init__(self, connection: Optional[Session] = None, populate_tcr: bool = True):
        self.session: Session = connection or get_db_connection()
        # Backward-compatible alias used by older call sites.
        self.connection = self.session
        self.populate_tcr = populate_tcr

    # ------------------------------------------------------------------
    # Dialect helpers
    # ------------------------------------------------------------------

    @property
    def dialect(self) -> str:
        return self.session.get_bind().dialect.name

    @property
    def uses_stored_procedures(self) -> bool:
        """
        True only on SQL Server *and* when ``BAGT_USE_STORED_PROCEDURES`` is
        enabled.  Otherwise every dialect reads the portable ``tblBAGrowthData``.
        """
        flag = os.getenv("BAGT_USE_STORED_PROCEDURES", "false").strip().lower()
        return self.dialect == "mssql" and flag in {"1", "true", "yes", "on"}

    @staticmethod
    def _coord_metadata_from_frame(df: pd.DataFrame) -> Dict[int, Dict[str, Optional[int]]]:
        """
        Best-effort Datum / UTM-zone metadata per PlotKey, taken from optional
        ``Datum`` / ``Zone`` columns of the extracted frame.  Site-specific
        stored procedures may include them; the portable table does not, in
        which case the TCR lookup uses the coordinates untransformed.
        """
        if "PlotKey" not in df.columns or not {"Datum", "Zone"} & set(df.columns):
            return {}
        result: Dict[int, Dict[str, Optional[int]]] = {}
        for _, row in df.drop_duplicates("PlotKey").iterrows():
            if pd.isna(row.get("PlotKey")):
                continue
            datum = row.get("Datum")
            zone = row.get("Zone")
            result[int(row["PlotKey"])] = {
                "datum": int(datum) if datum is not None and not pd.isna(datum) else None,
                "zone": int(zone) if zone is not None and not pd.isna(zone) else None,
            }
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_training_data(
        self,
        species_abbrev: Optional[str],
        forest_unit: Optional[str] = None,
        since_year: int = 0,
    ) -> pd.DataFrame:
        """
        Extract training data for one model.

        Model types:
          Species model : species_abbrev is set, forest_unit is None.
                          Selects all qualifying plots regardless of FU.
                          BA/DQ/q calculated for target species only.
          FU model      : species_abbrev is None, forest_unit is set.
                          Selects plots classified as the given FU.
                          BA/DQ/q aggregated across ALL species.

        Parameters
        ----------
        species_abbrev : str or None
            Species abbreviation (e.g. 'Mh', 'Cedar') for species models.
            Pass None for forest-unit models.
        forest_unit    : str or None
            Forest unit code (e.g. 'TH1', 'HDSL2') for FU models.
            Ignored for species models.
        since_year     : int
            0 = full extract; >0 = only intervals ending after this year.

        Returns
        -------
        pd.DataFrame  with columns:
            SpecCode, ForestUnit, PlotKey, PlotName, DatasetCode,
            StartYear, EndYear, GrowthInterval_Years,
            BA, AGSfrac, TCR, DQ, q, Cut,
            BAgrowth, RGR,
            Latitude, Longitude, Easting, Northing,
            TreeCount, ShrinkageCount, QFactorClasses, BAgrowth_Check
        """
        if species_abbrev is not None:
            # --- Species model ---
            if species_abbrev not in SPECIES_CONFIG:
                raise ValueError(f"Unknown species abbreviation: '{species_abbrev}'. "
                                 f"Valid options: {list(SPECIES_CONFIG)}")
            cfg = SPECIES_CONFIG[species_abbrev]
            logger.info(
                "Extracting %-14s (all FUs)%s ...",
                cfg["name"],
                f" (since {since_year})" if since_year else "",
            )
        else:
            # --- Forest-unit model ---
            if forest_unit is None or forest_unit not in FOREST_UNIT_CONFIG:
                raise ValueError(f"Unknown forest unit: '{forest_unit}'. "
                                 f"Valid options: {FOREST_UNITS}")
            cfg = FOREST_UNIT_CONFIG[forest_unit]
            logger.info(
                "Extracting FU %-8s%s ...",
                forest_unit,
                f" (since {since_year})" if since_year else "",
            )

        df = self._call_procedure(cfg, forest_unit, since_year)

        if df.empty:
            logger.warning("  ⚠  No data returned for %s/%s", species_abbrev, forest_unit)
            return df

        # Override TCR via spatial lookup when requested.
        # For training data quality, rows with unresolved/invalid TCR are dropped
        # instead of defaulting TCR=5.
        if self.populate_tcr and "Easting" in df.columns:
            _ensure_tcr_lookup_loaded()
            if _TCR_GDF is None or _TCR_COL is None:
                logger.warning("  TCR spatial lookup unavailable; retaining DB TCR values.")
            else:
                logger.info("  Populating TCR from spatial shapefile ...")
                before_lookup = len(df)

                coord_meta = self._coord_metadata_from_frame(df)

                def _row_tcr(row):
                    plot_key = row.get("PlotKey")
                    metadata = coord_meta.get(int(plot_key), {}) if pd.notna(plot_key) else {}
                    data_value = metadata.get("zone") if metadata.get("zone") is not None else metadata.get("datum")
                    return _lookup_tcr_with_data(row.get("Easting"), row.get("Northing"), data_value)

                df["TCR"] = df.apply(
                    _row_tcr,
                    axis=1,
                )
                df = df[df["TCR"].notna()].copy()
                df = df[df["TCR"].between(1, 7)].copy()
                dropped = before_lookup - len(df)
                if dropped > 0:
                    logger.warning("  Dropped %d rows with missing/invalid TCR (outside 1-7).", dropped)

        if df.empty:
            logger.warning(
                "  ⚠  No usable rows remain for %s/%s after TCR filtering.",
                species_abbrev,
                forest_unit,
            )
            return df

        logger.info(
            "  ✓ %d growth intervals | %d plots | %d-%d",
            len(df),
            df["PlotKey"].nunique(),
            int(df["StartYear"].min()),
            int(df["EndYear"].max()),
        )
        return df

    def extract_all_combinations(self) -> Dict[str, pd.DataFrame]:
        """
        Extract training data for all 13 model combinations:
          8 species models + 5 forest-unit models.

        Returns
        -------
        dict  {"{key}": pd.DataFrame}  where key is the species abbrev
              (e.g. 'Mh') or 'FU_{forest_unit}' (e.g. 'FU_TH1').
              Only combinations with >= MIN_SAMPLES rows are included.
        """
        all_data: Dict[str, pd.DataFrame] = {}

        for species_abbrev, forest_unit in VALID_COMBINATIONS:
            key = species_abbrev if species_abbrev is not None else f"FU_{forest_unit}"
            try:
                df = self.extract_training_data(species_abbrev, forest_unit)
                if len(df) >= MIN_SAMPLES:
                    all_data[key] = df
                else:
                    logger.warning(
                        "  ⚠  Skipping %s – only %d samples (need %d)",
                        key, len(df), MIN_SAMPLES,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("  ✗ Failed to extract %s: %s", key, exc)

        logger.info(
            "\n✓ Extracted %d valid combination(s) out of %d possible.",
            len(all_data),
            len(VALID_COMBINATIONS),
        )
        return all_data

    def get_max_available_year(
        self,
        species_abbrev: str,
    ) -> int:
        """
        Return the maximum EndYear available for this species (or Cedar group).
        Used by the incremental-training check.
        """
        cfg = SPECIES_CONFIG[species_abbrev]

        if self.uses_stored_procedures:
            # Site-specific procedure: ask it for everything and take the max.
            df = self._call_stored_procedure(cfg, None, since_year=0)
            if df.empty or "EndYear" not in df.columns:
                return 0
            value = pd.to_numeric(df["EndYear"], errors="coerce").max()
            return int(value) if pd.notna(value) else 0

        stmt = select(func.max(BAGrowthInterval.end_year)).where(
            self._growth_table_filter(cfg, forest_unit=None)
        )
        value = self.session.execute(stmt).scalar()
        return int(value) if value is not None else 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_procedure(self, cfg: Dict, forest_unit: Optional[str], since_year: int = 0) -> pd.DataFrame:
        """
        Fetch the growth intervals for one model configuration as a DataFrame.

        SQL Server executes the stored procedure named in ``cfg``; every other
        dialect queries the portable ``tblBAGrowthData`` table.  Both return
        the same column set (``db.models.GROWTH_DATA_COLUMNS``).
        """
        if self.uses_stored_procedures:
            return self._call_stored_procedure(cfg, forest_unit, since_year)
        return self._query_growth_table(cfg, forest_unit, since_year)

    @staticmethod
    def _growth_table_filter(cfg: Dict, forest_unit: Optional[str], since_year: int = 0):
        """WHERE clause selecting species rows or stand-level FU rows from tblBAGrowthData."""
        code = cfg.get("code")
        if code == 999:
            # Cedar combined group: individual Cw / CE codes or the pooled 999 code.
            conditions = [BAGrowthInterval.spec_code.in_([22, 181, 999])]
        elif code is None:
            # Forest-unit model: stand-level rows (no species) for that FU.
            conditions = [
                BAGrowthInterval.spec_code.is_(None),
                BAGrowthInterval.forest_unit == forest_unit,
            ]
        else:
            conditions = [BAGrowthInterval.spec_code == int(code)]

        if since_year and int(since_year) > 0:
            conditions.append(BAGrowthInterval.end_year > int(since_year))
        return and_(*conditions)

    def _query_growth_table(self, cfg: Dict, forest_unit: Optional[str], since_year: int = 0) -> pd.DataFrame:
        table = BAGrowthInterval.__table__
        columns = [table.c[name] for name in GROWTH_DATA_COLUMNS]
        stmt = (
            select(*columns)
            .where(self._growth_table_filter(cfg, forest_unit, since_year))
            .order_by(table.c.PlotKey, table.c.StartYear, table.c.EndYear)
        )
        rows = self.session.execute(stmt).mappings().all()
        return pd.DataFrame([dict(row) for row in rows], columns=GROWTH_DATA_COLUMNS)

    def _call_stored_procedure(self, cfg: Dict, forest_unit: Optional[str], since_year: int = 0) -> pd.DataFrame:
        """Execute the SQL Server stored procedure and return a DataFrame."""
        procedure = cfg["procedure"]
        code = cfg.get("code")

        try:
            # SET LOCK_TIMEOUT: -1 = no limit (READ UNCOMMITTED in SPs handles contention).
            # The per-query timeout (BAGT_SQL_QUERY_TIMEOUT) is applied at connection
            # level by db/engine.py, so no cursor.timeout handling is needed here.
            self.session.execute(text(f"SET LOCK_TIMEOUT {int(SQL_LOCK_TIMEOUT_MS)}"))

            if code == 999:
                # Cedar: no SpecCode, no ForestUnit filter (all FUs)
                stmt = text(f"EXEC {procedure} @SinceYear = :since_year")
                params: Dict[str, object] = {"since_year": int(since_year)}
            elif code is None:
                # Forest-unit model: ForestUnit + RequireTolerantSpecies + SinceYear
                require_tolerant_species = cfg.get("require_tolerant_species", 1)  # Default to 1 (True)
                stmt = text(
                    f"EXEC {procedure} @ForestUnit = :forest_unit, "
                    "@RequireTolerantSpecies = :require_tolerant_species, @SinceYear = :since_year"
                )
                params = {
                    "forest_unit": forest_unit,
                    "require_tolerant_species": int(require_tolerant_species),
                    "since_year": int(since_year),
                }
            else:
                # Species model: SpecCode only, no ForestUnit filter
                stmt = text(f"EXEC {procedure} @SpecCode = :spec_code, @SinceYear = :since_year")
                params = {"spec_code": int(code), "since_year": int(since_year)}

            result = self.session.execute(stmt, params)
            columns: List[str] = list(result.keys())
            rows = result.fetchall()
            return pd.DataFrame.from_records(rows, columns=columns)

        except Exception as exc:
            logger.error("  Stored procedure call failed: %s", exc)
            raise
