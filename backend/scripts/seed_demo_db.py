"""
scripts/seed_demo_db.py
=======================
Create the schema (Alembic ``upgrade head``) and populate the synthetic demo
dataset through the SQLAlchemy ORM.

    python scripts/seed_demo_db.py                 # default SQLite: backend/demo_basal_growth.db
    python scripts/seed_demo_db.py --db-path ./x.db
    DATABASE_URL=postgresql://... python scripts/seed_demo_db.py
    python scripts/seed_demo_db.py --no-reset      # keep existing rows, only add if empty

The dataset is 100% synthetic (see db/seed.py) – no real inventory data.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, Optional

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def seed_demo_database(
    db_path: Optional[str] = None,
    *,
    database_url: Optional[str] = None,
    reset: bool = True,
    rng_seed: int = 42,
) -> Dict[str, int]:
    """
    Migrate + seed the configured database.  Returns row counts per table.

    ``db_path`` keeps the old call signature (SQLite file); ``database_url``
    targets any SQLAlchemy URL.  Both override the environment for this call.
    """
    if database_url:
        os.environ["DATABASE_URL"] = database_url
    elif db_path:
        os.environ.pop("DATABASE_URL", None)
        os.environ["BAGT_SQLITE_PATH"] = os.path.abspath(db_path)

    from db.bootstrap import init_db
    from db.config import redact_url
    from db.engine import reset_engine, session_scope
    from db.seed import demo_database_is_empty, seed_demo_database as _seed, seed_summary

    reset_engine()  # pick up any env override above
    init_db(migrate=True, seed=False)

    with session_scope() as session:
        if reset or demo_database_is_empty(session):
            summary = _seed(session, reset=reset, rng_seed=rng_seed)
            logger.info("Inserted synthetic demo rows: %s", summary)
        else:
            logger.info("Database already contains data and --no-reset was given; nothing inserted.")
        counts = seed_summary(session)

    logger.info("Demo database ready at %s", redact_url())
    for table, count in counts.items():
        logger.info("  %-28s %6d rows", table, count)
    return counts


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the synthetic demo database via the ORM.")
    parser.add_argument("--db-path", help="SQLite file to create/seed (default: backend/demo_basal_growth.db)")
    parser.add_argument("--database-url", help="Any SQLAlchemy URL (overrides --db-path and DATABASE_URL)")
    parser.add_argument("--no-reset", action="store_true", help="Do not delete existing rows first")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the synthetic data (default 42)")
    args = parser.parse_args(argv)

    seed_demo_database(
        args.db_path,
        database_url=args.database_url,
        reset=not args.no_reset,
        rng_seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
