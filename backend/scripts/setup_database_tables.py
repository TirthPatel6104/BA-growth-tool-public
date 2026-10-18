"""
scripts/setup_database_tables.py
==================================
LEGACY (SQL Server only) – executes sql/create_rgr_tables.sql through the
raw ODBC connection.

The supported way to create or upgrade the schema on every dialect is now:

    alembic upgrade head          # SQLite / PostgreSQL / SQL Server

For an existing SQL Server database that was created with this script, adopt
Alembic once with ``alembic stamp head``.  This script is kept for operators
who still maintain the T-SQL path.

Run once:
    python scripts/setup_database_tables.py
"""

from __future__ import annotations

import logging
import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from DatabaseLinker.main import get_raw_connection, require_mssql

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_SQL_FILE = os.path.join(_BACKEND_DIR, "sql", "create_rgr_tables.sql")


def setup_tables() -> None:
    """Create all RGR-related tables if they do not already exist (T-SQL)."""
    require_mssql("scripts/setup_database_tables.py (use `alembic upgrade head` instead)")
    logger.warning("Legacy T-SQL setup – prefer `alembic upgrade head` (works on every dialect).")

    logger.info("Connecting to database ...")
    conn   = get_raw_connection()
    cursor = conn.cursor()

    # Read the SQL file and split on GO statements
    with open(_SQL_FILE, encoding="utf-8") as fh:
        raw = fh.read()

    batches = [b.strip() for b in raw.split("\nGO") if b.strip()]

    for i, batch in enumerate(batches, 1):
        try:
            cursor.execute(batch)
            conn.commit()
        except Exception as exc:
            # Ignore "table already exists" but raise all other errors
            msg = str(exc).lower()
            if "already exists" in msg or "there is already an object" in msg:
                logger.debug("Batch %d: table already exists, skipping.", i)
            else:
                logger.error("Batch %d failed: %s", i, exc)
                conn.rollback()

    cursor.close()
    conn.close()
    logger.info("✓ Database setup complete.  Now run `alembic stamp head` to adopt Alembic.")


if __name__ == "__main__":
    setup_tables()
