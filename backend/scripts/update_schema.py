"""
scripts/update_schema.py
=========================
LEGACY (SQL Server only) – applies sql/add_confidence_columns.sql through the
raw ODBC connection.  Idempotent; safe to run multiple times.

Schema changes are now tracked with Alembic (``alembic upgrade head``).  This
script remains for SQL Server installs that predate Alembic: run it once to
bring the schema up to the ``0001_initial_schema`` revision, then
``alembic stamp head``.

Run once:
    python scripts/update_schema.py
"""

from __future__ import annotations

import logging
import os
import re
import sys

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from DatabaseLinker.main import get_raw_connection, require_mssql

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_SQL_FILE = os.path.join(_BACKEND_DIR, "sql", "add_confidence_columns.sql")


def run_migration() -> None:
    require_mssql("scripts/update_schema.py (use `alembic upgrade head` instead)")
    logger.warning("Legacy T-SQL migration – prefer `alembic upgrade head` (works on every dialect).")

    logger.info("Connecting to database ...")
    conn   = get_raw_connection()
    cursor = conn.cursor()

    with open(_SQL_FILE, encoding="utf-8") as fh:
        raw = fh.read()

    batches = [b.strip() for b in re.split(r"^\s*GO\s*$", raw, flags=re.MULTILINE) if b.strip()]

    for i, batch in enumerate(batches, 1):
        try:
            cursor.execute(batch)
            conn.commit()
            logger.info("  Batch %d OK", i)
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" in msg or "duplicate column" in msg:
                logger.info("  Batch %d: column already exists, skipping.", i)
            else:
                logger.error("  Batch %d FAILED: %s", i, exc)
                raise

    cursor.close()
    conn.close()
    logger.info("✓ Schema migration complete.  Now run `alembic stamp head` to adopt Alembic.")


if __name__ == "__main__":
    run_migration()
