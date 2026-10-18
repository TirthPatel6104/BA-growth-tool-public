"""
scripts/deploy_stored_procedures.py
=====================================
Deploys site-specific ``sql/sp_*.sql`` stored procedures to SQL Server.

The procedures themselves are **not** part of this repository – they depend on
an organisation's inventory schema.  Drop your own files into ``backend/sql/``
(they are git-ignored) following the contract in ``sql/README.md``, then:

    python scripts/deploy_stored_procedures.py
    export BAGT_USE_STORED_PROCEDURES=true

SQLite / PostgreSQL deployments read the portable tblBAGrowthData table and
never need this script.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import sys

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from DatabaseLinker.main import get_raw_connection, require_mssql  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_SQL_DIR = os.path.join(_BACKEND_DIR, "sql")


def deploy_file(cursor, sql_path: str) -> None:
    """Execute all GO-delimited batches in *sql_path*."""
    name = os.path.basename(sql_path)
    logger.info("Deploying %s ...", name)

    with open(sql_path, encoding="utf-8") as fh:
        raw = fh.read()

    batches = [b.strip() for b in re.split(r"^\s*GO\s*$", raw, flags=re.MULTILINE) if b.strip()]
    for i, batch in enumerate(batches, 1):
        try:
            cursor.execute(batch)
            cursor.connection.commit()
            logger.info("  Batch %d OK", i)
        except Exception as exc:
            logger.error("  Batch %d FAILED: %s", i, exc)
            raise
    logger.info("  OK %s deployed", name)


def deploy_all() -> None:
    require_mssql("scripts/deploy_stored_procedures.py")
    files = sorted(glob.glob(os.path.join(_SQL_DIR, "sp_*.sql")))
    if not files:
        logger.warning("No sql/sp_*.sql files found – nothing to deploy. See sql/README.md.")
        return

    logger.info("Connecting to database ...")
    conn = get_raw_connection()
    cursor = conn.cursor()
    try:
        for path in files:
            deploy_file(cursor, path)
    finally:
        cursor.close()
        conn.close()
    logger.info("Deployed %d stored procedure file(s).", len(files))


if __name__ == "__main__":
    deploy_all()
