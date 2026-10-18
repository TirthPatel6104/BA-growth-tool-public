"""
DatabaseLinker/main.py
======================
Backward-compatible entry point for database access.

Historically this module built a pyodbc connection string for SQL Server.
The data layer now runs on SQLAlchemy 2.0 (see ``db/``), so
``get_db_connection()`` returns a SQLAlchemy ``Session``.  Existing callers
(``training/pipeline.py``, ``prediction/predictor.py``, scripts) keep working
because a ``Session`` offers the same ``commit()`` / ``rollback()`` /
``close()`` surface the old connection object exposed.

Configuration is env-driven (``db/config.py``):

    DATABASE_URL=postgresql://...            PostgreSQL
    DATABASE_URL=mssql+pyodbc://...          SQL Server  (optional pyodbc driver)
    BAGT_SQL_SERVER / BAGT_SQL_CONNECTION_STRING   legacy SQL Server keywords
    (nothing)                                SQLite file backend/demo_basal_growth.db

New code should prefer ``api.deps.get_db`` (FastAPI) or
``db.engine.session_scope()`` (scripts) over this helper.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from db.config import get_database_url, get_dialect, is_mssql, redact_url  # noqa: F401
from db.engine import (  # noqa: F401
    get_engine,
    get_raw_connection,
    get_server_version,
    get_session,
    list_tables,
    ping,
    session_scope,
)


def get_db_connection() -> Session:
    """Open a new SQLAlchemy session for the configured database (caller closes it)."""
    return get_session()


def require_mssql(what: str = "This operation") -> None:
    """Raise a clear error when a T-SQL-only script runs against SQLite / PostgreSQL."""
    if not is_mssql():
        raise RuntimeError(
            f"{what} requires Microsoft SQL Server (stored procedures / T-SQL), but the "
            f"configured database is {get_dialect()} ({redact_url()}). Set DATABASE_URL to an "
            "mssql+pyodbc:// URL or the BAGT_SQL_* variables and install requirements-mssql.txt."
        )


__all__ = [
    "get_db_connection",
    "get_engine",
    "get_raw_connection",
    "get_session",
    "get_server_version",
    "list_tables",
    "ping",
    "require_mssql",
    "session_scope",
]
