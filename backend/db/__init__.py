"""
db package
==========
SQLAlchemy 2.0 data layer for the Basal Area Growth Tool.

    config.py    – resolve the database URL from environment variables
    engine.py    – engine / session factory, FastAPI-friendly session helpers
    models.py    – ORM models mirroring the historical MSSQL schema
    bootstrap.py – Alembic-driven schema creation + demo seeding on startup
    seed.py      – synthetic demo dataset (coefficients + growth intervals)

Supported dialects: SQLite (zero-config default), PostgreSQL (DATABASE_URL),
and Microsoft SQL Server via the optional ``pyodbc`` driver.
"""

from db.config import get_database_url, get_dialect, is_mssql, is_sqlite  # noqa: F401
from db.engine import (  # noqa: F401
    get_engine,
    get_session,
    get_session_factory,
    reset_engine,
    session_scope,
)
from db.models import Base  # noqa: F401
