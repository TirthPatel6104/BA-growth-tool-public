"""
db/bootstrap.py
===============
Schema creation and demo seeding at application start-up.

``init_db()`` is called from the FastAPI lifespan hook and from
``scripts/seed_demo_db.py``.  Behaviour is driven by two env flags whose
defaults depend on the dialect:

    BAGT_AUTO_MIGRATE   run ``alembic upgrade head`` on start-up
                        default: true for SQLite, false otherwise
    BAGT_AUTO_SEED      insert the synthetic demo dataset when the
                        coefficient table is empty
                        default: true for SQLite, false otherwise

Production-style deployments (PostgreSQL / MSSQL) are expected to run
``alembic upgrade head`` explicitly as a deployment step.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from db.config import BACKEND_DIR, get_database_url, get_dialect, redact_url, sqlite_file_path
from db.engine import get_engine, reset_engine, session_scope
from db.models import Base

logger = logging.getLogger(__name__)

ALEMBIC_INI = os.path.join(BACKEND_DIR, "alembic.ini")
ALEMBIC_DIR = os.path.join(BACKEND_DIR, "alembic")


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def alembic_config(url: Optional[str] = None) -> Config:
    """Alembic ``Config`` pointing at ``backend/alembic`` for the given URL."""
    cfg = Config(ALEMBIC_INI)
    cfg.set_main_option("script_location", ALEMBIC_DIR)
    cfg.attributes["configure_logging"] = False
    cfg.attributes["sqlalchemy_url"] = url or get_database_url()
    return cfg


def run_migrations(url: Optional[str] = None, revision: str = "head") -> None:
    """Programmatic ``alembic upgrade <revision>``."""
    url = url or get_database_url()
    logger.info("Running Alembic migrations → %s on %s", revision, redact_url(url))
    command.upgrade(alembic_config(url), revision)


def stamp_head(url: Optional[str] = None) -> None:
    """Mark an existing (pre-Alembic) database as being at ``head`` without running DDL."""
    command.stamp(alembic_config(url), "head")


def create_all(engine: Optional[Engine] = None) -> None:
    """Fallback for environments without Alembic history (tests, scratch DBs)."""
    Base.metadata.create_all(engine or get_engine())


def has_alembic_version(engine: Engine) -> bool:
    return "alembic_version" in inspect(engine).get_table_names()


def is_legacy_database(engine: Engine) -> bool:
    """True when app tables exist but Alembic has never stamped this database."""
    tables = set(inspect(engine).get_table_names())
    return "tblRGRCoefficients" in tables and "alembic_version" not in tables


def _retire_legacy_sqlite_file(path: str) -> str:
    """Move a pre-Alembic SQLite demo file aside (it is synthetic + git-ignored)."""
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup = f"{path}.legacy-{stamp}.bak"
    reset_engine()
    shutil.move(path, backup)
    logger.warning(
        "Legacy pre-Alembic SQLite demo database moved to %s; a fresh one will be created.", backup
    )
    return backup


def init_db(*, migrate: Optional[bool] = None, seed: Optional[bool] = None) -> Dict[str, Any]:
    """
    Make sure the configured database has the current schema and (for the
    demo path) contains data.  Returns a small summary dict for logging/tests.
    """
    url = get_database_url()
    dialect = get_dialect(url)
    is_sqlite = dialect == "sqlite"

    do_migrate = _flag("BAGT_AUTO_MIGRATE", default=is_sqlite) if migrate is None else migrate
    do_seed = _flag("BAGT_AUTO_SEED", default=is_sqlite) if seed is None else seed

    summary: Dict[str, Any] = {
        "url": redact_url(url),
        "dialect": dialect,
        "migrated": False,
        "legacy_backup": None,
        "seeded": False,
        "seed_summary": None,
    }

    engine = get_engine()

    if is_sqlite and do_migrate:
        sqlite_path = sqlite_file_path(url)
        if sqlite_path and os.path.exists(sqlite_path) and is_legacy_database(engine):
            summary["legacy_backup"] = _retire_legacy_sqlite_file(sqlite_path)
            engine = get_engine()

    if do_migrate:
        if not is_sqlite and is_legacy_database(engine):
            raise RuntimeError(
                "Existing pre-Alembic schema detected. Run `alembic stamp head` once "
                "(after verifying the schema matches sql/create_rgr_tables.sql + "
                "sql/add_confidence_columns.sql), then restart."
            )
        run_migrations(url)
        summary["migrated"] = True

    if do_seed:
        from db.seed import demo_database_is_empty, seed_demo_database

        with session_scope() as session:
            if demo_database_is_empty(session):
                summary["seed_summary"] = seed_demo_database(session)
                summary["seeded"] = True
                logger.info("Seeded synthetic demo data: %s", summary["seed_summary"])

    return summary
