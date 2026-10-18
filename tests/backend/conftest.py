"""
tests/backend/conftest.py
=========================
Shared fixtures for the backend test suite.

Every test runs against a throw-away SQLite database created with the real
Alembic migrations and populated with the synthetic demo seed – no SQL
Server (or any database server) is required.  Set ``BAGT_TEST_DATABASE_URL``
to point the suite at PostgreSQL instead (e.g. in a CI service container).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# --- Point the data layer at a scratch database BEFORE any backend import ----
_TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="bagt_test_db_"))
TEST_DATABASE_URL = os.getenv("BAGT_TEST_DATABASE_URL") or (
    "sqlite:///" + (_TEST_DB_DIR / "test_bagt.db").as_posix()
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ.pop("BAGT_SQL_CONNECTION_STRING", None)
os.environ.pop("BAGT_SQL_SERVER", None)
os.environ["BAGT_AUTO_MIGRATE"] = "true"
os.environ["BAGT_AUTO_SEED"] = "false"  # fixtures seed explicitly


@pytest.fixture(scope="session")
def database_url() -> str:
    return TEST_DATABASE_URL


@pytest.fixture(scope="session")
def migrated_db(database_url: str) -> Iterator[str]:
    """Scratch database with the schema created by ``alembic upgrade head``."""
    from db.bootstrap import run_migrations
    from db.engine import reset_engine

    reset_engine()
    run_migrations(database_url)
    yield database_url
    reset_engine()


@pytest.fixture(scope="session")
def seeded_db(migrated_db: str) -> Iterator[str]:
    """Migrated database populated with the synthetic demo dataset."""
    from db.engine import session_scope
    from db.seed import seed_demo_database

    with session_scope() as session:
        seed_demo_database(session, reset=True)
    yield migrated_db


@pytest.fixture
def db_session(seeded_db: str):
    """A fresh SQLAlchemy session per test (rolled back + closed afterwards)."""
    from db.engine import get_session

    session = get_session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client(seeded_db: str):
    """FastAPI TestClient wired to the scratch database through ``api.deps.get_db``."""
    from fastapi.testclient import TestClient

    from api.deps import get_db
    from db.engine import get_session
    from main import _clear_runtime_caches, app

    def _override_get_db():
        session = get_session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db
    _clear_runtime_caches()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)
        _clear_runtime_caches()
