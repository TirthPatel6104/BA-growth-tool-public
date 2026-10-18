"""
db/engine.py
============
Engine and session management.

* ``get_engine()``          – process-wide cached ``Engine`` for the configured URL
* ``get_session_factory()`` – cached ``sessionmaker``
* ``get_session()``         – new ``Session`` (caller closes it)
* ``session_scope()``       – context manager: commit on success, rollback on error
* ``reset_engine()``        – dispose + forget the cached engine (tests / URL change)

Dialect-specific behaviour lives here so the rest of the code base can stay
dialect-agnostic: SQLite gets ``PRAGMA foreign_keys=ON``; MSSQL honours the
legacy ``BAGT_SQL_LOGIN_TIMEOUT`` / ``BAGT_SQL_QUERY_TIMEOUT`` variables.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from db.config import get_database_url, get_dialect, redact_url, sqlite_file_path

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_engine: Optional[Engine] = None
_engine_url: Optional[str] = None
_session_factory: Optional[sessionmaker] = None


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _engine_kwargs(url: str) -> Dict[str, Any]:
    dialect = get_dialect(url)
    kwargs: Dict[str, Any] = {
        "echo": _truthy(os.getenv("BAGT_SQL_ECHO")),
        "pool_pre_ping": True,
    }
    if dialect == "sqlite":
        # FastAPI may touch a session from a different thread than the one that opened it.
        kwargs["connect_args"] = {"check_same_thread": False}
    elif dialect == "mssql":
        login_timeout = int(os.getenv("BAGT_SQL_LOGIN_TIMEOUT", "60"))
        kwargs["connect_args"] = {"timeout": login_timeout}
        kwargs["fast_executemany"] = True
    return kwargs


def _register_events(engine: Engine) -> None:
    dialect = engine.dialect.name

    if dialect == "sqlite":

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    elif dialect == "mssql":
        query_timeout = int(os.getenv("BAGT_SQL_QUERY_TIMEOUT", "0"))  # 0 = no timeout

        @event.listens_for(engine, "connect")
        def _mssql_query_timeout(dbapi_conn, _record):  # noqa: ANN001
            try:
                dbapi_conn.timeout = query_timeout
            except Exception:  # noqa: BLE001 – not every pyodbc build supports it
                pass


def build_engine(url: Optional[str] = None) -> Engine:
    """Create a *new* engine (not cached).  Prefer :func:`get_engine`."""
    url = url or get_database_url()
    sqlite_path = sqlite_file_path(url)
    if sqlite_path:
        os.makedirs(os.path.dirname(os.path.abspath(sqlite_path)), exist_ok=True)
    engine = create_engine(url, **_engine_kwargs(url))
    _register_events(engine)
    logger.info("Database engine ready: %s (%s)", redact_url(url), engine.dialect.name)
    return engine


def get_engine() -> Engine:
    """Return the shared engine, creating it on first use (or after :func:`reset_engine`)."""
    global _engine, _engine_url, _session_factory
    url = get_database_url()
    with _lock:
        if _engine is None or _engine_url != url:
            if _engine is not None:
                _engine.dispose()
            _engine = build_engine(url)
            _engine_url = url
            _session_factory = None
        return _engine


def get_session_factory() -> sessionmaker:
    global _session_factory
    engine = get_engine()
    with _lock:
        if _session_factory is None:
            _session_factory = sessionmaker(
                bind=engine,
                autoflush=False,
                expire_on_commit=False,
                class_=Session,
            )
        return _session_factory


def get_session() -> Session:
    """Open a new session.  The caller is responsible for ``close()``."""
    return get_session_factory()()


@contextmanager
def session_scope() -> Iterator[Session]:
    """``with session_scope() as db:`` – commits on success, rolls back on error."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Dispose the cached engine so the next call re-reads the configuration."""
    global _engine, _engine_url, _session_factory
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _engine_url = None
        _session_factory = None


# ---------------------------------------------------------------------------
# Small dialect-aware helpers used by the API + legacy scripts
# ---------------------------------------------------------------------------

def get_raw_connection():
    """
    DBAPI-level connection (``.cursor()`` / ``.commit()`` / ``.close()``).

    Only needed by the legacy T-SQL deployment scripts under ``scripts/``;
    application code should use sessions.
    """
    return get_engine().raw_connection()


def get_server_version(session: Session) -> str:
    dialect = session.get_bind().dialect.name
    if dialect == "sqlite":
        row = session.execute(text("SELECT sqlite_version()")).scalar()
        return f"SQLite {row}"
    if dialect == "postgresql":
        return str(session.execute(text("SELECT version()")).scalar())
    if dialect == "mssql":
        return str(session.execute(text("SELECT @@VERSION")).scalar())
    return f"{dialect} (version query not implemented)"


def list_tables(engine: Optional[Engine] = None) -> List[str]:
    engine = engine or get_engine()
    return sorted(inspect(engine).get_table_names())


def ping(session: Session) -> bool:
    try:
        session.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False
