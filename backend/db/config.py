"""
db/config.py
============
Resolve the SQLAlchemy database URL from environment variables.

Precedence (first match wins):

1. ``DATABASE_URL``                – any SQLAlchemy URL, e.g.
                                     ``postgresql://user:pw@host:5432/bagt``
                                     ``sqlite:///./demo_basal_growth.db``
                                     ``mssql+pyodbc:///?odbc_connect=...``
2. ``BAGT_SQL_CONNECTION_STRING``  – raw ODBC connection string (legacy MSSQL path)
3. ``BAGT_SQL_SERVER``             – legacy keyword-style MSSQL settings
                                     (BAGT_SQL_DRIVER / DATABASE / TRUSTED / USER / PASSWORD ...)
4. default                         – SQLite file ``backend/demo_basal_growth.db``
                                     (override the path with ``BAGT_SQLITE_PATH``)

The MSSQL paths require the optional ``pyodbc`` package
(``pip install -r requirements-mssql.txt``).
"""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import quote_plus

try:  # python-dotenv is in requirements.txt but keep the import optional
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None  # type: ignore[assignment]

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SQLITE_FILENAME = "demo_basal_growth.db"
DEFAULT_SQLITE_PATH = os.path.join(BACKEND_DIR, DEFAULT_SQLITE_FILENAME)

if load_dotenv is not None:
    # Never override variables that are already exported in the shell.
    load_dotenv(os.path.join(BACKEND_DIR, ".env"), override=False)


def _truthy(value: Optional[str], default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _sqlite_url(path: str) -> str:
    # SQLAlchemy wants forward slashes even on Windows: sqlite:///E:/path/file.db
    normalized = os.path.abspath(path).replace("\\", "/")
    return f"sqlite:///{normalized}"


def _legacy_mssql_odbc_string() -> Optional[str]:
    """Rebuild the ODBC string the old DatabaseLinker assembled from BAGT_SQL_* vars."""
    explicit = os.getenv("BAGT_SQL_CONNECTION_STRING", "").strip()
    if explicit:
        return explicit

    server = os.getenv("BAGT_SQL_SERVER", "").strip()
    if not server:
        return None

    driver = os.getenv("BAGT_SQL_DRIVER", "ODBC Driver 17 for SQL Server")
    database = os.getenv("BAGT_SQL_DATABASE", "demo_growth_db")
    encrypt = os.getenv("BAGT_SQL_ENCRYPT", "no")
    trust_server_cert = os.getenv("BAGT_SQL_TRUST_SERVER_CERT", "yes")
    trusted = _truthy(os.getenv("BAGT_SQL_TRUSTED"), default=True)

    if trusted:
        auth_segment = "Trusted_Connection=yes;"
    else:
        username = os.getenv("BAGT_SQL_USER", "demo_user")
        password = os.getenv("BAGT_SQL_PASSWORD", "demo_password")
        auth_segment = f"UID={username};PWD={password};"

    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Encrypt={encrypt};"
        f"TrustServerCertificate={trust_server_cert};"
        f"{auth_segment}"
    )


def normalize_database_url(url: str) -> str:
    """Normalise shorthand URLs (Heroku-style ``postgres://``) to SQLAlchemy 2.0 form."""
    url = url.strip()
    if url.startswith("postgres://"):
        url = "postgresql+psycopg2://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    elif url.startswith("mssql://"):
        url = "mssql+pyodbc://" + url[len("mssql://"):]
    return url


def get_database_url() -> str:
    """Return the SQLAlchemy URL for the configured database (see module docstring)."""
    explicit = os.getenv("DATABASE_URL", "").strip()
    if explicit:
        return normalize_database_url(explicit)

    odbc = _legacy_mssql_odbc_string()
    if odbc:
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}"

    sqlite_path = os.getenv("BAGT_SQLITE_PATH", "").strip() or DEFAULT_SQLITE_PATH
    return _sqlite_url(sqlite_path)


def get_dialect(url: Optional[str] = None) -> str:
    """Return ``'sqlite'``, ``'postgresql'`` or ``'mssql'`` for the configured URL."""
    url = url or get_database_url()
    scheme = url.split(":", 1)[0].lower()
    return scheme.split("+", 1)[0]


def is_sqlite(url: Optional[str] = None) -> bool:
    return get_dialect(url) == "sqlite"


def is_postgres(url: Optional[str] = None) -> bool:
    return get_dialect(url) == "postgresql"


def is_mssql(url: Optional[str] = None) -> bool:
    return get_dialect(url) == "mssql"


def sqlite_file_path(url: Optional[str] = None) -> Optional[str]:
    """Return the on-disk file for a ``sqlite:///`` URL (None for in-memory / other dialects)."""
    url = url or get_database_url()
    if not is_sqlite(url):
        return None
    path = url[len("sqlite:///"):]
    if not path or path == ":memory:":
        return None
    return path


def redact_url(url: Optional[str] = None) -> str:
    """URL safe for logging (password / ODBC secrets stripped)."""
    url = url or get_database_url()
    if is_mssql(url):
        return "mssql+pyodbc://<odbc_connect redacted>"
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        creds, host = rest.rsplit("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return url
