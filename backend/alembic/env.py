"""
Alembic environment.

The target URL is resolved the same way the application resolves it
(``db.config.get_database_url``), so ``alembic upgrade head`` always migrates
the database the API would connect to.  Programmatic callers
(``db.bootstrap.run_migrations``) can override the URL or hand in an open
connection through ``config.attributes``.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from db.config import get_database_url  # noqa: E402
from db.engine import build_engine  # noqa: E402
from db.models import Base  # noqa: E402

config = context.config

# Only configure logging when invoked from the CLI – the API sets up its own.
if config.config_file_name is not None and config.attributes.get("configure_logging", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_url() -> str:
    return (
        config.attributes.get("sqlalchemy_url")
        or config.get_main_option("sqlalchemy.url")
        or get_database_url()
    )


def _configure(connection=None, url=None) -> None:
    dialect_name = connection.dialect.name if connection is not None else url.split(":", 1)[0]
    context.configure(
        connection=connection,
        url=url,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=False,
        # SQLite cannot ALTER most things in place; batch mode rebuilds the table.
        render_as_batch=dialect_name.startswith("sqlite"),
        literal_binds=connection is None,
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live connection (``alembic upgrade head --sql``)."""
    _configure(url=_resolve_url())
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    existing = config.attributes.get("connection")
    if existing is not None:
        _configure(connection=existing)
        with context.begin_transaction():
            context.run_migrations()
        return

    engine = build_engine(_resolve_url())
    try:
        with engine.connect() as connection:
            _configure(connection=connection)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
