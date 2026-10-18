"""
FastAPI Dependency Injection Helpers
====================================

    from api.deps import DbSession, CoefficientStoreDep, ExtractorDep

    @app.get("/example")
    def example(db: DbSession, store: CoefficientStoreDep):
        ...

``get_db`` yields one SQLAlchemy ``Session`` per request and always closes it.
The store / extractor providers wrap that same session so a request works
inside a single unit of work.  Tests override ``get_db`` with
``app.dependency_overrides[get_db] = ...`` to point at a scratch database.
"""

from __future__ import annotations

from typing import Annotated, Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from db.engine import get_session


def get_db() -> Generator[Session, None, None]:
    """Dependency provider: one database session per request."""
    db = get_session()
    try:
        yield db
    finally:
        db.close()


DbSession = Annotated[Session, Depends(get_db)]


def get_coefficient_store(db: DbSession):
    """Dependency provider: ``CoefficientStore`` bound to the request session."""
    from models.coefficient_store import CoefficientStore  # local import: avoids import cycles

    return CoefficientStore(db)


def get_extractor(db: DbSession):
    """Dependency provider: ``BAGrowthDataExtractor`` bound to the request session."""
    from data.extractors import BAGrowthDataExtractor

    return BAGrowthDataExtractor(db, populate_tcr=True)


# Annotated aliases – keep endpoint signatures short.
from models.coefficient_store import CoefficientStore  # noqa: E402
from data.extractors import BAGrowthDataExtractor  # noqa: E402

CoefficientStoreDep = Annotated[CoefficientStore, Depends(get_coefficient_store)]
ExtractorDep = Annotated[BAGrowthDataExtractor, Depends(get_extractor)]
