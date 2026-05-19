import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.deps.db import get_db
from app.main import app
from app.models.user import User
from app.models.refresh_token import RefreshToken
from app.models.run import Run

# StaticPool ensures all sessions share one connection, so test data is
# visible to the HTTP request handler without cross-connection isolation issues.
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Only create the tables needed for auth tests (avoids JSONB incompatibility in SQLite)
_AUTH_TABLES = [User.__table__, RefreshToken.__table__]
_RUN_TABLES = [User.__table__, Run.__table__]


def _make_client_fixture(tables: list):
    """Return a (db, client) fixture pair that creates only the given tables."""
    @pytest.fixture()
    def _db():
        for table in tables:
            table.create(bind=engine, checkfirst=True)
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()
            for table in reversed(tables):
                table.drop(bind=engine, checkfirst=True)
            for table in tables:
                table.create(bind=engine, checkfirst=True)

    @pytest.fixture()
    def _client(_db):
        def override_get_db():
            session = TestingSessionLocal()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()

    return _db, _client


@pytest.fixture()
def db():
    for table in _AUTH_TABLES:
        table.create(bind=engine, checkfirst=True)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        for table in reversed(_AUTH_TABLES):
            table.drop(bind=engine, checkfirst=True)
        for table in _AUTH_TABLES:
            table.create(bind=engine, checkfirst=True)


@pytest.fixture()
def client(db):
    # Each request gets its own session from the same SQLite engine.
    # Test data must be committed (not just flushed) before requests are made,
    # so that these per-request sessions can see it.
    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def run_db():
    for table in _RUN_TABLES:
        table.create(bind=engine, checkfirst=True)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        for table in reversed(_RUN_TABLES):
            table.drop(bind=engine, checkfirst=True)
        for table in _RUN_TABLES:
            table.create(bind=engine, checkfirst=True)


@pytest.fixture()
def run_client(run_db):
    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
