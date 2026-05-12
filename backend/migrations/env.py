# migrations/env.py
from __future__ import annotations

import os, sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool



# --- Make project root importable ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # …/migrations
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                 # …/user-auth-backend
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.db.base import Base
from app.models.user import User
from app.models.run import Run
from app.models.profile_info import ProfileInfo
from app.models.analysis_result import AnalysisResult

# Load app settings (reads .env)
try:
    from app.core.config import settings  # type: ignore
except Exception as e:
    raise RuntimeError(f"Alembic failed to import app settings: {e!r}") from e

config = context.config

# Optional: logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Use the exact URL from your app settings (.env)
db_url = settings.DATABASE_URL
if not db_url:
    raise RuntimeError("DATABASE_URL is empty. Check your .env.")

print(f"[alembic] using DATABASE_URL = {db_url}")

# We'll write migrations manually for now
target_metadata = Base.metadata

def run_migrations_offline():
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = create_engine(db_url, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
