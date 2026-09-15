"""Postgres checkpointer + application-schema bootstrap.

Two distinct sets of tables live in this database, created at different times
by different owners:

  1. LangGraph's checkpoint tables — created by `AsyncPostgresSaver.setup()`.
     Library-owned. Idempotent. Safe to call on every boot.

  2. conversation_history — ours, defined in schema.sql. Applied explicitly via
     `make migrate`, NOT on boot. See apply_schema() below for why.
"""

from pathlib import Path

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection

SCHEMA_SQL = Path(__file__).resolve().parents[2] / "schema.sql"


async def setup_checkpointer(checkpointer: AsyncPostgresSaver) -> None:
    """Create/upgrade LangGraph's four checkpoint tables.

    Idempotent, and version-aware: `setup()` consults checkpoint_migrations and
    applies only what is missing. It MUST run before the first invoke, and must
    match the langgraph-checkpoint-postgres version pinned in pyproject.toml —
    local and deployed share one database, so a version skew here means one side
    migrates the schema out from under the other.
    """
    await checkpointer.setup()


async def apply_schema(postgres_url: str) -> None:
    """Apply schema.sql (conversation_history).

    Deliberately NOT wired into the app lifespan. Compass shares a single remote
    database between local dev and the deployed instance, so DDL-on-boot means
    concurrent CREATE TABLE from two processes — which CREATE TABLE IF NOT
    EXISTS does not make safe (it can still deadlock on the system catalogs).

    Run it once, on purpose, from `make migrate`.
    """
    async with await AsyncConnection.connect(postgres_url, autocommit=True) as conn:
        await conn.execute(SCHEMA_SQL.read_text())
