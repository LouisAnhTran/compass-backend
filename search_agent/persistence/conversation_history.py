"""Projection over LangGraph's checkpoints.

checkpoints is the source of truth; this table exists so the sidebar and the
ownership check are cheap reads instead of get_state_history() scans. Written
by the endpoint after every invoke — never by a graph node.
"""

import json

from psycopg_pool import AsyncConnectionPool

from search_agent.config import settings

# open=False so constructing the module does not perform I/O at import time.
# The app lifespan opens it; without that every query raises PoolClosed.
_pool = AsyncConnectionPool(settings.postgres_url, open=False)


async def open_pool() -> None:
    await _pool.open(wait=True, timeout=30)


async def close_pool() -> None:
    await _pool.close()


async def owner_of(thread_id: str) -> int | None:
    async with _pool.connection() as conn:
        cur = await conn.execute(
            "SELECT user_id FROM conversation_history WHERE thread_id = %s", (thread_id,)
        )
        row = await cur.fetchone()
    return row[0] if row else None


def _serialize_messages(messages: list) -> list[dict]:
    """LangChain message objects are not JSON-serialisable, and json.dumps with
    default=str silently stores their repr:

        "content='show me …' additional_kwargs={} response_metadata={} id='…'"

    which the frontend then renders verbatim, with no role to distinguish a
    user turn from an assistant one. Normalise to the same {role, content}
    shape FastAPI produces for the /search-agent response, so both endpoints
    return identical message objects.
    """
    out = []
    for m in messages:
        if isinstance(m, dict):
            out.append({"role": m.get("role") or m.get("type"), "content": m.get("content", "")})
            continue
        # LangChain: .type is "human" | "ai" | "tool".
        role = getattr(m, "type", None) or getattr(m, "role", None)
        out.append({"role": role, "content": getattr(m, "content", str(m))})
    return out


async def project(thread_id: str, user_id: int, snapshot) -> None:
    awaiting = snapshot.interrupts[0].value if snapshot.next else None
    status = "awaiting_input" if snapshot.next else "completed"
    messages = _serialize_messages(snapshot.values.get("messages", []))

    async with _pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO conversation_history
              (thread_id, user_id, title, messages, awaiting_input, status, updated_at)
            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, now())
            ON CONFLICT (thread_id) DO UPDATE SET
              messages       = EXCLUDED.messages,
              awaiting_input = EXCLUDED.awaiting_input,
              status         = EXCLUDED.status,
              updated_at     = now();
            """,
            # user_id is deliberately absent from DO UPDATE SET — ownership is
            # fixed at creation and must never be reassigned by a later write.
            (
                thread_id,
                user_id,
                _derive_title(messages),
                json.dumps(messages),
                json.dumps(awaiting) if awaiting else None,
                status,
            ),
        )


async def list_for_user(user_id: int, limit: int, offset: int) -> list[dict]:
    async with _pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT thread_id, title, status, updated_at
              FROM conversation_history
             WHERE user_id = %s
             ORDER BY updated_at DESC
             LIMIT %s OFFSET %s
            """,
            (user_id, limit, offset),
        )
        rows = await cur.fetchall()

    return [
        {"thread_id": r[0], "title": r[1], "status": r[2], "updated_at": r[3].isoformat()}
        for r in rows
    ]


async def get(thread_id: str) -> dict | None:
    async with _pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT thread_id, title, messages, awaiting_input, status, updated_at
              FROM conversation_history WHERE thread_id = %s
            """,
            (thread_id,),
        )
        row = await cur.fetchone()

    if row is None:
        return None
    return {
        "thread_id": row[0],
        "title": row[1],
        "messages": row[2],
        "awaiting_input": row[3],
        "status": row[4],
        "updated_at": row[5].isoformat(),
    }


def _derive_title(messages: list[dict]) -> str | None:
    """First human turn — the assistant's are just "Model: …" confirmations."""
    for m in messages:
        if m.get("role") in ("human", "user") and m.get("content"):
            return str(m["content"])[:80]
    return next((str(m["content"])[:80] for m in messages if m.get("content")), None)
