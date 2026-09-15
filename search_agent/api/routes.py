"""HTTP surface. Three endpoints — there is deliberately no /login (SPEC §2.1):
the frontend authenticates against Staple directly and passes the JWT through.
"""

from fastapi import APIRouter, Header, HTTPException, Request
from langgraph.types import Command

from search_agent.auth.identity import resolve_user_id
from search_agent.persistence import conversation_history as history

router = APIRouter()


async def _authorize_thread(thread_id: str, user_id: int) -> None:
    """SPEC §2.5. thread_id is FE-generated and opaque, so without this anyone
    who guesses or replays one resumes that thread and reads its results.

    404 rather than 403 — a 403 confirms the thread exists.
    """
    owner = await history.owner_of(thread_id)
    if owner is not None and owner != user_id:
        raise HTTPException(404)


@router.post("/search-agent")
async def search_agent(
    req: Request,
    body: dict,
    authorization: str | None = Header(None),
):
    user_id = await resolve_user_id(authorization)
    thread_id = body["thread_id"]
    await _authorize_thread(thread_id, user_id)

    graph = req.app.state.graph
    config = {
        "configurable": {
            "thread_id": thread_id,
            # Per-invocation, NOT in graph state: state is checkpointed to
            # Postgres, so a token in state would outlive its session on disk.
            "authorization": authorization,
        }
    }

    snapshot = await graph.aget_state(config)
    payload = body["input"]

    if not snapshot.created_at:
        await graph.ainvoke({"query": payload["query"], "status": "initializing"}, config)
    elif snapshot.next:
        await graph.ainvoke(Command(resume=payload), config)
    else:
        await graph.ainvoke({"query": payload["query"]}, config)

    snapshot = await graph.aget_state(config)
    await history.project(thread_id, user_id, snapshot)
    return _format_response(snapshot, thread_id)


@router.get("/conversations")
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    authorization: str | None = Header(None),
):
    user_id = await resolve_user_id(authorization)
    return await history.list_for_user(user_id, limit, offset)


@router.get("/conversations/{thread_id}")
async def get_conversation(thread_id: str, authorization: str | None = Header(None)):
    user_id = await resolve_user_id(authorization)
    await _authorize_thread(thread_id, user_id)
    row = await history.get(thread_id)
    if row is None:
        raise HTTPException(404)
    return row


def _format_response(snapshot, thread_id: str) -> dict:
    if snapshot.next:
        return {
            "status": "awaiting_input",
            "interrupt_payload": snapshot.interrupts[0].value,
            "thread_id": thread_id,
            "messages": snapshot.values.get("messages", []),
        }
    return {
        "status": "completed",
        "results": snapshot.values.get("results"),
        "thread_id": thread_id,
        "messages": snapshot.values.get("messages", []),
    }
