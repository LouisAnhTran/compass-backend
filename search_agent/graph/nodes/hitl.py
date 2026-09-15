"""The three mandatory human confirmations (SPEC §9).

Every entity resolution interrupts — there is no auto-select path, however
confident the fuzzy score. The score only orders the shortlist.

`display` is composed here, by the backend, per entity type. The frontend
renders it verbatim and never builds these strings itself, so the id-prefixed
queue format only has to be right in one place.
"""

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from search_agent.graph.nodes.rank import display_name
from search_agent.graph.state import SearchAgentState


def _display(candidate: dict, entity: str) -> str:
    if entity == "queue":
        # Queue names collide across groups ("Invoices EU" under two parents),
        # so the id is part of the label, not just the value.
        return f"{candidate['id']} · {candidate['name']}"
    return display_name(candidate, entity)


def _option(candidate: dict, entity: str) -> dict:
    return {
        "id": candidate["id"],
        "name": display_name(candidate, entity),
        "display": _display(candidate, entity),
    }


def _decorate(group: dict, candidates: list[dict], entity: str) -> dict:
    """Attach `display` to each shortlist match, keyed by id off the candidates."""
    by_id = {c["id"]: c for c in candidates}
    return {
        **group,
        "matches": [
            {**m, "display": _display(by_id.get(m["id"], {"id": m["id"], "name": m["name"]}), entity)}
            for m in group["matches"]
        ],
    }


def _payload(reason: str, groups: list[dict], candidates: list[dict], entity: str) -> dict:
    return {
        "reason": reason,
        "items": [_decorate(g, candidates, entity) for g in groups],
        "all_options": [_option(c, entity) for c in candidates],
    }


# ── Nodes ────────────────────────────────────────────────────────────────────

async def hitl_confirm_model(state: SearchAgentState) -> dict:
    candidates = state.get("model_candidates", [])
    resume = interrupt(
        _payload("confirm_model", [state["model_matches"]], candidates, "model")
    )
    picked = resume["selection"]
    return {
        "model_id": picked["id"],
        "model_name": picked["name"],
        "messages": [AIMessage(f"Model: {picked['name']}")],
    }


async def hitl_confirm_queues(state: SearchAgentState) -> dict:
    candidates = state.get("queue_candidates", [])
    resume = interrupt(
        _payload("confirm_queues", state["queue_matches"], candidates, "queue")
    )
    picks = resume["selections"]  # may legitimately be empty — "no queue filter"
    summary = ", ".join(p["name"] for p in picks) if picks else "any"
    return {
        "queue_ids": [p["id"] for p in picks],
        "messages": [AIMessage(f"Queues: {summary}")],
    }


async def hitl_confirm_members(state: SearchAgentState) -> dict:
    candidates = state.get("member_candidates", [])
    resume = interrupt(
        _payload("confirm_members", state["member_matches"], candidates, "member")
    )
    picks = resume["selections"]
    summary = ", ".join(p["name"] for p in picks) if picks else "any"
    return {
        "member_ids": [p["id"] for p in picks],
        "messages": [AIMessage(f"Members: {summary}")],
    }
