"""Deterministic fuzzy ranking (SPEC §8).

The score is PRESENTATION ONLY. It decides what appears in the "we found
these" shortlist and in what order. It never auto-selects — every entity goes
to a human regardless of how confident the match looks.
"""

from typing import Literal

from rapidfuzz import fuzz

from search_agent.config import settings
from search_agent.graph.state import SearchAgentState

Entity = Literal["model", "queue", "member"]
MAX_SHORTLIST = 10


def display_name(candidate: dict, entity: Entity) -> str:
    if entity == "member":
        first = (candidate.get("firstName") or "").strip()
        last = (candidate.get("lastName") or "").strip()
        return f"{first} {last}".strip() or (candidate.get("email") or "")
    return candidate["name"]


def rank_matches(extracted: str | None, candidates: list[dict], entity: Entity) -> dict:
    """Returns one MatchGroup: {extracted, sub_reason, matches}."""
    if not extracted or not extracted.strip():
        return {"extracted": extracted, "sub_reason": "no_extraction", "matches": []}

    needle = extracted.strip()
    scored = []
    for c in candidates:
        name = display_name(c, entity)
        score = fuzz.token_sort_ratio(needle, name) / 100.0

        # A bare numeric queue id is an exact reference, not a fuzzy one —
        # "2321" scores ~0 against "Q1 - Invoices EU" by string similarity.
        if entity == "queue" and needle.isdigit() and str(c["id"]) == needle:
            score = 1.0

        scored.append({"id": c["id"], "name": name, "score": round(score, 4)})

    survivors = sorted(
        [s for s in scored if s["score"] >= settings.fuzzy_threshold],
        key=lambda s: -s["score"],
    )[:MAX_SHORTLIST]

    return {
        "extracted": extracted,
        "sub_reason": "found_matches" if survivors else "no_match",
        "matches": survivors,
    }


# ── Nodes ────────────────────────────────────────────────────────────────────

async def rank_model_matches(state: SearchAgentState) -> dict:
    extracted = state.get("extracted_models") or []
    # Single-model rule: at most one. Anything beyond the first is dropped here
    # rather than surfaced, so the HITL payload stays length-1 by construction.
    first = extracted[0] if extracted else None
    return {"model_matches": rank_matches(first, state.get("model_candidates", []), "model")}


async def rank_queue_matches(state: SearchAgentState) -> dict:
    extracted = state.get("extracted_queues") or []
    candidates = state.get("queue_candidates", [])
    # Empty extraction still produces one group, so the HITL can offer the full
    # dropdown rather than being skipped.
    groups = [rank_matches(e, candidates, "queue") for e in extracted] or [
        rank_matches(None, candidates, "queue")
    ]
    return {"queue_matches": groups}


async def rank_member_matches(state: SearchAgentState) -> dict:
    extracted = state.get("extracted_members") or []
    candidates = state.get("member_candidates", [])
    return {"member_matches": [rank_matches(e, candidates, "member") for e in extracted]}
