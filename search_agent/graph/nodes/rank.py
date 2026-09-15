"""Deterministic fuzzy ranking (SPEC §8).

The score is PRESENTATION ONLY. It decides what appears in the "we found
these" shortlist and in what order. It never auto-selects — every entity goes
to a human regardless of how confident the match looks.
"""

import html
from typing import Literal

from rapidfuzz import fuzz, utils

from search_agent.config import settings
from search_agent.graph.state import SearchAgentState

Entity = Literal["model", "queue", "member"]
MAX_SHORTLIST = 10


def _unescape(text: str) -> str:
    """Staple returns HTML-escaped names, sometimes multiply escaped:
    "Fusable&amp;amp;amp;#39;s" is really "Fusable's". Left alone this breaks
    both rendering AND matching — a user typing "Fusable's" scores near zero
    against the raw entity string. Loop because one pass only peels one layer.
    """
    for _ in range(5):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return text


def _score(needle: str, candidate_text: str) -> float:
    """Best of two scorers, both with default_process (lowercase + strip
    punctuation).

    default_process is NOT optional: token_sort_ratio sorts tokens before
    comparing and that sort is case-sensitive, so "Singapore e-invoice" sorts
    to "Singapore e-invoice" while "Singapore E-invoice" sorts to
    "E-invoice Singapore". Untreated, that exact match scores 0.47.

    token_set_ratio carries the partial case: it compares the token
    intersection, so "Yashdeep" -> "Yashdeep Kumar" and "invoicing" ->
    "Q2 - Invoicing EU" both score 1.0 where token_sort_ratio gives ~0.7 and
    would drop them below threshold. Taking the max keeps full-string matches
    ranked above mere subset matches.
    """
    p = utils.default_process
    return max(
        fuzz.token_sort_ratio(needle, candidate_text, processor=p),
        fuzz.token_set_ratio(needle, candidate_text, processor=p),
    ) / 100.0


def display_name(candidate: dict, entity: Entity) -> str:
    if entity == "member":
        # Real data has trailing spaces in firstName ("Aman ").
        first = (candidate.get("firstName") or "").strip()
        last = (candidate.get("lastName") or "").strip()
        full = f"{first} {last}".strip()
        return _unescape(full) or (candidate.get("email") or "")
    return _unescape((candidate.get("name") or "").strip())


def rank_matches(extracted: str | None, candidates: list[dict], entity: Entity) -> dict:
    """Returns one MatchGroup: {extracted, sub_reason, matches}."""
    if not extracted or not extracted.strip():
        return {"extracted": extracted, "sub_reason": "no_extraction", "matches": []}

    needle = extracted.strip()
    scored = []
    for c in candidates:
        name = display_name(c, entity)
        score = _score(needle, name)

        # Members are usually named partially — "Yashdeep" for "Yashdeep
        # Kumar" — and may be given as an email fragment. Score against the
        # address too and keep the better of the two.
        if entity == "member" and c.get("email"):
            score = max(score, _score(needle, c["email"]))

        # A bare numeric queue id is an exact reference, not a fuzzy one —
        # "2321" scores ~0 against "Q1 - Invoices EU" by string similarity.
        if entity == "queue" and needle.isdigit() and str(c["id"]) == needle:
            score = 1.0

        scored.append({"id": c["id"], "name": name, "score": round(score, 4)})

    above = sorted(
        [s for s in scored if s["score"] >= settings.fuzzy_threshold],
        key=lambda s: -s["score"],
    )
    survivors = above[:MAX_SHORTLIST]

    return {
        "extracted": extracted,
        "sub_reason": "found_matches" if survivors else "no_match",
        "matches": survivors,
        # Real data has 16 queues literally named "Invoice", so a shortlist of
        # 10 can silently hide matches. Tell the UI, so it can say
        # "showing 10 of 47" and point at Browse all rather than implying
        # these are the only candidates.
        "total_above_threshold": len(above),
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
