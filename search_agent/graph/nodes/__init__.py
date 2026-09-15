"""Node stubs — backbone only. Each raises until implemented.

Signatures are fixed by SPEC §6. Nodes that call Staple (fetch_models,
fetch_queues, fetch_members, search_documents) read the bearer token from
`config["configurable"]["authorization"]`, never from state.
"""

from langgraph.types import interrupt

from search_agent.graph.state import SearchAgentState

_TODO = "not implemented — see SPEC.md §6"


# ── Model ────────────────────────────────────────────────────────────────────

async def extract_model(state: SearchAgentState) -> dict:
    raise NotImplementedError(_TODO)


async def fetch_models(state: SearchAgentState, config) -> dict:
    raise NotImplementedError(_TODO)


async def rank_model_matches(state: SearchAgentState) -> dict:
    raise NotImplementedError(_TODO)


async def hitl_confirm_model(state: SearchAgentState) -> dict:
    picked = interrupt(...)["selection"]        # always interrupts
    return {"model_id": picked["id"], "model_name": picked["name"]}


# ── Queues ───────────────────────────────────────────────────────────────────

async def extract_other_entities(state: SearchAgentState) -> dict:
    """One LLM call → extracted_queues, extracted_members, date_range."""
    raise NotImplementedError(_TODO)


async def fetch_queues(state: SearchAgentState, config) -> dict:
    raise NotImplementedError(_TODO)


async def rank_queue_matches(state: SearchAgentState) -> dict:
    raise NotImplementedError(_TODO)


async def hitl_confirm_queues(state: SearchAgentState) -> dict:
    picks = interrupt(...)["selections"]        # list, possibly empty
    return {"queue_ids": [p["id"] for p in picks]}


# ── Members ──────────────────────────────────────────────────────────────────

async def fetch_members(state: SearchAgentState, config) -> dict:
    raise NotImplementedError(_TODO)


async def rank_member_matches(state: SearchAgentState) -> dict:
    raise NotImplementedError(_TODO)


async def hitl_confirm_members(state: SearchAgentState) -> dict:
    picks = interrupt(...)["selections"]
    return {"member_ids": [p["id"] for p in picks]}


# ── Search ───────────────────────────────────────────────────────────────────

async def build_filter_payload(state: SearchAgentState) -> dict:
    """SPEC §12. modelId only — docType is an independent filter dimension and
    is not used. The extracted_members guard prevents a previous query's
    member_ids leaking into a new query on the same thread."""
    filter_dict = {"others": {"modelId": state["model_id"], "fields": []}}

    if state.get("queue_ids"):
        filter_dict["queue"] = state["queue_ids"]
    if state.get("extracted_members") and state.get("member_ids"):
        filter_dict["uploadedBy"] = state["member_ids"]
    if state.get("date_range"):
        filter_dict["uploadedAt"] = state["date_range"]

    return {"filter_payload": {"filter": filter_dict, "pagination": {"skip": 0, "take": 10}}}


async def search_documents(state: SearchAgentState, config) -> dict:
    raise NotImplementedError(_TODO)
