from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages


class Match(TypedDict):
    id: int
    name: str
    score: float


class MatchGroup(TypedDict):
    extracted: Optional[str]
    sub_reason: Literal["no_extraction", "found_matches", "no_match"]
    matches: list[Match]


class DateRange(TypedDict):
    # dd/mm/yyyy — confirmed by the server's own advanced-search adapter tests.
    from_: str
    to: str


class SearchAgentState(TypedDict, total=False):
    """Only `messages` accumulates. Every other field REPLACES on write.

    Stale-field hazard: a field whose node does not run keeps its previous
    value. On a second query in the same thread with no members mentioned, the
    member subgraph is skipped and member_ids silently retains the previous
    query's ids. build_filter guards against this by checking extracted_members.
    """

    messages: Annotated[list, add_messages]
    query: str

    # Model — required, resolves to exactly one id.
    extracted_models: list[str]
    model_candidates: list[dict]
    model_matches: MatchGroup
    model_id: Optional[int]
    model_name: Optional[str]

    # Queues — multi. Empty extraction still triggers HITL.
    extracted_queues: list[str]
    queue_candidates: list[dict]
    queue_matches: list[MatchGroup]
    queue_ids: list[int]

    # Members — multi and optional. Empty extraction skips the whole subgraph.
    extracted_members: list[str]
    member_candidates: list[dict]
    member_matches: list[MatchGroup]
    member_ids: list[int]

    date_range: Optional[DateRange]

    filter_payload: dict
    results: Optional[dict]

    status: Literal["initializing", "awaiting_input", "searching", "completed", "failed"]
    error: Optional[str]
