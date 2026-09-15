"""Graph assembly (SPEC §4-§6).

Thirteen nodes, one conditional edge, three mandatory interrupts. Flat — no
subgraphs and no intent classification.
"""

from langgraph.graph import END, START, StateGraph

from search_agent.graph.nodes import (
    build_filter_payload,
    extract_model,
    extract_other_entities,
    fetch_members,
    fetch_models,
    fetch_queues,
    hitl_confirm_members,
    hitl_confirm_model,
    hitl_confirm_queues,
    rank_member_matches,
    rank_model_matches,
    rank_queue_matches,
    search_documents,
)
from search_agent.graph.state import SearchAgentState


def _route_after_queues(state: SearchAgentState) -> str:
    """The graph's only conditional edge. Members are optional; an empty
    extraction skips fetch/rank/confirm entirely."""
    return "build_filter_payload" if not state.get("extracted_members") else "fetch_members"


def build_graph(checkpointer):
    b = StateGraph(SearchAgentState)

    for fn in (
        extract_model,
        fetch_models,
        rank_model_matches,
        hitl_confirm_model,
        extract_other_entities,
        fetch_queues,
        rank_queue_matches,
        hitl_confirm_queues,
        fetch_members,
        rank_member_matches,
        hitl_confirm_members,
        build_filter_payload,
        search_documents,
    ):
        b.add_node(fn.__name__, fn)

    b.add_edge(START, "extract_model")
    b.add_edge("extract_model", "fetch_models")
    b.add_edge("fetch_models", "rank_model_matches")
    b.add_edge("rank_model_matches", "hitl_confirm_model")
    b.add_edge("hitl_confirm_model", "extract_other_entities")
    b.add_edge("extract_other_entities", "fetch_queues")
    b.add_edge("fetch_queues", "rank_queue_matches")
    b.add_edge("rank_queue_matches", "hitl_confirm_queues")

    b.add_conditional_edges(
        "hitl_confirm_queues",
        _route_after_queues,
        {"fetch_members": "fetch_members", "build_filter_payload": "build_filter_payload"},
    )

    b.add_edge("fetch_members", "rank_member_matches")
    b.add_edge("rank_member_matches", "hitl_confirm_members")
    b.add_edge("hitl_confirm_members", "build_filter_payload")
    b.add_edge("build_filter_payload", "search_documents")
    b.add_edge("search_documents", END)

    return b.compile(checkpointer=checkpointer)
