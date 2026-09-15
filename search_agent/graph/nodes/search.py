"""Filter construction and the search call (SPEC §12)."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from search_agent.clients.graphql import gql
from search_agent.graph.state import SearchAgentState

SEARCH_QUERY = """
query searchDocumentsByAdvancedFilter(
  $filter: DocumentsAdvancedSearchFilterInput!
  $pagination: Pagination!
) {
  searchDocumentsByAdvancedFilter(filter: $filter, pagination: $pagination) {
    total
    data {
      id uid qid status uploadedAt createdAt fileName completeness errorMessage
      reason { id rejectedReasons { message } exportedFailReasons { message } }
      queue { name group { name } }
      isMatched matchedURL
    }
    pageInfo { lastCursor hasNextPage hasPreviousPage }
  }
}
"""


async def build_filter_payload(state: SearchAgentState) -> dict:
    """modelId only — docType is an independent filter dimension (it targets the
    extractor's classification, not the org's model) and is not used here."""
    filter_dict = {"others": {"modelId": state["model_id"], "fields": []}}

    if state.get("queue_ids"):
        filter_dict["queue"] = state["queue_ids"]

    # Guard against the stale-field hazard: fields replace only when written, so
    # on a second query in the same thread that mentions no members, the member
    # subgraph is skipped and member_ids still holds the PREVIOUS query's ids.
    if state.get("extracted_members") and state.get("member_ids"):
        filter_dict["uploadedBy"] = state["member_ids"]

    if state.get("date_range"):
        filter_dict["uploadedAt"] = state["date_range"]

    return {
        "filter_payload": {
            "filter": filter_dict,
            "pagination": {"skip": 0, "take": 10},
        },
        "status": "searching",
    }


async def search_documents(state: SearchAgentState, config: RunnableConfig) -> dict:
    body = await gql(
        config["configurable"]["authorization"],
        SEARCH_QUERY,
        state["filter_payload"],
    )
    results = body["data"]["searchDocumentsByAdvancedFilter"]
    total = (results or {}).get("total", 0)
    return {
        "results": results,
        "status": "completed",
        "messages": [AIMessage(f"Found {total} document(s).")],
    }
