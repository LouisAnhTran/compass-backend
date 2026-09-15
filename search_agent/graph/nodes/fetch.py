"""Lookup nodes — the three GraphQL calls that populate the candidate lists.

Each reads the caller's bearer token from config, never from state: state is
checkpointed to Postgres, so a token in state would outlive its session on disk.
"""

from langchain_core.runnables import RunnableConfig

from search_agent.clients.graphql import gql
from search_agent.graph.state import SearchAgentState

MODELS_QUERY = """
query getModelsForAnalyticsDashboard {
  models: getModels { id name }
}
"""

QUEUES_QUERY = """
query getSelectableQueues {
  queues: getAssignedQueues { id name }
}
"""

MEMBERS_QUERY = """
query getTeamMembers {
  getListMembersWithoutPagination { id firstName lastName email role roleColor }
}
"""


def _auth(config: RunnableConfig) -> str:
    return config["configurable"]["authorization"]


async def fetch_models(state: SearchAgentState, config: RunnableConfig) -> dict:
    body = await gql(_auth(config), MODELS_QUERY)
    # getModels is [Model] — nullable list of nullable entries.
    return {"model_candidates": [m for m in (body["data"]["models"] or []) if m]}


async def fetch_queues(state: SearchAgentState, config: RunnableConfig) -> dict:
    body = await gql(_auth(config), QUEUES_QUERY)
    return {"queue_candidates": [q for q in (body["data"]["queues"] or []) if q]}


async def fetch_members(state: SearchAgentState, config: RunnableConfig) -> dict:
    body = await gql(_auth(config), MEMBERS_QUERY)
    members = body["data"]["getListMembersWithoutPagination"] or []
    return {"member_candidates": [m for m in members if m]}
