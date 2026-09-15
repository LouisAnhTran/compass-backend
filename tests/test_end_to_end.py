"""Full four-turn walkthrough with Staple and the LLM mocked.

Patching happens at each module's own binding (nodes.fetch.gql, not
clients.graphql.gql) because `from x import y` copies the reference — patching
the source module would leave the node still holding the original.

Everything else is the real code: ranking, interrupt payload construction,
the conditional edge, and filter assembly.
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from search_agent.graph.builder import build_graph
from search_agent.graph.nodes import extract as extract_mod
from search_agent.graph.nodes import fetch as fetch_mod
from search_agent.graph.nodes import search as search_mod

MODELS = [{"id": 20, "name": "Receipt"}, {"id": 21, "name": "Invoice"}]
QUEUES = [{"id": 2321, "name": "Q1 - Invoices EU"}, {"id": 2313, "name": "Q2 - Invoicing EU"}]
MEMBERS = [{"id": 862, "firstName": "John", "lastName": "Smith", "email": "j@x.io"}]


class FakeLLM:
    """Stands in for ChatAnthropic().with_structured_output(Model)."""

    def __init__(self, result):
        self._result = result

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, _messages):
        return self._result


@pytest.fixture
def mocked(monkeypatch):
    captured = {}

    async def fake_gql(auth, query, variables=None):
        if "getModels" in query:
            return {"data": {"models": MODELS}}
        if "getAssignedQueues" in query:
            return {"data": {"queues": QUEUES}}
        if "getListMembersWithoutPagination" in query:
            return {"data": {"getListMembersWithoutPagination": MEMBERS}}
        if "searchDocumentsByAdvancedFilter" in query:
            captured["variables"] = variables
            return {
                "data": {
                    "searchDocumentsByAdvancedFilter": {
                        "total": 3, "data": [], "pageInfo": {}
                    }
                }
            }
        raise AssertionError(f"unexpected query: {query[:60]}")

    monkeypatch.setattr(fetch_mod, "gql", fake_gql)
    monkeypatch.setattr(search_mod, "gql", fake_gql)
    return captured


def set_llm(monkeypatch, models, other):
    calls = iter([models, other])
    monkeypatch.setattr(extract_mod, "get_llm", lambda: FakeLLM(next(calls)))


def cfg(thread_id):
    return {"configurable": {"thread_id": thread_id, "authorization": "Bearer fake"}}


async def test_four_turn_flow(mocked, monkeypatch):
    set_llm(
        monkeypatch,
        extract_mod.ModelExtraction(extracted_models=["receipt"]),
        extract_mod.OtherExtraction(
            extracted_queues=["Q1", "invoicing"],
            extracted_members=["John"],
            date_range={"from": "01/09/2026", "to": "14/09/2026"},
        ),
    )
    g = build_graph(MemorySaver())
    c = cfg("t1")

    # ── Turn 1: stop at model confirmation ───────────────────────────────────
    await g.ainvoke({"query": "find receipts in Q1 or invoicing from John"}, c)
    payload = (await g.aget_state(c)).interrupts[0].value

    assert payload["reason"] == "confirm_model"
    assert len(payload["items"]) == 1, "single-model rule"
    assert payload["items"][0]["sub_reason"] == "found_matches"
    assert payload["items"][0]["matches"][0]["display"] == "Receipt"

    # ── Turn 2: resume with model, stop at queues ────────────────────────────
    await g.ainvoke(Command(resume={"selection": {"id": 20, "name": "Receipt"}}), c)
    payload = (await g.aget_state(c)).interrupts[0].value

    assert payload["reason"] == "confirm_queues"
    assert len(payload["items"]) == 2, "one group per extracted queue string"
    # Queue display carries the id — names collide across groups.
    assert payload["all_options"][0]["display"] == "2321 · Q1 - Invoices EU"

    # ── Turn 3: resume with queues → member subgraph ─────────────────────────
    await g.ainvoke(
        Command(resume={"selections": [{"id": 2321, "name": "Q1 - Invoices EU"}]}), c
    )
    payload = (await g.aget_state(c)).interrupts[0].value

    assert payload["reason"] == "confirm_members"
    assert payload["all_options"][0]["display"] == "John Smith"

    # ── Turn 4: resume with members, run to END ──────────────────────────────
    await g.ainvoke(Command(resume={"selections": [{"id": 862, "name": "John Smith"}]}), c)
    snap = await g.aget_state(c)

    assert snap.next == (), "graph should be finished"
    assert snap.values["status"] == "completed"

    built = mocked["variables"]["filter"]
    assert built["others"] == {"modelId": 20, "fields": []}, "modelId, never docType"
    assert built["queue"] == [2321]
    assert built["uploadedBy"] == [862]
    assert built["uploadedAt"] == {"from": "01/09/2026", "to": "14/09/2026"}


async def test_no_members_skips_subgraph(mocked, monkeypatch):
    """Empty member extraction bypasses fetch/rank/confirm — the only branch."""
    set_llm(
        monkeypatch,
        extract_mod.ModelExtraction(extracted_models=["receipt"]),
        extract_mod.OtherExtraction(
            extracted_queues=["Q1"], extracted_members=[], date_range=None
        ),
    )
    g = build_graph(MemorySaver())
    c = cfg("t2")

    await g.ainvoke({"query": "find receipts in Q1"}, c)
    await g.ainvoke(Command(resume={"selection": {"id": 20, "name": "Receipt"}}), c)
    await g.ainvoke(Command(resume={"selections": [{"id": 2321, "name": "Q1"}]}), c)

    snap = await g.aget_state(c)
    assert snap.next == (), "should finish without a members interrupt"
    # Key absent entirely, not an empty list.
    assert "uploadedBy" not in mocked["variables"]["filter"]


async def test_numeric_queue_id_scores_exact(mocked, monkeypatch):
    """A bare id is an exact reference — string similarity would score it ~0."""
    set_llm(
        monkeypatch,
        extract_mod.ModelExtraction(extracted_models=["receipt"]),
        extract_mod.OtherExtraction(
            extracted_queues=["2321"], extracted_members=[], date_range=None
        ),
    )
    g = build_graph(MemorySaver())
    c = cfg("t3")

    await g.ainvoke({"query": "find receipts in queue 2321"}, c)
    await g.ainvoke(Command(resume={"selection": {"id": 20, "name": "Receipt"}}), c)

    payload = (await g.aget_state(c)).interrupts[0].value
    match = payload["items"][0]["matches"][0]
    assert match["id"] == 2321 and match["score"] == 1.0
