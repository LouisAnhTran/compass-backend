"""The two LLM extraction nodes (SPEC §7)."""

from datetime import date

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from search_agent.clients.llm import get_llm
from search_agent.graph.prompts import EXTRACT_MODEL, EXTRACT_OTHER
from search_agent.graph.state import SearchAgentState


class ModelExtraction(BaseModel):
    extracted_models: list[str] = Field(
        default_factory=list,
        description="Document type/model mentions in the user's own phrasing. Empty if none.",
    )


class DateRangeOut(BaseModel):
    from_: str = Field(alias="from", description="dd/mm/yyyy, inclusive")
    to: str = Field(description="dd/mm/yyyy, inclusive")

    model_config = {"populate_by_name": True}


class OtherExtraction(BaseModel):
    extracted_queues: list[str] = Field(default_factory=list)
    extracted_members: list[str] = Field(default_factory=list)
    date_range: DateRangeOut | None = None


async def extract_model(state: SearchAgentState) -> dict:
    llm = get_llm().with_structured_output(ModelExtraction)
    out = await llm.ainvoke([HumanMessage(EXTRACT_MODEL.format(query=state["query"]))])
    return {
        "extracted_models": out.extracted_models[:1],  # single-model rule
        "messages": [HumanMessage(state["query"])],
    }


async def extract_other_entities(state: SearchAgentState) -> dict:
    llm = get_llm().with_structured_output(OtherExtraction)
    prompt = EXTRACT_OTHER.format(
        query=state["query"],
        # The model has no reliable sense of "now", so relative dates like
        # "last week" are unresolvable without this.
        today=date.today().strftime("%d/%m/%Y"),
    )
    out = await llm.ainvoke([HumanMessage(prompt)])

    date_range = None
    if out.date_range:
        # Serialised by alias so the key is "from", matching
        # AdvancedSearchDateRangeInput — "from" is a Python keyword, hence from_.
        date_range = {"from": out.date_range.from_, "to": out.date_range.to}

    return {
        "extracted_queues": out.extracted_queues,
        "extracted_members": out.extracted_members,
        "date_range": date_range,
    }
