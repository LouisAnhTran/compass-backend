"""FastAPI app + lifespan.

Table creation happens here for LangGraph's checkpoint tables ONLY, via
setup(). conversation_history is applied out-of-band by `make migrate` —
see persistence/checkpointer.py for the reasoning.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from search_agent.api.routes import router
from search_agent.config import settings
from search_agent.graph.builder import build_graph
from search_agent.persistence.checkpointer import setup_checkpointer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # from_conn_string() returns a CONTEXT MANAGER, not a saver — binding it
    # directly (as the original spec draft did) yields an unusable object.
    async with AsyncPostgresSaver.from_conn_string(settings.postgres_url) as checkpointer:
        await setup_checkpointer(checkpointer)
        app.state.graph = build_graph(checkpointer)
        yield


app = FastAPI(title="Compass", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
