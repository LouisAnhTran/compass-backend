ENV := ../local/.env

run:
	uv run --env-file $(ENV) uvicorn search_agent.api.main:app --reload --port 8000

# Applies schema.sql (conversation_history) ONCE, on purpose.
# LangGraph's checkpoint tables are NOT created here — setup() does those at
# app startup. See search_agent/persistence/checkpointer.py.
migrate:
	uv run --env-file $(ENV) python -c "import asyncio, os; \
	from search_agent.persistence.checkpointer import apply_schema; \
	asyncio.run(apply_schema(os.environ['POSTGRES_URL']))"

test:
	uv run pytest -q

.PHONY: run migrate test
