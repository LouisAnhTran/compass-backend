"""LangSmith tracing.

LangChain and LangGraph trace automatically when LANGSMITH_TRACING=true and the
credentials are present in os.environ — there is nothing to wire into the graph.
What this module does is make a misconfiguration loud, because the failure mode
is silent: runs simply never appear in the UI and the app looks fine.

Two ways that happens:

  1. LANGSMITH_WORKSPACE_ID missing. The key we use is ORG-scoped (lsv2_sk_).
     Org endpoints authenticate, but every workspace-scoped call — including
     trace ingestion — returns 403. The SDK converts this variable into the
     X-Tenant-Id header that fixes it.

  2. The variables never reach os.environ. pydantic-settings reads .env into
     the `settings` object, but the LangSmith SDK reads the process
     environment, so loading the file is not enough on its own — hence
     `uv run --env-file .env` locally and `env_file:` in compose.
"""

import logging
import os

logger = logging.getLogger(__name__)

REQUIRED = ("LANGSMITH_API_KEY", "LANGSMITH_WORKSPACE_ID")


def configure_tracing() -> bool:
    """Log what tracing will actually do. Returns True if it is live."""
    if os.getenv("LANGSMITH_TRACING", "").lower() not in ("true", "1"):
        logger.info("langsmith: disabled (LANGSMITH_TRACING not true)")
        return False

    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        # Deliberately a warning, not an exception: losing traces should not
        # take down the service.
        logger.warning(
            "langsmith: TRACING IS ON BUT WILL FAIL — missing %s. "
            "An org-scoped key without LANGSMITH_WORKSPACE_ID 403s silently.",
            ", ".join(missing),
        )
        return False

    key = os.environ["LANGSMITH_API_KEY"]
    if key.startswith("lsv2_sk_") and not os.getenv("LANGSMITH_WORKSPACE_ID"):
        logger.warning("langsmith: org-scoped key with no workspace id — traces will 403")
        return False

    logger.info(
        "langsmith: tracing → project=%s workspace=%s endpoint=%s",
        os.getenv("LANGSMITH_PROJECT", "default"),
        os.environ["LANGSMITH_WORKSPACE_ID"],
        os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    )
    return True
