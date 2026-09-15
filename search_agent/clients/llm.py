from langchain_anthropic import ChatAnthropic

from search_agent.config import settings


def get_llm():
    """Cheap fast model — both extraction nodes are short structured-output
    calls, not reasoning. Temperature 0: extraction should be reproducible."""
    return ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=0,
        max_tokens=1024,
    )
