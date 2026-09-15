from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres. Needs ?sslmode=require — the Cloud SQL instance is ENCRYPTED_ONLY.
    postgres_url: str

    # dash-dev, NOT api.staple.io (SPEC §16).
    staple_graphql_url: str = "https://dash-dev.staple.io/graphql"

    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key: str = ""

    fuzzy_threshold: float = 0.85

    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
