from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str = (  # pragma: allowlist secret
        "postgresql+asyncpg://postgres:postgres@localhost:5432/projectdb"
    )
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    jwt_secret: str = "replace-in-production"  # noqa: S105
    jwt_expire_minutes: int = 60
    litellm_query_model: str = "gpt-4o-mini"
    litellm_context_model: str = "gpt-4o-mini"
    default_chunking_strategy: str = "naive"
    default_chunk_size: int = 512
    default_chunk_overlap: int = 64
    default_cag_rebuild_threshold: int = 50
    max_upload_size_mb: int = 50
    arq_max_jobs: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
