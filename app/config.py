import os
import re
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Settings
    app_name: str = "rag-system"
    port: int = 8080
    api_prefix: str = "/api"

    # Database Settings
    database_url: str = "postgresql://postgres:postgres@localhost:5432/postgres"
    database_username: str = "postgres"
    database_password: str = "postgres"

    # Connection Pool Settings
    # Since we use Supabase's transaction pooler on port 6543, we can support 
    # much larger connection pools without running out of slots.
    # pool_size   — persistent connections kept alive per process worker
    # max_overflow — extra connections allowed above pool_size during bursts
    # pool_timeout — seconds to wait for a free connection before raising
    # pool_recycle — recycle connections after N seconds (prevents stale TCP)
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    # JWT Settings
    jwt_secret: str = "change_this_to_a_secure_random_secret_key_32chars"
    jwt_expiration_ms: int = 86400000  # 24 hours

    # Gemini API Settings
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_embedding_dimension: int = 768

    # Embedding Cache Settings
    # maxsize — maximum number of distinct texts held in the LRU cache
    # ttl     — seconds before a cached embedding is considered stale
    embedding_cache_maxsize: int = 1024
    embedding_cache_ttl: int = 3600  # 1 hour

    # Router Cache Settings
    router_cache_enabled: bool = True
    router_cache_ttl: int = 120  # 2 minutes

    # Retrieval Settings
    # rrf_k          — RRF constant (60 is the well-established default)
    # rrf_candidate_k — candidate pool multiplier; final results = maxResults,
    #                   candidate pool = maxResults * rrf_candidate_k
    rrf_k: int = 60
    rrf_candidate_multiplier: int = 6

    # LLM Re-Ranking Settings
    rerank_enabled: bool = True
    rerank_candidate_pool_size: int = 10

    # Chunking Settings
    semantic_chunking_enabled: bool = False

    # Concurrency Limits
    # bcrypt_max_concurrent — limits simultaneous bcrypt hash/verify calls to prevent
    #                         thread pool saturation (thundering herd at high load)
    bcrypt_max_concurrent: int = 4

    # doc_processing_max_concurrent — limits parallel background document embedding jobs
    #                                  to prevent Gemini API rate-limit bursts
    doc_processing_max_concurrent: int = 3

    # Upload Settings
    max_upload_size_mb: int = 50

    # Rate Limiting (slowapi, per-IP)
    # Format: "<count>/<period>" where period is second|minute|hour|day
    auth_rate_limit: str = "20/minute"

    # CORS Settings
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def sqlalchemy_database_url(self) -> str:
        """
        Converts a Spring JDBC URL (e.g. jdbc:postgresql://host:port/db) 
        to an asyncpg SQLAlchemy URL (postgresql+asyncpg://host:port/db).
        """
        url = self.database_url
        # Remove jdbc: prefix if present
        if url.startswith("jdbc:"):
            url = url[5:]
        
        # Replace driver prefix with postgresql+asyncpg
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif not url.startswith("postgresql+asyncpg://"):
            url = "postgresql+asyncpg://" + url.split("://")[-1]

        # Fix sslmode parameter for asyncpg
        url = url.replace("sslmode=require", "ssl=require")

        # Inject username and password if they are in fields but not in the URL
        # e.g., url could be postgresql://host:port/db and username/password are separate.
        # However, usually DATABASE_URL has credentials included. If not:
        if "://" in url:
            parts = url.split("://")
            prefix = parts[0]
            rest = parts[1]
            if "@" not in rest and self.database_username:
                creds = self.database_username
                if self.database_password:
                    creds += f":{self.database_password}"
                url = f"{prefix}://{creds}@{rest}"

        return url

# Load configurations automatically from .env or environment
settings = Settings()
