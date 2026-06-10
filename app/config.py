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

    # JWT Settings
    jwt_secret: str = "change_this_to_a_secure_random_secret_key_32chars"
    jwt_expiration_ms: int = 86400000  # 24 hours

    # Gemini API Settings
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_chat_model: str = "gemini-2.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_embedding_dimension: int = 768

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
