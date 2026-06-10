from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from app.config import settings

# ---------------------------------------------------------------------------
# Async engine — pool settings come from config (overridable via .env)
#
#   pool_size     : persistent connections kept alive per worker process.
#   max_overflow  : extra connections allowed during traffic bursts.
#   pool_timeout  : seconds to wait for a free slot before TimeoutError.
#   pool_recycle  : forcefully recycle connections every N seconds to
#                   prevent silent TCP-level drops from firewalls/NAT.
#   pool_pre_ping : issues a lightweight SELECT 1 before handing a
#                   connection to caller; detects stale connections early.
# ---------------------------------------------------------------------------
engine = create_async_engine(
    settings.sqlalchemy_database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_recycle=settings.db_pool_recycle,
    pool_pre_ping=True,
    future=True,
    echo=False,
    connect_args={"statement_cache_size": 0}
)

# Async session factory — shared across the entire application lifetime.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Declarative base for all ORM models.
Base = declarative_base()


# ---------------------------------------------------------------------------
# FastAPI dependency — yields a request-scoped async session.
# ---------------------------------------------------------------------------
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
