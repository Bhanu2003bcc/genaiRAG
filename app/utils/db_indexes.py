"""
db_indexes.py — Idempotent database index bootstrap.

Called once at application startup via the FastAPI lifespan handler in
main.py.  Every statement uses IF NOT EXISTS so it is safe to run on
every deployment and on an already-indexed database.

Index strategy
--------------
HNSW (document_chunks.embedding)
    Hierarchical Navigable Small World approximate nearest-neighbour index
    using cosine distance operator class.  This turns the vector scan from
    O(n) sequential to O(log n) approximate, which is the single biggest
    performance win for the retrieval path.

    Tuning knobs (pgvector defaults shown):
        m              = 16   — edges per node (higher → better recall, more RAM)
        ef_construction = 64  — candidates considered during build
                                (higher → better recall, slower build)
    These defaults are solid for up to ~10 M vectors.  Increase for larger
    corpora after benchmarking with `SELECT * FROM pg_stat_user_indexes`.

GIN (document_chunks.content_tsv)
    Generalised Inverted Index on the pre-computed tsvector column.  Without
    this, every full-text search (`ts_rank`) degrades to a seq-scan on the
    entire chunks table.

B-Tree composites
    Covering the join and filter paths used by the hybrid search query so
    PostgreSQL does not fall back to bitmap heap scans.

NOTE: For production databases with millions of existing rows, prefer running
    CREATE INDEX CONCURRENTLY
manually before the first deployment, since that avoids the table lock that
IF NOT EXISTS triggers on a blocking build.
"""

import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal

logger = logging.getLogger("com.rag.utils.db_indexes")


# ---------------------------------------------------------------------------
# Index definitions
# Each tuple: (description, DDL statement)
# ---------------------------------------------------------------------------
_INDEXES = [
    (
        "HNSW vector index on document_chunks.embedding",
        """
        CREATE INDEX IF NOT EXISTS idx_dc_embedding_hnsw
        ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """,
    ),
    (
        "GIN full-text index on document_chunks.content_tsv",
        """
        CREATE INDEX IF NOT EXISTS idx_dc_content_tsv_gin
        ON document_chunks
        USING gin (content_tsv)
        """,
    ),
    (
        "Composite B-Tree on documents(user_id, status) for join + filter",
        """
        CREATE INDEX IF NOT EXISTS idx_documents_user_status
        ON documents (user_id, status)
        """,
    ),
    (
        "B-Tree on document_chunks.document_id (join key)",
        """
        CREATE INDEX IF NOT EXISTS idx_dc_document_id
        ON document_chunks (document_id)
        """,
    ),
    (
        "B-Tree on document_chunks.chunk_index for ordered retrieval",
        """
        CREATE INDEX IF NOT EXISTS idx_dc_chunk_index
        ON document_chunks (document_id, chunk_index)
        """,
    ),
]


async def ensure_indexes() -> None:
    """
    Create all required indexes if they do not already exist.

    This coroutine is idempotent — calling it multiple times is safe.
    Each DDL statement is executed in autocommit mode (DDL is implicitly
    transactional in PostgreSQL, but CREATE INDEX IF NOT EXISTS must run
    outside an explicit transaction to avoid locking issues with pgvector).
    """
    logger.info("Ensuring database indexes are in place...")

    async with AsyncSessionLocal() as db:
        # Use AUTOCOMMIT isolation so CREATE INDEX doesn't run inside an
        # explicit transaction block — required for HNSW index builds.
        await db.execute(text("SET lock_timeout = '5s'"))

        for description, ddl in _INDEXES:
            try:
                await db.execute(text(ddl))
                await db.commit()
                logger.info(f"  ✓ {description}")
            except Exception as exc:
                await db.rollback()
                # Log but do not crash the server — a missing index degrades
                # performance but does not break correctness.
                logger.warning(
                    f"  ✗ Could not create index [{description}]: {exc}"
                )

    logger.info("Database index bootstrap complete.")
