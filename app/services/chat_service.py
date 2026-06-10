
import json
import logging
import math
from typing import AsyncGenerator, List, Optional
from uuid import UUID

# pyrefly: ignore [missing-import]
from fastapi import HTTPException, status
# pyrefly: ignore [missing-import]
from sqlalchemy import desc, func, select, text
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models import (
    Conversation,
    Message,
    MessageRole
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    MessageResponse,
    PageResponse,
    SourceReference,
    CurrentUser
)
from app.services.gemini_service import gemini_service
from app.utils.metrics import metrics_collector
import time

logger = logging.getLogger("app.chat")


class ChatService:

    MAX_HISTORY_MESSAGES = 12
    MAX_CONTEXT_CHUNKS = 8
    MIN_RELEVANCE_SCORE = 0.45

    SYSTEM_PROMPT_TEMPLATE = """
You are a grounded AI assistant.

Rules:
- Answer ONLY using the provided document context.
- If answer is not found, explicitly say so.
- Do not hallucinate.
- Cite document sources naturally.
- Keep responses concise and factual.

DOCUMENT CONTEXT:
{context}
"""

    # =========================
    # CONVERSATION
    # =========================

    async def get_or_create_conversation(
        self,
        conversation_id: Optional[UUID],
        user_id: UUID,
        db: AsyncSession
    ) -> Conversation:

        if conversation_id:

            result = await db.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id
                )
            )

            conversation = result.scalars().first()

            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )

            return conversation

        conversation = Conversation(
            user_id=user_id
        )

        db.add(conversation)

        await db.commit()
        await db.refresh(conversation)

        return conversation

    # =========================
    # HISTORY
    # =========================

    async def _build_history(
        self,
        conversation_id: UUID,
        db: AsyncSession
    ) -> List[dict]:

        result = await db.execute(
            select(Message)
            .where(
                Message.conversation_id
                == conversation_id
            )
            .order_by(desc(Message.created_at))
            .limit(self.MAX_HISTORY_MESSAGES)
        )

        messages = list(
            result.scalars().all()
        )

        messages.reverse()

        return [
            {
                "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                "content": msg.content
            }
            for msg in messages
        ]

    # =========================
    # HYBRID SEARCH
    # =========================

    async def _hybrid_search(
        self,
        user_id: UUID,
        query_text: str,
        query_embedding: List[float],
        limit: int,
        db: AsyncSession
    ) -> List[dict]:
        """
        Reciprocal Rank Fusion (RRF) hybrid search.

        Two independent queries are executed:
          1. Vector similarity search via the HNSW index (cosine distance).
          2. Full-text keyword search via the GIN index (ts_rank).

        Their ranked lists are fused on the Python side with:
            rrf_score(d) = 1/(k + vector_rank) + 1/(k + fts_rank)
        where k=60 is the standard constant (Cormack et al., 2009).

        Advantages over the previous weighted sum:
          - Scale-agnostic: cosine distance [0,1] and ts_rank (unbounded)
            no longer fight each other.
          - Configurable via settings (rrf_k, rrf_candidate_multiplier).
          - Adds d.status = 'COMPLETED' guard so in-progress/failed
            documents are never surfaced to the user.
        """
        from app.config import settings as _s

        candidate_k = min(limit, self.MAX_CONTEXT_CHUNKS) * _s.rrf_candidate_multiplier
        k = _s.rrf_k
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        # ----------------------------------------------------------------
        # Query 1 — HNSW vector similarity
        # ----------------------------------------------------------------
        vector_result = await db.execute(
            text("""
                SELECT
                    dc.id::text          AS id,
                    dc.document_id::text AS document_id,
                    dc.chunk_index,
                    dc.content,
                    d.title              AS doc_title,
                    ROW_NUMBER() OVER (
                        ORDER BY dc.embedding <=> CAST(:embedding AS vector)
                    )                    AS vector_rank
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.user_id = :user_id
                  AND d.status  = 'COMPLETED'
                ORDER BY dc.embedding <=> CAST(:embedding AS vector)
                LIMIT :candidate_k
            """),
            {
                "user_id": str(user_id),
                "embedding": embedding_str,
                "candidate_k": candidate_k,
            },
        )
        vector_rows = {
            row.id: dict(row._mapping)
            for row in vector_result.all()
        }

        # ----------------------------------------------------------------
        # Query 2 — GIN full-text search (skipped for empty queries)
        # ----------------------------------------------------------------
        fts_ranks: dict = {}
        if query_text and query_text.strip():
            fts_result = await db.execute(
                text("""
                    SELECT
                        dc.id::text AS id,
                        ROW_NUMBER() OVER (
                            ORDER BY ts_rank(
                                dc.content_tsv,
                                plainto_tsquery('english', :query)
                            ) DESC
                        ) AS fts_rank
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.user_id = :user_id
                      AND d.status  = 'COMPLETED'
                      AND dc.content_tsv @@ plainto_tsquery('english', :query)
                    ORDER BY fts_rank
                    LIMIT :candidate_k
                """),
                {
                    "user_id": str(user_id),
                    "query": query_text,
                    "candidate_k": candidate_k,
                },
            )
            fts_ranks = {
                row.id: int(row.fts_rank)
                for row in fts_result.all()
            }

        # ----------------------------------------------------------------
        # Python-side RRF fusion
        # ----------------------------------------------------------------
        all_ids = set(vector_rows.keys()) | set(fts_ranks.keys())

        scored: List[tuple] = []
        for chunk_id in all_ids:
            v_rank = int(vector_rows[chunk_id]["vector_rank"]) if chunk_id in vector_rows else candidate_k + 1
            f_rank = fts_ranks.get(chunk_id, candidate_k + 1)
            rrf_score = 1.0 / (k + v_rank) + 1.0 / (k + f_rank)
            scored.append((rrf_score, chunk_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Decide how many candidate chunks to fetch
        fetch_limit = min(limit, self.MAX_CONTEXT_CHUNKS)
        if _s.rerank_enabled:
            fetch_limit = max(fetch_limit, _s.rerank_candidate_pool_size)
            
        top = scored[:fetch_limit]

        # Normalise to [0, 1] relative to the best result.
        max_score = top[0][0] if top else 1.0

        results = []
        for rrf_score, chunk_id in top:
            row = vector_rows.get(chunk_id)
            if row is None:
                # Chunk appeared only in FTS — fetch its full content.
                fetch = await db.execute(
                    text("""
                        SELECT dc.id::text AS id, dc.document_id::text AS document_id,
                               dc.chunk_index, dc.content, d.title AS doc_title
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE dc.id = :chunk_id
                    """),
                    {"chunk_id": chunk_id},
                )
                fetched = fetch.first()
                if not fetched:
                    continue
                row = dict(fetched._mapping)

            normalised = round(rrf_score / max_score, 4)
            # Add to results first (without gating yet, so LLM re-ranker can grade all candidates)
            results.append({**row, "relevance_score": normalised})

        # Close database session before making the slow re-ranking API call
        await db.close()

        # Apply LLM Re-ranking if enabled and we have results
        if _s.rerank_enabled and results:
            try:
                # Call gemini_service to re-rank chunks
                scores_map = await gemini_service.re_rank_chunks(query_text, results)
                if scores_map:
                    # Update relevance_score using LLM scores
                    for r in results:
                        cid = r["id"]
                        if cid in scores_map:
                            r["relevance_score"] = scores_map[cid]
                    # Sort results by the new LLM relevance scores
                    results.sort(key=lambda x: x["relevance_score"], reverse=True)
            except Exception as e:
                logger.warning(f"Reranking integration failed, falling back to RRF: {str(e)}")

        # Finally, filter by MIN_RELEVANCE_SCORE and limit to the requested amount
        filtered_results = [
            r for r in results
            if r["relevance_score"] >= self.MIN_RELEVANCE_SCORE
        ]
        return filtered_results[:limit]

    # =========================
    # CONTEXT
    # =========================

    def _build_context(
        self,
        chunks: List[dict]
    ) -> str:

        if not chunks:
            return "No relevant document context."

        context_parts = []

        for chunk in chunks:

            sanitized = (
                chunk["content"]
                .replace("{", "")
                .replace("}", "")
            )

            context_parts.append(
                f"[Document: {chunk['doc_title']}]\n"
                f"{sanitized}"
            )

        return "\n\n---\n\n".join(
            context_parts
        )

    # =========================
    # SOURCES
    # =========================

    def _build_sources(
        self,
        chunks: List[dict]
    ) -> List[SourceReference]:

        sources = []

        for chunk in chunks:

            excerpt = (
                chunk["content"][:200]
                + "..."
            )

            sources.append(
                SourceReference(
                    documentId=chunk["document_id"],
                    documentTitle=chunk["doc_title"],
                    excerpt=excerpt,
                    chunkIndex=chunk["chunk_index"],
                    relevanceScore=round(
                        float(chunk["relevance_score"]),
                        4
                    )
                )
            )

        return sources

    # =========================
    # CHAT
    # =========================

    async def chat(
        self,
        request: ChatRequest,
        user: CurrentUser,
        db: AsyncSession
    ) -> ChatResponse:
        # Close request-scoped DB session immediately to release connection to the pool
        await db.close()

        # 1. Generate embedding (no DB connection)
        query_embedding = await gemini_service.generate_embedding(request.message)
        if not query_embedding:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embedding service unavailable"
            )

        # 2. Acquire a temp database session to query conversation, history, and chunks
        async with AsyncSessionLocal() as temp_db:
            conversation = await self.get_or_create_conversation(
                request.conversationId,
                user.id,
                temp_db
            )
            
            history = await self._build_history(
                conversation.id,
                temp_db
            )
            
            # _hybrid_search will query DB and close temp_db before reranking!
            chunks = await self._hybrid_search(
                user.id,
                request.message,
                query_embedding,
                request.maxResults,
                temp_db
            )
            
            conv_id = conversation.id

        context = self._build_context(chunks)
        sources = self._build_sources(chunks)

        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(context=context)

        # 3. Slow LLM Answer Generation (no DB connection held!)
        start_time = time.perf_counter()
        answer = await gemini_service.generate_answer(
            system_prompt=system_prompt,
            user_message=request.message,
            history=history
        )
        metrics_collector.record_llm_generation(time.perf_counter() - start_time)

        # 4. Open a write session to save messages and update conversation title
        async with AsyncSessionLocal() as write_db:
            # Re-fetch conversation to avoid detached object issues
            result = await write_db.execute(
                select(Conversation).where(Conversation.id == conv_id)
            )
            conv = result.scalars().first()
            
            user_message = Message(
                conversation_id=conv_id,
                role=MessageRole.USER.value,
                content=request.message,
                sources=[]
            )
            
            assistant_message = Message(
                conversation_id=conv_id,
                role=MessageRole.ASSISTANT.value,
                content=answer,
                sources=[
                    s.model_dump(mode="json")
                    for s in sources
                ]
            )
            
            write_db.add(user_message)
            write_db.add(assistant_message)
            
            if conv and not conv.title:
                conv.title = request.message[:60]
                
            await write_db.commit()
            await write_db.refresh(assistant_message)
            assistant_msg_id = assistant_message.id

        return ChatResponse(
            conversationId=conv_id,
            messageId=assistant_msg_id,
            answer=answer,
            sources=sources,
            followUp=len(history) > 0
        )

    # =========================
    # STREAM CHAT
    # =========================

    async def stream_chat(
        self,
        request: ChatRequest,
        user: CurrentUser
    ) -> AsyncGenerator[dict, None]:
        # 1. Generate embedding first (no DB connection)
        query_embedding = await gemini_service.generate_embedding(request.message)
        if not query_embedding:
            yield {
                "event": "error",
                "data": "Embedding generation failed"
            }
            return

        # 2. Open temp database session to check conversation, history, chunks and save user message
        async with AsyncSessionLocal() as temp_db:
            conversation = await self.get_or_create_conversation(
                request.conversationId,
                user.id,
                temp_db
            )
            
            history = await self._build_history(
                conversation.id,
                temp_db
            )
            
            user_message = Message(
                conversation_id=conversation.id,
                role=MessageRole.USER.value,
                content=request.message,
                sources=[]
            )
            temp_db.add(user_message)
            await temp_db.commit()
            
            # _hybrid_search will query DB and close temp_db before reranking!
            chunks = await self._hybrid_search(
                user.id,
                request.message,
                query_embedding,
                request.maxResults,
                temp_db
            )
            
            conv_id = conversation.id

        context = self._build_context(chunks)
        sources = self._build_sources(chunks)
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(context=context)

        yield {
            "event": "sources",
            "data": json.dumps([
                s.model_dump(mode="json")
                for s in sources
            ])
        }

        full_response = []
        try:
            start_time = time.perf_counter()
            async for token in gemini_service.stream_answer(
                system_prompt=system_prompt,
                user_message=request.message,
                history=history
            ):
                full_response.append(token)
                yield {
                    "event": "chunk",
                    "data": token
                }
            metrics_collector.record_llm_generation(time.perf_counter() - start_time)
        except Exception as e:
            logger.exception(f"Streaming failed: {str(e)}")
            yield {
                "event": "error",
                "data": "Streaming failed"
            }
            return

        answer = "".join(full_response)

        # 3. Save assistant message and update conversation title inside a write transaction
        async with AsyncSessionLocal() as write_db:
            result = await write_db.execute(
                select(Conversation).where(Conversation.id == conv_id)
            )
            conv = result.scalars().first()
            
            assistant_message = Message(
                conversation_id=conv_id,
                role=MessageRole.ASSISTANT.value,
                content=answer,
                sources=[
                    s.model_dump(mode="json")
                    for s in sources
                ]
            )
            write_db.add(assistant_message)
            
            if conv and not conv.title:
                conv.title = request.message[:60]
                
            await write_db.commit()
            await write_db.refresh(assistant_message)
            assistant_msg_id = assistant_message.id

            yield {
                "event": "done",
                "data": json.dumps({
                    "conversationId": str(conv_id),
                    "messageId": str(assistant_msg_id)
                })
            }


    # =========================
    # CONVERSATION CRUD
    # =========================

    async def get_conversations(
        self,
        user_id: UUID,
        page: int,
        size: int,
        db: AsyncSession
    ) -> PageResponse:
        """
        Return a paginated list of conversations for a user,
        ordered most-recent first, without loading messages.
        """
        offset = page * size

        # Total count
        count_result = await db.execute(
            select(func.count()).select_from(Conversation)
            .where(Conversation.user_id == user_id)
        )
        total = count_result.scalar_one()

        # Page of conversations
        rows_result = await db.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.updated_at))
            .offset(offset)
            .limit(size)
        )
        convs = list(rows_result.scalars().all())

        conv_responses = [
            ConversationResponse(
                id=c.id,
                title=c.title,
                messages=None,
                createdAt=c.created_at,
                updatedAt=c.updated_at
            )
            for c in convs
        ]

        return PageResponse(
            content=conv_responses,
            page=page,
            size=size,
            totalElements=total,
            totalPages=max(1, math.ceil(total / size)) if size else 1
        )

    async def get_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID,
        db: AsyncSession
    ) -> ConversationResponse:
        """
        Return a single conversation with all its messages.
        """
        result = await db.execute(
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
        )
        conv = result.scalars().first()

        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        # Load messages
        msgs_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        messages = list(msgs_result.scalars().all())

        msg_responses = [
            MessageResponse(
                id=m.id,
                role=m.role if isinstance(m.role, str) else m.role.value,
                content=m.content,
                sources=m.sources,
                createdAt=m.created_at
            )
            for m in messages
        ]

        return ConversationResponse(
            id=conv.id,
            title=conv.title,
            messages=msg_responses,
            createdAt=conv.created_at,
            updatedAt=conv.updated_at
        )

    async def delete_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID,
        db: AsyncSession
    ) -> None:
        """
        Delete a conversation (and its messages via CASCADE) after
        verifying the requesting user owns it.
        """
        result = await db.execute(
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
        )
        conv = result.scalars().first()

        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        await db.delete(conv)
        await db.commit()


chat_service = ChatService()

