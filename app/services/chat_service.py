import logging
import json
import math
from uuid import UUID
from typing import List, Optional, AsyncGenerator
from fastapi import HTTPException, status
from sqlalchemy import select, func, desc, delete, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Message, User
from app.schemas import ChatRequest, ChatResponse, ConversationResponse, MessageResponse, SourceReference, PageResponse
from app.database import AsyncSessionLocal
from app.services.gemini_service import gemini_service

logger = logging.getLogger("com.rag.service.impl.ChatServiceImpl")

class ChatService:
    SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant with access to the user's documents.
Use the following context retrieved from the user's documents to answer the question.
If the answer cannot be found in the context, say so clearly.
Always cite which document the information comes from when possible.
Be concise, accurate, and helpful.

CONTEXT FROM DOCUMENTS:
{}"""

    async def get_or_create_conversation(self, conversation_id: Optional[UUID], user_id: UUID, db: AsyncSession) -> Conversation:
        if conversation_id:
            stmt = select(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
            result = await db.execute(stmt)
            conversation = result.scalars().first()
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Conversation not found with id: {str(conversation_id)}"
                )
            return conversation
        
        conversation = Conversation(user_id=user_id)
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        return conversation

    async def _build_history(self, conversation_id: UUID, db: AsyncSession) -> List[str]:
        # Load top 10 messages ordered by created_at desc (newest first)
        stmt = (
            select(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(desc(Message.created_at))
            .limit(10)
        )
        result = await db.execute(stmt)
        messages = list(result.scalars().all())
        # Reverse to get oldest first (ascending order)
        messages.reverse()
        return [m.content for m in messages]

    async def _hybrid_search(self, user_id: UUID, query_text: str, query_embedding: List[float], limit: int, db: AsyncSession) -> List[dict]:
        # Perform hybrid search query via native SQL execution
        query_str = """
            SELECT dc.id, dc.document_id, dc.chunk_index, dc.content, 
                   d.title AS doc_title
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.user_id = :user_id
            ORDER BY (
                0.7 * (1 - (dc.embedding <=> CAST(:embedding AS vector))) +
                0.3 * ts_rank(dc.content_tsv, plainto_tsquery('english', :query))
            ) DESC
            LIMIT :limit
        """
        
        # Format embedding list to PostgreSQL array-like vector string e.g. [0.1, 0.2, ...]
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        
        result = await db.execute(
            text(query_str),
            {
                "user_id": user_id,
                "embedding": embedding_str,
                "query": query_text,
                "limit": limit
            }
        )
        
        # Return list of dictionaries containing properties
        return [dict(row._mapping) for row in result.all()]

    def _build_context(self, chunks: List[dict]) -> str:
        if not chunks:
            return "No relevant documents found."
        
        formatted_chunks = []
        for c in chunks:
            formatted_chunks.append(f"[From: {c['doc_title']}]\n{c['content']}")
            
        return "\n\n---\n\n".join(formatted_chunks)

    def _build_sources(self, chunks: List[dict]) -> List[SourceReference]:
        sources = []
        for c in chunks:
            content = c["content"]
            excerpt = content[:200] + "..." if len(content) > 200 else content
            sources.append(SourceReference(
                documentId=c["document_id"],
                documentTitle=c["doc_title"],
                excerpt=excerpt,
                chunkIndex=c["chunk_index"],
                relevanceScore=0.85  # Fixed score from original Java implementation
            ))
        return sources

    async def chat(self, request: ChatRequest, user: User, db: AsyncSession) -> ChatResponse:
        # Get or create conversation
        conversation = await self.get_or_create_conversation(request.conversationId, user.id, db)

        # Generate query embedding
        query_embedding = await gemini_service.generate_embedding(request.message)

        # Retrieve relevant chunks via hybrid search
        chunks = await self._hybrid_search(user.id, request.message, query_embedding, request.maxResults, db)

        # Build prompt and history
        context = self._build_context(chunks)
        sources = self._build_sources(chunks)
        history = await self._build_history(conversation.id, db)
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(context)

        # Ask Gemini
        answer = await gemini_service.generate_answer(system_prompt, request.message, history)

        # Save messages to database
        user_msg = Message(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
            sources=[]
        )
        assistant_msg = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            sources=[s.model_dump(mode='json') for s in sources]
        )
        
        db.add(user_msg)
        db.add(assistant_msg)

        # Auto-title conversation
        if not conversation.title:
            title = request.message[:60] + "..." if len(request.message) > 60 else request.message
            conversation.title = title

        await db.commit()
        await db.refresh(assistant_msg)

        return ChatResponse(
            conversationId=conversation.id,
            messageId=assistant_msg.id,
            answer=answer,
            sources=sources,
            followUp=len(history) > 0
        )

    async def stream_chat(self, request: ChatRequest, user: User) -> AsyncGenerator[dict, None]:
        """
        Asynchronously streams the chat response over Server-Sent Events (SSE).
        Uses a separate session connection inside the generator to keep DB requests isolated.
        """
        # Step 1: Initialize metadata using isolated DB session
        async with AsyncSessionLocal() as db:
            conversation = await self.get_or_create_conversation(request.conversationId, user.id, db)
            conv_id = conversation.id
            
            # Save user message immediately
            user_msg = Message(
                conversation_id=conv_id,
                role="user",
                content=request.message,
                sources=[]
            )
            db.add(user_msg)
            await db.commit()
            
            # Retrieve search chunks & build history
            query_embedding = await gemini_service.generate_embedding(request.message)
            chunks = await self._hybrid_search(user.id, request.message, query_embedding, request.maxResults, db)
            history = await self._build_history(conv_id, db)

        # Step 2: Build context & sources
        context = self._build_context(chunks)
        sources = self._build_sources(chunks)
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(context)

        # Yield sources metadata first matching SseEmitter flow in Spring Boot
        sources_list = [s.model_dump(mode='json') for s in sources]
        yield {
            "event": "sources",
            "data": json.dumps(sources_list)
        }

        # Step 3: Stream content chunks
        full_answer = []
        async for chunk in gemini_service.stream_answer(system_prompt, request.message, history):
            full_answer.append(chunk)
            yield {
                "event": "chunk",
                "data": chunk
            }

        # Step 4: Finalize conversation & save response
        async with AsyncSessionLocal() as db:
            answer_text = "".join(full_answer)
            assistant_msg = Message(
                conversation_id=conv_id,
                role="assistant",
                content=answer_text,
                sources=sources_list
            )
            db.add(assistant_msg)
            
            # Load conversation to check title & update updated_at
            stmt = select(Conversation).filter(Conversation.id == conv_id)
            res = await db.execute(stmt)
            conv = res.scalars().first()
            if conv and not conv.title:
                title = request.message[:60] + "..." if len(request.message) > 60 else request.message
                conv.title = title
            
            await db.commit()
            await db.refresh(assistant_msg)
            assistant_id = assistant_msg.id

        # Yield done event indicating full stream completion
        yield {
            "event": "done",
            "data": json.dumps({
                "conversationId": str(conv_id),
                "messageId": str(assistant_id)
            })
        }

    async def get_conversations(self, user_id: UUID, page: int, size: int, db: AsyncSession) -> PageResponse[ConversationResponse]:
        # Count total elements
        count_stmt = select(func.count()).select_from(Conversation).filter(Conversation.user_id == user_id)
        count_result = await db.execute(count_stmt)
        total_elements = count_result.scalar() or 0

        # Query paginated conversations
        stmt = (
            select(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(desc(Conversation.updated_at))
            .offset(page * size)
            .limit(size)
        )
        result = await db.execute(stmt)
        conversations = result.scalars().all()

        content = []
        for c in conversations:
            content.append(ConversationResponse(
                id=c.id,
                title=c.title,
                createdAt=c.created_at,
                updatedAt=c.updated_at
            ))

        total_pages = math.ceil(total_elements / size) if size > 0 else 0

        return PageResponse(
            content=content,
            page=page,
            size=size,
            totalElements=total_elements,
            totalPages=total_pages
        )

    async def get_conversation(self, conversation_id: UUID, user_id: UUID, db: AsyncSession) -> ConversationResponse:
        # Get conversation
        stmt = select(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
        result = await db.execute(stmt)
        conv = result.scalars().first()
        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found with id: {str(conversation_id)}"
            )

        # Get messages in ascending order
        msg_stmt = select(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at.asc())
        msg_result = await db.execute(msg_stmt)
        messages = msg_result.scalars().all()

        message_responses = []
        for m in messages:
            # Parse sources from JSON array/field
            sources_data = m.sources if m.sources is not None else []
            message_responses.append(MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                sources=sources_data,
                createdAt=m.created_at
            ))

        return ConversationResponse(
            id=conv.id,
            title=conv.title,
            messages=message_responses,
            createdAt=conv.created_at,
            updatedAt=conv.updated_at
        )

    async def delete_conversation(self, conversation_id: UUID, user_id: UUID, db: AsyncSession) -> None:
        # Verify ownership
        stmt = select(Conversation).filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
        result = await db.execute(stmt)
        conv = result.scalars().first()
        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found with id: {str(conversation_id)}"
            )

        # Cascade deletes messages automatically due to ForeignKey cascade delete
        await db.delete(conv)
        await db.commit()

# Instantiate service singleton
chat_service = ChatService()
