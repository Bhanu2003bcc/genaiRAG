from uuid import UUID
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.database import get_db
from app.security import get_current_user
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    PageResponse,
    ApiResponse,
    CurrentUser
)
from app.services.chat_service import chat_service
from app.config import settings
from app.utils.router_cache import router_cache

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model=ApiResponse[ChatResponse])
async def chat(
    request: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    request.stream = False
    chat_resp = await chat_service.chat(request, current_user, db)
    
    if settings.router_cache_enabled:
        router_cache.invalidate_tag(f"user:{current_user.id}:conversations")
        
    return ApiResponse.success_response(data=chat_resp)

@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    # Returns an SSE Stream of Server-Sent Events
    generator = chat_service.stream_chat(request, current_user)
    
    if settings.router_cache_enabled:
        router_cache.invalidate_tag(f"user:{current_user.id}:conversations")
        
    return EventSourceResponse(generator)

@router.get("/conversations", response_model=ApiResponse[PageResponse[ConversationResponse]])
async def get_conversations(
    page: int = 0,
    size: int = 20,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    cache_key = f"user:{current_user.id}:conversations:page={page}:size={size}"
    if settings.router_cache_enabled:
        cached = router_cache.get(cache_key)
        if cached is not None:
            return cached

    paginated_convs = await chat_service.get_conversations(current_user.id, page, size, db)
    resp = ApiResponse.success_response(data=paginated_convs)
    
    if settings.router_cache_enabled:
        router_cache.set(cache_key, resp, tags=[f"user:{current_user.id}:conversations"])
        
    return resp

@router.get("/conversations/{conversation_id}", response_model=ApiResponse[ConversationResponse])
async def get_conversation(
    conversation_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    cache_key = f"user:{current_user.id}:conversation:{conversation_id}"
    if settings.router_cache_enabled:
        cached = router_cache.get(cache_key)
        if cached is not None:
            return cached

    conv_resp = await chat_service.get_conversation(conversation_id, current_user.id, db)
    resp = ApiResponse.success_response(data=conv_resp)
    
    if settings.router_cache_enabled:
        router_cache.set(cache_key, resp, tags=[f"user:{current_user.id}:conversations"])
        
    return resp

@router.delete("/conversations/{conversation_id}", response_model=ApiResponse[None])
async def delete_conversation(
    conversation_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    await chat_service.delete_conversation(conversation_id, current_user.id, db)
    
    if settings.router_cache_enabled:
        router_cache.invalidate_tag(f"user:{current_user.id}:conversations")
        
    return ApiResponse.success_response(message="Conversation deleted")
