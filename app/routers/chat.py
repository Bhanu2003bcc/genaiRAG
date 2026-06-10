from uuid import UUID
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.database import get_db
from app.models import User
from app.security import get_current_user
from app.schemas import ChatRequest, ChatResponse, ConversationResponse, PageResponse, ApiResponse
from app.services.chat_service import chat_service

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model=ApiResponse[ChatResponse])
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    request.stream = False
    chat_resp = await chat_service.chat(request, current_user, db)
    return ApiResponse.success_response(data=chat_resp)

@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    # Returns an SSE Stream of Server-Sent Events
    generator = chat_service.stream_chat(request, current_user)
    return EventSourceResponse(generator)

@router.get("/conversations", response_model=ApiResponse[PageResponse[ConversationResponse]])
async def get_conversations(
    page: int = 0,
    size: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    paginated_convs = await chat_service.get_conversations(current_user.id, page, size, db)
    return ApiResponse.success_response(data=paginated_convs)

@router.get("/conversations/{conversation_id}", response_model=ApiResponse[ConversationResponse])
async def get_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    conv_resp = await chat_service.get_conversation(conversation_id, current_user.id, db)
    return ApiResponse.success_response(data=conv_resp)

@router.delete("/conversations/{conversation_id}", response_model=ApiResponse[None])
async def delete_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    await chat_service.delete_conversation(conversation_id, current_user.id, db)
    return ApiResponse.success_response(message="Conversation deleted")
