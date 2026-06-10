from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Generic, TypeVar, Any
from uuid import UUID
from datetime import datetime

T = TypeVar('T')

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)

# Lightweight identity extracted from a verified JWT — no database lookup required.
# Use this as the return type of get_current_user for all protected endpoints.
class CurrentUser(BaseModel):
    id: UUID
    username: str
    role: str

class ChatRequest(BaseModel):
    message: str
    conversationId: Optional[UUID] = None
    stream: bool = True
    maxResults: int = 5

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    message: Optional[str] = None
    data: Optional[T] = None

    @classmethod
    def success_response(cls, data: Any = None, message: Optional[str] = None):
        return cls(success=True, message=message, data=data)

    @classmethod
    def error_response(cls, message: str):
        return cls(success=False, message=message, data=None)

class AuthResponse(BaseModel):
    token: str
    tokenType: str = "Bearer"
    userId: UUID
    username: str
    email: str
    role: str

class DocumentResponse(BaseModel):
    id: UUID
    title: str
    fileName: str
    fileType: Optional[str] = None
    fileSize: Optional[int] = None
    language: str
    status: str
    chunkCount: int
    errorMessage: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True

class SourceReference(BaseModel):
    documentId: UUID
    documentTitle: str
    excerpt: str
    chunkIndex: int
    relevanceScore: float

class ChatResponse(BaseModel):
    conversationId: UUID
    messageId: UUID
    answer: str
    sources: List[SourceReference]
    followUp: bool

class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    sources: Any  # Can be list or parsed JSON string
    createdAt: datetime

    class Config:
        from_attributes = True

class ConversationResponse(BaseModel):
    id: UUID
    title: Optional[str] = None
    messages: Optional[List[MessageResponse]] = None
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True

class PageResponse(BaseModel, Generic[T]):
    content: List[T]
    page: int
    size: int
    totalElements: int
    totalPages: int
