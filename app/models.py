
import uuid
from datetime import datetime
from enum import Enum

# pyrefly: ignore [missing-import]
from sqlalchemy import (
    Column,
    String,
    Boolean,
    Integer,
    BigInteger,
    ForeignKey,
    Text,
    DateTime,
    func,
    Index,
    Enum as SqlEnum
)
# pyrefly: ignore [missing-import]
from sqlalchemy.dialects.postgresql import UUID, JSONB
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import relationship, declarative_base
# pyrefly: ignore [missing-import]
from pgvector.sqlalchemy import Vector

Base = declarative_base()


# =========================
# ENUMS
# =========================

class UserRole(str, Enum):
    USER = "USER"
    ADMIN = "ADMIN"


class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# =========================
# MIXINS
# =========================

class TimestampMixin:
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now()
    )

    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )


# =========================
# USER
# =========================

class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    username = Column(
        String(50),
        unique=True,
        nullable=False,
        index=True
    )

    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    password_hash = Column(
        String(255),
        nullable=False
    )

    role = Column(
        SqlEnum(UserRole, native_enum=False),
        nullable=False,
        default=UserRole.USER
    )

    is_active = Column(
        Boolean,
        nullable=False,
        default=True
    )

    # Relationships
    documents = relationship(
        "Document",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    conversations = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True
    )


# =========================
# DOCUMENT
# =========================

class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    title = Column(
        String(500),
        nullable=False
    )

    file_name = Column(
        String(500),
        nullable=False
    )

    file_type = Column(
        String(100),
        nullable=False
    )

    file_size = Column(
        BigInteger,
        nullable=False
    )

    language = Column(
        String(20),
        nullable=False,
        default="en"
    )

    status = Column(
        SqlEnum(DocumentStatus, native_enum=False),
        nullable=False,
        default=DocumentStatus.PENDING,
        index=True
    )

    chunk_count = Column(
        Integer,
        nullable=False,
        default=0
    )

    error_message = Column(Text)

    doc_metadata = Column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}"
    )

    user = relationship(
        "User",
        back_populates="documents"
    )

    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True
    )


# =========================
# DOCUMENT CHUNK
# =========================

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    chunk_index = Column(
        Integer,
        nullable=False
    )

    content = Column(
        Text,
        nullable=False
    )

    embedding = Column(
        Vector(768),
        nullable=False
    )

    token_count = Column(
        Integer,
        nullable=False,
        default=0
    )

    doc_metadata = Column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}"
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now()
    )

    document = relationship(
        "Document",
        back_populates="chunks"
    )

    __table_args__ = (
        Index("idx_chunk_document_chunk", "document_id", "chunk_index"),
    )


# =========================
# CONVERSATION
# =========================

class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    title = Column(String(500))

    user = relationship(
        "User",
        back_populates="conversations"
    )

    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at.asc()"
    )


# =========================
# MESSAGE
# =========================

class Message(Base):
    __tablename__ = "messages"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    role = Column(String(20), nullable=False)

    content = Column(
        Text,
        nullable=False
    )

    sources = Column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]"
    )

    token_count = Column(Integer)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now()
    )

    conversation = relationship(
        "Conversation",
        back_populates="messages"
    )

    __table_args__ = (
        Index("idx_message_conversation_created", "conversation_id", "created_at"),
    )

