import logging
import math
from uuid import UUID
from typing import List, Tuple
from fastapi import HTTPException, status, UploadFile
from sqlalchemy import select, func, desc, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocumentChunk, User
from app.schemas import DocumentResponse, PageResponse
from app.database import AsyncSessionLocal
from app.utils.chunker import TextChunker
from app.utils.parser import parse_document
from app.services.gemini_service import gemini_service

logger = logging.getLogger("com.rag.service.impl.DocumentServiceImpl")

class DocumentService:
    def __init__(self):
        self.text_chunker = TextChunker()
        self.supported_types = {
            "application/pdf",
            "text/plain",
            "text/html",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        }

    def _strip_extension(self, filename: str) -> str:
        if not filename:
            return "Untitled"
        dot_idx = filename.rfind('.')
        return filename[:dot_idx] if dot_idx > 0 else filename

    def to_response(self, doc: Document) -> DocumentResponse:
        return DocumentResponse(
            id=doc.id,
            title=doc.title,
            fileName=doc.file_name,
            fileType=doc.file_type,
            fileSize=doc.file_size,
            language=doc.language,
            status=doc.status,
            chunkCount=doc.chunk_count,
            errorMessage=doc.error_message,
            createdAt=doc.created_at,
            updatedAt=doc.updated_at
        )

    async def upload(self, file: UploadFile, user: User, db: AsyncSession) -> Tuple[DocumentResponse, bytes]:
        # Validate file
        # Reading size directly
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )

        file_type = file.content_type
        if file_type not in self.supported_types:
            # Check file extension as a fallback in case browser sent generic octet-stream
            ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
            if ext == 'pdf':
                file_type = 'application/pdf'
            elif ext in ['docx', 'doc']:
                file_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            elif ext in ['pptx', 'ppt']:
                file_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
            elif ext in ['html', 'htm']:
                file_type = 'text/html'
            elif ext == 'txt':
                file_type = 'text/plain'
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file_type}"
                )

        # Create document record
        document = Document(
            user_id=user.id,
            title=self._strip_extension(file.filename),
            file_name=file.filename,
            file_type=file_type,
            file_size=len(file_bytes),
            status="PENDING",
            language="en"
        )
        db.add(document)
        await db.commit()
        await db.refresh(document)

        return self.to_response(document), file_bytes

    async def process_document_async(self, document_id: UUID, file_bytes: bytes):
        """
        Background task to process the document text, chunk it, 
        generate embeddings, and save document chunks.
        """
        # Create a dedicated db session for background execution
        async with AsyncSessionLocal() as db:
            # Fetch document
            stmt = select(Document).filter(Document.id == document_id)
            result = await db.execute(stmt)
            document = result.scalars().first()
            if not document:
                logger.error(f"Async document processing failed: Document not found: {document_id}")
                return

            try:
                document.status = "PROCESSING"
                await db.commit()
                await db.refresh(document)

                # Parse document text
                raw_text = parse_document(document.file_type, file_bytes)
                if not raw_text or not raw_text.strip():
                    raise ValueError("Could not extract text from document")

                # Chunk document content
                chunks = self.text_chunker.chunk(raw_text)
                if not chunks:
                    raise ValueError("No text chunks generated")

                # Generate embeddings and save chunks
                chunk_entities = []
                for i, chunk_text in enumerate(chunks):
                    # Call Gemini embedding API
                    embedding = await gemini_service.generate_embedding(chunk_text)
                    token_count = len(chunk_text.split())
                    
                    chunk = DocumentChunk(
                        document_id=document.id,
                        chunk_index=i,
                        content=chunk_text,
                        embedding=embedding,
                        token_count=token_count
                    )
                    chunk_entities.append(chunk)

                db.add_all(chunk_entities)
                document.chunk_count = len(chunks)
                document.status = "COMPLETED"
                await db.commit()
                logger.info(f"Document processed successfully: {document_id} with {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Document processing failed for id: {document_id}. Error: {str(e)}", exc_info=True)
                document.status = "FAILED"
                document.error_message = str(e)
                await db.commit()

    async def get_documents(self, user_id: UUID, page: int, size: int, db: AsyncSession) -> PageResponse[DocumentResponse]:
        # Count total elements
        count_stmt = select(func.count()).select_from(Document).filter(Document.user_id == user_id)
        count_result = await db.execute(count_stmt)
        total_elements = count_result.scalar() or 0

        # Query paginated documents
        stmt = (
            select(Document)
            .filter(Document.user_id == user_id)
            .order_by(desc(Document.created_at))
            .offset(page * size)
            .limit(size)
        )
        result = await db.execute(stmt)
        documents = result.scalars().all()

        content = [self.to_response(doc) for doc in documents]
        total_pages = math.ceil(total_elements / size) if size > 0 else 0

        return PageResponse(
            content=content,
            page=page,
            size=size,
            totalElements=total_elements,
            totalPages=total_pages
        )

    async def get_document(self, document_id: UUID, user_id: UUID, db: AsyncSession) -> DocumentResponse:
        stmt = select(Document).filter(Document.id == document_id, Document.user_id == user_id)
        result = await db.execute(stmt)
        doc = result.scalars().first()
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found with id: {str(document_id)}"
            )
        return self.to_response(doc)

    async def delete_document(self, document_id: UUID, user_id: UUID, db: AsyncSession) -> None:
        # Check ownership
        stmt = select(Document).filter(Document.id == document_id, Document.user_id == user_id)
        result = await db.execute(stmt)
        doc = result.scalars().first()
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found with id: {str(document_id)}"
            )
        
        # Deleting chunks (SQLAlchemy cascade deletes, but we can also issue bulk delete for safety)
        await db.execute(delete(DocumentChunk).filter(DocumentChunk.document_id == document_id))
        await db.delete(doc)
        await db.commit()

# Instantiate service singleton
document_service = DocumentService()
