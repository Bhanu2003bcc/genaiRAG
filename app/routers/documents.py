from uuid import UUID
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models import User
from app.security import get_current_user
from app.schemas import DocumentResponse, PageResponse, ApiResponse
from app.services.document_service import document_service

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", response_model=ApiResponse[DocumentResponse])
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    doc_resp, file_bytes = await document_service.upload(file, current_user, db)
    
    # Schedule asynchronous processing of document text & vector embeddings
    background_tasks.add_task(
        document_service.process_document_async, 
        doc_resp.id, 
        file_bytes
    )
    
    return ApiResponse.success_response(
        data=doc_resp, 
        message="Document uploaded and processing started"
    )

@router.get("", response_model=ApiResponse[PageResponse[DocumentResponse]])
async def get_documents(
    page: int = 0,
    size: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    paginated_docs = await document_service.get_documents(current_user.id, page, size, db)
    return ApiResponse.success_response(data=paginated_docs)

@router.get("/{document_id}", response_model=ApiResponse[DocumentResponse])
async def get_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    doc_resp = await document_service.get_document(document_id, current_user.id, db)
    return ApiResponse.success_response(data=doc_resp)

@router.delete("/{document_id}", response_model=ApiResponse[None])
async def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    await document_service.delete_document(document_id, current_user.id, db)
    return ApiResponse.success_response(message="Document deleted")
