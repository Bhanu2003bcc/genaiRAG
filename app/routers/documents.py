from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import (
    ApiResponse,
    DocumentResponse,
    PageResponse,
    CurrentUser
)
from app.security import get_current_user
from app.services.document_service import document_service

router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)

# =========================
# UPLOAD DOCUMENT
# =========================

@router.post(
    "/upload",
    response_model=ApiResponse[DocumentResponse],
    status_code=status.HTTP_200_OK
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename missing"
        )

    document_response, file_bytes = await document_service.upload(
        file=file,
        user=current_user,
        db=db
    )

    background_tasks.add_task(
        document_service.process_document_async,
        document_response.id,
        file_bytes
    )

    return ApiResponse.success_response(
        data=document_response,
        message="Document uploaded. Processing started."
    )

# =========================
# LIST DOCUMENTS
# =========================

@router.get(
    "",
    response_model=ApiResponse[PageResponse[DocumentResponse]]
)
async def get_documents(
    page: int = Query(0, ge=0),
    size: int = Query(20, ge=1, le=100),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    documents = await document_service.get_documents(
        current_user.id,
        page,
        size,
        db
    )

    return ApiResponse.success_response(data=documents)

# =========================
# GET DOCUMENT
# =========================

@router.get(
    "/{document_id}",
    response_model=ApiResponse[DocumentResponse]
)
async def get_document(
    document_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    document = await document_service.get_document(
        document_id,
        current_user.id,
        db
    )

    return ApiResponse.success_response(data=document)

# =========================
# DELETE DOCUMENT
# =========================

@router.delete(
    "/{document_id}",
    response_model=ApiResponse[None],
    status_code=status.HTTP_200_OK
)
async def delete_document(
    document_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    await document_service.delete_document(
        document_id,
        current_user.id,
        db
    )

    return ApiResponse.success_response(message="Document deleted")
