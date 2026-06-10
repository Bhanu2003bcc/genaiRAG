
# pyrefly: ignore [missing-import]
from fastapi import (
    APIRouter,
    Depends,
    status,
    Request
)
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import (
    ApiResponse,
    AuthResponse,
    LoginRequest,
    RegisterRequest
)
from app.services.user_service import user_service
from app.utils.limiter import limiter
from app.config import settings

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

# =========================
# REGISTER
# =========================

@router.post(
    "/register",
    response_model=ApiResponse[AuthResponse],
    status_code=status.HTTP_200_OK
)
@limiter.limit(settings.auth_rate_limit)
async def register(
    request: Request,
    register_request: RegisterRequest,
    db: AsyncSession = Depends(get_db)
):

    auth_response = await user_service.register(
        register_request,
        db
    )

    return ApiResponse.success_response(
        data=auth_response,
        message="User registered successfully"
    )

# =========================
# LOGIN
# =========================

@router.post(
    "/login",
    response_model=ApiResponse[AuthResponse],
    status_code=status.HTTP_200_OK
)
@limiter.limit(settings.auth_rate_limit)
async def login(
    request: Request,
    login_request: LoginRequest,
    db: AsyncSession = Depends(get_db)
):

    auth_response = await user_service.login(
        login_request,
        db
    )

    return ApiResponse.success_response(
        data=auth_response,
        message="Login successful"
    )
