from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas import RegisterRequest, LoginRequest, AuthResponse, ApiResponse
from app.services.user_service import user_service

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=ApiResponse[AuthResponse])
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    auth_data = await user_service.register(request, db)
    return ApiResponse.success_response(
        data=auth_data, 
        message="User registered successfully"
    )

@router.post("/login", response_model=ApiResponse[AuthResponse])
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    auth_data = await user_service.login(request, db)
    return ApiResponse.success_response(data=auth_data)
