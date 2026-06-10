import asyncio
from uuid import UUID
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models import User
from app.schemas import RegisterRequest, LoginRequest, AuthResponse
from app.security import hash_password_async, verify_password_async, generate_token

class UserService:
    async def register(self, request: RegisterRequest, db: AsyncSession) -> AuthResponse:
        hashed_password = await hash_password_async(request.password)
        return await self.register_with_hash(request, hashed_password, db)

    async def register_with_hash(self, request: RegisterRequest, hashed_password: str, db: AsyncSession) -> AuthResponse:
        # Check if username exists
        stmt = select(User).filter(User.username == request.username)
        result = await db.execute(stmt)
        if result.scalars().first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )

        # Check if email exists
        stmt = select(User).filter(User.email == request.email)
        result = await db.execute(stmt)
        if result.scalars().first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Create user
        user = User(
            username=request.username,
            email=request.email,
            password_hash=hashed_password,
            role="USER",
            is_active=True
        )
        db.add(user)
        try:
            await db.commit()
            await db.refresh(user)
        except Exception as e:
            await db.rollback()
            from sqlalchemy.exc import IntegrityError
            if isinstance(e, IntegrityError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username or email already registered"
                )
            raise e

        # Generate token
        token = generate_token(user.id, user.username, user.role)
        return self._build_auth_response(user, token)

    async def login(self, request: LoginRequest, db: AsyncSession) -> AuthResponse:
        # Fetch user
        stmt = select(User).filter(User.username == request.username)
        result = await db.execute(stmt)
        user = result.scalars().first()
        
        # Release the connection back to the pool BEFORE the slow bcrypt operation!
        await db.close()
        
        # Verify user and password
        is_valid = False
        if user:
            is_valid = await verify_password_async(request.password, user.password_hash)
            
        if not user or not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
            
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated"
            )

        # Generate token
        token = generate_token(user.id, user.username, user.role)
        return self._build_auth_response(user, token)

    async def find_by_id(self, user_id: UUID, db: AsyncSession) -> User:
        stmt = select(User).filter(User.id == user_id)
        result = await db.execute(stmt)
        user = result.scalars().first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User not found with id: {str(user_id)}"
            )
        return user

    async def find_by_username(self, username: str, db: AsyncSession) -> User:
        stmt = select(User).filter(User.username == username)
        result = await db.execute(stmt)
        user = result.scalars().first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User not found: {username}"
            )
        return user

    def _build_auth_response(self, user: User, token: str) -> AuthResponse:
        return AuthResponse(
            token=token,
            tokenType="Bearer",
            userId=user.id,
            username=user.username,
            email=user.email,
            role=user.role
        )

# Instantiate service singleton
user_service = UserService()
