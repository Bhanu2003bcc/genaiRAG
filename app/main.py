import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.routers import auth, documents, chat
from app.schemas import ApiResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

# Initialize FastAPI application
app = FastAPI(
    title="RAG System Backend",
    description="Python/FastAPI rewrite of the Java pgvector RAG backend",
    version="1.0.0"
)

# CORS Middleware setup
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(documents.router, prefix=settings.api_prefix)
app.include_router(chat.router, prefix=settings.api_prefix)


# Spring Boot Health Actuator match
@app.get(f"{settings.api_prefix}/actuator/health", tags=["Actuator"])
async def health_check():
    # Return Spring-style UP response
    return {"status": "UP"}


# Exception Handlers to match Spring Boot GlobalExceptionHandler formats

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles Pydantic payload validation errors and maps them to HTTP 400.
    Matches Spring's MethodArgumentNotValidException mapping.
    """
    errors = []
    for err in exc.errors():
        # Get field name from loc, ignore 'body' prefix
        loc_parts = err.get("loc", [])
        field_name = loc_parts[-1] if len(loc_parts) > 1 else str(loc_parts)
        msg = err.get("msg", "Validation error")
        errors.append(f"'{field_name}' {msg}")
        
    combined_errors = ", ".join(errors)
    logger.warn(f"Validation failed: {combined_errors}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ApiResponse.error_response(message=combined_errors).dict()
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Handles HTTPExceptions thrown explicitly inside controllers or dependencies.
    """
    # Map status codes to custom error messages if needed, matching Spring exceptions
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponse.error_response(message=exc.detail).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Fallback exception handler logging errors and returning standard 500 error code.
    """
    logger.exception("Unexpected error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ApiResponse.error_response(message="An unexpected error occurred").dict()
    )
