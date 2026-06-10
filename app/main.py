import time
import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.routers import auth, documents, chat
from app.schemas import ApiResponse
from app.utils.db_indexes import ensure_indexes
from app.utils.metrics import metrics_collector
from app.observability.logging import configure_logging, request_id_var
from app.utils.limiter import limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Configure production-grade structured JSON logging
configure_logging()
logger = logging.getLogger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    On startup: create any missing database indexes (HNSW, GIN, B-Tree).
    The function is idempotent — safe to run on every deploy.
    """
    logger.info("Starting up RAG System Backend...")
    await ensure_indexes()
    yield
    logger.info("Shutting down RAG System Backend.")


# Initialize FastAPI application
app = FastAPI(
    title="RAG System Backend",
    description="Python/FastAPI rewrite of the Java pgvector RAG backend",
    version="1.0.0",
    lifespan=lifespan,
)

# Wire SlowAPI Rate Limiter
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

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


# Request ID Middleware - executes first in the ASGI stack
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    token = request_id_var.set(req_id)
    try:
        response = await call_next(request)
        response.headers["x-request-id"] = req_id
        return response
    finally:
        request_id_var.reset(token)


# Metrics Middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    path = request.url.path
    if "actuator" in path or path == "/metrics":
        return await call_next(request)

    metrics_collector.increment_active_requests()
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        metrics_collector.record_request(request.method, path, response.status_code, duration)
        return response
    except Exception as e:
        duration = time.perf_counter() - start_time
        metrics_collector.record_request(request.method, path, 500, duration)
        raise e
    finally:
        metrics_collector.decrement_active_requests()


# Spring Boot Health Actuator match
@app.get(f"{settings.api_prefix}/actuator/health", tags=["Actuator"])
async def health_check():
    # Return Spring-style UP response
    return {"status": "UP"}


@app.get(f"{settings.api_prefix}/actuator/metrics", tags=["Actuator"])
async def get_metrics():
    return metrics_collector.get_stats()


@app.get(f"{settings.api_prefix}/actuator/metrics/prometheus", tags=["Actuator"])
@app.get("/metrics", tags=["Actuator"])
async def get_prometheus_metrics():
    return Response(
        content=metrics_collector.get_prometheus_metrics(),
        media_type="text/plain"
    )


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


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    Handles RateLimitExceeded exceptions from SlowAPI rate limiter.
    """
    logger.warning(f"Rate limit exceeded: {exc.detail}")
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=ApiResponse.error_response(message="Too many requests. Please try again later.").dict()
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
