# RAG System Backend (Python/FastAPI)

A high-performance Python/FastAPI rewrite of the Java pgvector RAG (Retrieval-Augmented Generation) backend. It connects to a Supabase PostgreSQL database using `pgvector` for hybrid search retrieval, and processes chats and embeddings using Google's Gemini API.

## Key Features

- **Hybrid Document Search**: Combines semantic vector similarity search (using `pgvector` & `gemini-embedding-2`) and keyword-based lexical search (`ts_rank` text search) for optimal context retrieval.
- **Asynchronous Document Processing**: Documents (PDF, TXT, DOCX, PPTX, HTML) are uploaded, parsed, chunked, and embedded asynchronously in background tasks.
- **SSE Streaming Chat**: Real-time server-sent events (SSE) streaming of chat responses from `gemini-2.5-flash-lite`.
- **JWT Authentication**: Secure API endpoints with sign-up and login capabilities matching Spring Boot authority conventions.
- **Unified Schema & Error Handling**: Pydantic models structure all payloads and response validation errors return clean Spring-compatible global error bodies.

---

## Tech Stack

- **Framework**: FastAPI (Uvicorn ASGI server)
- **Database ORM**: SQLAlchemy 2.0 (with `asyncpg` for async execution)
- **Vector Extension**: `pgvector-python`
- **LLM / Embedding Services**: Google Gemini API via `httpx` (featuring retry/backoff policy and rate-limiting semaphores)
- **Authentication**: `python-jose` (HS256 JWT) & `passlib` (Bcrypt hashing)
- **Parsing**: `pypdf`, `python-docx`, `python-pptx`, `beautifulsoup4`

---

## Getting Started

### Prerequisites

- Python 3.11+
- A running PostgreSQL instance with the `pgvector` extension enabled (e.g. Supabase).

### Installation & Local Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository_url>
   cd backend-python
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory and specify the following variables:
   ```env
   # Gemini API Key
   GEMINI_API_KEY=your_gemini_api_key_here

   # Database URL (JDBC or standard PostgreSQL URL is converted automatically by app/config.py)
   DATABASE_URL=postgresql://username:password@host:port/database_name?sslmode=require
   DATABASE_USERNAME=postgres
   DATABASE_PASSWORD=your_db_password

   # JWT Secret (min 256-bit / 32 characters)
   JWT_SECRET=your_secure_random_jwt_secret_key_here

   # CORS Allowed Origins
   CORS_ORIGINS=http://localhost:5173,http://localhost:3000
   ```

---

## Running the Application

Start the development server with hot-reloading:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at `http://localhost:8080` with documentation interactive dashboards at:
- **Swagger UI**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **ReDoc**: [http://localhost:8080/redoc](http://localhost:8080/redoc)

---

## Running Tests

The project includes two types of tests:

1. **Component / Unit Tests**:
   Tests internal utilities (text chunker, document parsers, security helper logic, and Gemini service mocks).
   ```bash
   PYTHONPATH=. python tests/test_components.py
   ```

2. **Live Integration API Tests**:
   Tests the entire workflow against a running server at `http://localhost:8080` (health, registration, logins, uploads, non-streaming chat, and streaming SSE chat). Ensure your server is running before executing:
   ```bash
   PYTHONPATH=. python tests/test_api_live.py
   ```

---

## Docker Support

You can run the backend server inside a Docker container:

1. **Build the image**:
   ```bash
   docker build -t rag-backend-python .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8080:8080 --env-file .env rag-backend-python
   ```

---

## API Documentation Quick Reference

### Authentication
- `POST /api/auth/register` - Create a new user account.
- `POST /api/auth/login` - Authenticate credentials and receive a JWT token.

### Documents
- `POST /api/documents/upload` - Upload document file (`multipart/form-data`).
- `GET /api/documents` - Fetch paginated user documents list.
- `GET /api/documents/{id}` - Fetch single document metadata.
- `DELETE /api/documents/{id}` - Delete document and all associated chunks.

### Chats & Conversations
- `POST /api/chat` - Send a message (returns final JSON answer with sources).
- `POST /api/chat/stream` - Send a message (SSE stream yielding metadata sources, answer chunks, and `done`).
- `GET /api/chat/conversations` - Fetch paginated conversation list.
- `GET /api/chat/conversations/{id}` - Fetch full conversation message history.
- `DELETE /api/chat/conversations/{id}` - Delete conversation history.

### Health
- `GET /api/actuator/health` - Return application running status (`{"status": "UP"}`).
