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

## System Architecture & Data Flow

The backend uses a hybrid lexical (FTS) and semantic (vector) search engine powered by `pgvector` and the Google Gemini API. Below is the system flow showing how document ingestion and chat query retrieval are structured:

```mermaid
graph TD
    %% Styling
    classDef client fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef backend fill:#efebe9,stroke:#3e2723,stroke-width:2px;
    classDef db fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef external fill:#fff3e0,stroke:#e65100,stroke-width:2px;

    User[Client Application]:::client
    
    subgraph FastAPI Backend
        API[FastAPI Routers]:::backend
        DocService[Document Service]:::backend
        ChatService[Chat Service]:::backend
        Chunker[Text / Semantic Chunker]:::backend
        Cache[In-Memory Cache]:::backend
    end
    
    subgraph Database
        PG[(PostgreSQL + pgvector)]:::db
        HNSW[HNSW Vector Index]:::db
        GIN[GIN Full-Text Index]:::db
    end

    subgraph External Services
        Gemini[Google Gemini API]:::external
    end

    %% Document Ingestion Flow
    User -->|1. Upload Document| API
    API -->|Async Task| DocService
    DocService -->|2. Parse Text| Chunker
    Chunker -->|3. Generate Embeddings| Gemini
    DocService -->|4. Save Chunks + Vectors| PG

    %% Chat & Retrieval Flow
    User -->|5. Chat Query| API
    API -->|6. Search Query| ChatService
    ChatService -->|7. Embed Query| Gemini
    ChatService -->|8. Hybrid Search| PG
    PG -->|Vector Search| HNSW
    PG -->|Keyword Search| GIN
    ChatService -->|9. Reciprocal Rank Fusion| ChatService
    ChatService -->|10. Re-rank Chunks| Gemini
    ChatService -->|11. Generate Answer| Gemini
    ChatService -->|12. Save History| PG
    ChatService -->|13. Stream SSE / JSON Response| User
```

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

## API Documentation Reference

All API responses wrap their data inside a unified wrapper response schema:
```json
{
  "success": true,
  "message": "Optional message string",
  "data": null
}
```

### 1. Authentication

#### Register a New User
* **Endpoint**: `POST /api/auth/register`
* **Request Body**:
```json
{
  "username": "johndoe",
  "email": "johndoe@example.com",
  "password": "securepassword123"
}
```
* **Response Body (`200 OK`)**:
```json
{
  "success": true,
  "message": "User registered successfully",
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "tokenType": "Bearer",
    "userId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "username": "johndoe",
    "email": "johndoe@example.com",
    "role": "ROLE_USER"
  }
}
```

#### User Login
* **Endpoint**: `POST /api/auth/login`
* **Request Body**:
```json
{
  "username": "johndoe",
  "password": "securepassword123"
}
```
* **Response Body (`200 OK`)**:
```json
{
  "success": true,
  "message": "Login successful",
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "tokenType": "Bearer",
    "userId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "username": "johndoe",
    "email": "johndoe@example.com",
    "role": "ROLE_USER"
  }
}
```

---

### 2. Documents

> [!NOTE]
> All document management endpoints require an `Authorization: Bearer <token>` header.

#### Upload Document
* **Endpoint**: `POST /api/documents/upload`
* **Request**: `multipart/form-data` with key `file` containing the file (PDF, TXT, DOCX, PPTX, HTML).
* **Response Body (`200 OK`)**:
```json
{
  "success": true,
  "message": "Document uploaded. Processing started.",
  "data": {
    "id": "e4a5d89f-2244-42b7-a36c-2fbdde24ea6b",
    "title": "financial_report_2026",
    "fileName": "financial_report_2026.pdf",
    "fileType": "application/pdf",
    "fileSize": 142850,
    "language": "en",
    "status": "PENDING",
    "chunkCount": 0,
    "errorMessage": null,
    "createdAt": "2026-06-10T23:00:00Z",
    "updatedAt": "2026-06-10T23:00:00Z"
  }
}
```

#### Fetch Paginated Documents
* **Endpoint**: `GET /api/documents?page=0&size=20`
* **Response Body (`200 OK`)**:
```json
{
  "success": true,
  "message": null,
  "data": {
    "content": [
      {
        "id": "e4a5d89f-2244-42b7-a36c-2fbdde24ea6b",
        "title": "financial_report_2026",
        "fileName": "financial_report_2026.pdf",
        "fileType": "application/pdf",
        "fileSize": 142850,
        "language": "en",
        "status": "COMPLETED",
        "chunkCount": 8,
        "errorMessage": null,
        "createdAt": "2026-06-10T23:00:00Z",
        "updatedAt": "2026-06-10T23:00:05Z"
      }
    ],
    "page": 0,
    "size": 20,
    "totalElements": 1,
    "totalPages": 1
  }
}
```

#### Delete Document
* **Endpoint**: `DELETE /api/documents/{document_id}`
* **Response Body (`200 OK`)**:
```json
{
  "success": true,
  "message": "Document deleted",
  "data": null
}
```

---

### 3. Chat & Conversations

> [!NOTE]
> All chat/conversation endpoints require an `Authorization: Bearer <token>` header.

#### Send Chat Message (Non-Streaming)
* **Endpoint**: `POST /api/chat`
* **Request Body**:
```json
{
  "message": "What is our Q1 net profit margin?",
  "conversationId": null,
  "stream": false,
  "maxResults": 5
}
```
* **Response Body (`200 OK`)**:
```json
{
  "success": true,
  "message": null,
  "data": {
    "conversationId": "3490fdba-38a4-4f27-9dbf-1c4b8b60b6fa",
    "messageId": "a90dfbaa-f5c2-480c-a99f-7f89db8a73b2",
    "answer": "According to the financial report, our Q1 net profit margin is 24.5%, representing a 2% growth year-over-year.",
    "sources": [
      {
        "documentId": "e4a5d89f-2244-42b7-a36c-2fbdde24ea6b",
        "documentTitle": "financial_report_2026",
        "excerpt": "...net profit margin for Q1 has reached 24.5%, driven by operational efficiency gains...",
        "chunkIndex": 2,
        "relevanceScore": 0.8924
      }
    ],
    "followUp": false
  }
}
```

#### Send Chat Message (SSE Streaming)
* **Endpoint**: `POST /api/chat/stream`
* **Request Body**:
```json
{
  "message": "What is our Q1 net profit margin?",
  "conversationId": "3490fdba-38a4-4f27-9dbf-1c4b8b60b6fa",
  "stream": true,
  "maxResults": 5
}
```
* **Response Response (Server-Sent Events)**:
```text
event: sources
data: [{"documentId": "e4a5d89f-2244-42b7-a36c-2fbdde24ea6b", "documentTitle": "financial_report_2026", "excerpt": "...net profit margin for Q1...", "chunkIndex": 2, "relevanceScore": 0.8924}]

event: chunk
data: According

event: chunk
data:  to

event: chunk
data:  the

event: chunk
data:  financial

event: chunk
data:  report...

event: done
data: {"conversationId": "3490fdba-38a4-4f27-9dbf-1c4b8b60b6fa", "messageId": "b61cf26a-c5d0-4e12-bb2f-cc810d7b21ea"}
```

#### Fetch Full Conversation History
* **Endpoint**: `GET /api/chat/conversations/{conversation_id}`
* **Response Body (`200 OK`)**:
```json
{
  "success": true,
  "message": null,
  "data": {
    "id": "3490fdba-38a4-4f27-9dbf-1c4b8b60b6fa",
    "title": "What is our Q1 net profit margin?",
    "messages": [
      {
        "id": "c61dfaba-b4a1-432d-8bfe-ef871bfa26cd",
        "role": "user",
        "content": "What is our Q1 net profit margin?",
        "sources": [],
        "createdAt": "2026-06-10T23:02:10Z"
      },
      {
        "id": "a90dfbaa-f5c2-480c-a99f-7f89db8a73b2",
        "role": "assistant",
        "content": "According to the financial report, our Q1 net profit margin is 24.5%...",
        "sources": [
          {
            "documentId": "e4a5d89f-2244-42b7-a36c-2fbdde24ea6b",
            "documentTitle": "financial_report_2026",
            "excerpt": "...net profit margin for Q1 has reached 24.5%...",
            "chunkIndex": 2,
            "relevanceScore": 0.8924
          }
        ],
        "createdAt": "2026-06-10T23:02:12Z"
      }
    ],
    "createdAt": "2026-06-10T23:02:10Z",
    "updatedAt": "2026-06-10T23:02:12Z"
  }
}
```

---

### 4. System Health

#### App Health Status
* **Endpoint**: `GET /api/actuator/health`
* **Response Body (`200 OK`)**:
```json
{
  "success": true,
  "message": null,
  "data": {
    "status": "UP"
  }
}
```
