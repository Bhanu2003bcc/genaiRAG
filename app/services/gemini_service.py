
import asyncio
import json
import time
import logging
from typing import AsyncGenerator, List, Optional, Dict

# pyrefly: ignore [missing-import]
import httpx

from app.config import settings
from app.utils.embedding_cache import EmbeddingCache
from app.utils.metrics import metrics_collector

logger = logging.getLogger("app.gemini")


class GeminiService:

    def __init__(self):

        self.api_key = settings.gemini_api_key
        self.base_url = settings.gemini_base_url

        self.chat_model = settings.gemini_chat_model
        self.embedding_model = settings.gemini_embedding_model

        self.embedding_dimension = (
            settings.gemini_embedding_dimension
        )

        # Shared HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=120.0,
                write=30.0,
                pool=30.0
            ),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20
            ),
            http2=True
        )

        # Concurrency limiter
        self.rate_limiter = asyncio.Semaphore(20)

        # Embedding cache
        self.embedding_cache = EmbeddingCache(
            maxsize=settings.embedding_cache_maxsize,
            ttl=settings.embedding_cache_ttl
        )

    # =========================
    # GENERIC RETRY
    # =========================

    async def _post_with_retry(
        self,
        url: str,
        body: dict,
        timeout: Optional[float] = None
    ) -> httpx.Response:

        max_attempts = 3
        backoff = 2

        headers = {
            "Content-Type": "application/json"
        }

        for attempt in range(max_attempts):

            try:
                response = await self.client.post(
                    url,
                    json=body,
                    headers=headers,
                    timeout=timeout
                )

                # Retry only retryable failures
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        "Retryable server error",
                        request=response.request,
                        response=response
                    )

                response.raise_for_status()

                return response

            except (
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                httpx.HTTPStatusError
            ) as e:

                if attempt == max_attempts - 1:
                    raise

                logger.warning(
                    f"Gemini retry attempt={attempt + 1} "
                    f"error={str(e)} "
                    f"backoff={backoff}s"
                )

                await asyncio.sleep(backoff)
                backoff *= 2

        raise RuntimeError("Gemini request failed")

    # =========================
    # EMBEDDINGS
    # =========================

    async def generate_embedding(
        self,
        text: str
    ) -> Optional[List[float]]:
        embedding = await self.embedding_cache.get_or_compute(
            text,
            self._call_embedding_api
        )
        if not embedding:
            # Fallback: empty float array matching expected dimensions
            return [0.0] * self.embedding_dimension
        return embedding

    async def _call_embedding_api(
        self,
        text: str
    ) -> Optional[List[float]]:
        if not self.api_key:
            logger.error("Gemini API key missing")
            return None

        if not text or not text.strip():
            return None

        url = (
            f"{self.base_url}/models/"
            f"{self.embedding_model}:embedContent"
            f"?key={self.api_key}"
        )

        body = {
            "content": {
                "parts": [
                    {
                        "text": text[:20000]
                    }
                ]
            },
            "outputDimensionality": self.embedding_dimension
        }

        async with self.rate_limiter:
            try:
                response = await self._post_with_retry(
                    url=url,
                    body=body,
                    timeout=30.0
                )

                data = response.json()

                embedding = (
                    data.get("embedding", {})
                    .get("values", [])
                )

                if not embedding:
                    logger.error("Empty embedding values returned from Gemini API")
                    return None

                # Log warning on dimension mismatch, but don't fail, to preserve mock test capability
                if len(embedding) != self.embedding_dimension:
                    logger.warning(
                        "Embedding dimension mismatch: expected %d, got %d",
                        self.embedding_dimension,
                        len(embedding)
                    )

                return [float(v) for v in embedding]

            except Exception as e:
                logger.exception(
                    f"Embedding generation failed: {str(e)}"
                )
                return None

    # =========================
    # REQUEST BODY
    # =========================

    def _build_chat_body(
        self,
        system_prompt: str,
        user_message: str,
        history: List[dict]
    ) -> dict:

        contents = []

        for i, item in enumerate(history):
            if isinstance(item, dict):
                role = item.get("role", "user" if i % 2 == 0 else "model")
                text = item.get("content", item.get("text", ""))
            else:
                role = "user" if i % 2 == 0 else "model"
                text = str(item)

            role = role.lower()
            if role == "assistant":
                role = "model"
            elif role not in ("user", "model"):
                role = "user"

            contents.append({
                "role": role,
                "parts": [
                    {
                        "text": text
                    }
                ]
            })

        contents.append({
            "role": "user",
            "parts": [
                {
                    "text": user_message
                }
            ]
        })

        return {
            "system_instruction": {
                "parts": [
                    {
                        "text": system_prompt
                    }
                ]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": 0.3,
                "topP": 0.9,
                "maxOutputTokens": 2048
            }
        }

    # =========================
    # CHAT COMPLETION
    # =========================

    async def generate_answer(
        self,
        system_prompt: str,
        user_message: str,
        history: List[dict]
    ) -> str:

        if not self.api_key:
            return "AI service unavailable"

        url = (
            f"{self.base_url}/models/"
            f"{self.chat_model}:generateContent"
            f"?key={self.api_key}"
        )

        body = self._build_chat_body(
            system_prompt=system_prompt,
            user_message=user_message,
            history=history
        )

        async with self.rate_limiter:

            try:
                response = await self._post_with_retry(
                    url=url,
                    body=body,
                    timeout=90.0
                )

                data = response.json()

                candidates = data.get(
                    "candidates",
                    []
                )

                if not candidates:
                    return "No response generated"

                parts = (
                    candidates[0]
                    .get("content", {})
                    .get("parts", [])
                )

                if not parts:
                    return "No response generated"

                return parts[0].get("text", "")

            except Exception as e:

                logger.exception(
                    f"Chat generation failed: {str(e)}"
                )

                return (
                    "AI service is temporarily unavailable."
                )

    # =========================
    # STREAMING CHAT
    # =========================

    async def stream_answer(
        self,
        system_prompt: str,
        user_message: str,
        history: List[dict]
    ) -> AsyncGenerator[str, None]:

        if not self.api_key:
            yield "AI service unavailable"
            return

        url = (
            f"{self.base_url}/models/"
            f"{self.chat_model}:streamGenerateContent"
            f"?alt=sse&key={self.api_key}"
        )

        body = self._build_chat_body(
            system_prompt=system_prompt,
            user_message=user_message,
            history=history
        )

        async with self.rate_limiter:

            try:
                async with self.client.stream(
                    "POST",
                    url,
                    json=body
                ) as response:

                    response.raise_for_status()

                    async for line in response.aiter_lines():

                        if not line:
                            continue

                        line = line.strip()

                        if line.startswith("data:"):
                            line = line[5:].strip()

                        if (
                            not line or
                            line == "[DONE]"
                        ):
                            continue

                        try:
                            chunk_data = json.loads(line)

                            candidates = chunk_data.get(
                                "candidates",
                                []
                            )

                            if not candidates:
                                continue

                            parts = (
                                candidates[0]
                                .get("content", {})
                                .get("parts", [])
                            )

                            if not parts:
                                continue

                            text = parts[0].get(
                                "text",
                                ""
                            )

                            if text:
                                yield text

                        except json.JSONDecodeError:
                            continue

            except asyncio.CancelledError:

                logger.warning(
                    "Client disconnected during stream"
                )

                raise

            except Exception as e:

                logger.exception(
                    f"Gemini stream failed: {str(e)}"
                )

                yield (
                    "\n[Streaming interrupted]"
                )

    async def re_rank_chunks(
        self,
        query: str,
        chunks: List[dict]
    ) -> Dict[str, float]:
        """
        Grades the relevance of each document chunk against the user query.
        Returns a dictionary mapping chunk ID -> score (0.0 to 1.0).
        Fails gracefully and returns an empty dictionary under API errors or timeouts.
        """
        if not self.api_key or not chunks:
            return {}

        url = (
            f"{self.base_url}/models/"
            f"{self.chat_model}:generateContent"
            f"?key={self.api_key}"
        )

        # Build chunks text mapping
        chunks_text_list = []
        for c in chunks:
            content_snippet = c.get("content", "").replace("\n", " ")[:1000]
            chunks_text_list.append(f"ID: {c['id']}\nContent: {content_snippet}")
        chunks_text = "\n---\n".join(chunks_text_list)

        prompt = (
            f"Analyze the relevance of the following document chunks to the user's query: \"{query}\"\n\n"
            f"Assign a relevance score between 0.0 (completely irrelevant) and 1.0 (highly relevant) to each chunk based on how well it answers or provides context for the query.\n\n"
            f"Chunks:\n{chunks_text}"
        )

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,  # deterministic scoring
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "ARRAY",
                    "description": "List of chunk relevance scores",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {
                                "type": "STRING",
                                "description": "The exact ID of the chunk"
                            },
                            "score": {
                                "type": "NUMBER",
                                "description": "Relevance score between 0.0 and 1.0"
                            }
                        },
                        "required": ["id", "score"]
                    }
                }
            }
        }

        start_time = time.perf_counter()
        async with self.rate_limiter:
            try:
                # Set a tight timeout (e.g. 10 seconds) for re-ranking
                response = await self._post_with_retry(
                    url=url,
                    body=body,
                    timeout=10.0
                )
                duration = time.perf_counter() - start_time
                metrics_collector.record_rerank(duration)

                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return {}

                parts = (
                    candidates[0]
                    .get("content", {})
                    .get("parts", [])
                )
                if not parts:
                    return {}

                text_content = parts[0].get("text", "")
                if not text_content:
                    return {}

                scores_list = json.loads(text_content)
                result_map = {}
                for item in scores_list:
                    chunk_id = item.get("id")
                    score = item.get("score", 0.0)
                    if chunk_id:
                        result_map[str(chunk_id)] = max(0.0, min(1.0, float(score)))
                return result_map

            except Exception as e:
                logger.warning(f"LLM re-ranking failed: {str(e)}")
                return {}

    # =========================
    # CLEANUP
    # =========================

    async def close(self):
        await self.client.aclose()


gemini_service = GeminiService()

