import logging
import asyncio
import json
from typing import List, Callable, Generator, AsyncGenerator
import httpx
from app.config import settings

logger = logging.getLogger("com.rag.service.impl.GeminiServiceImpl")

class GeminiService:
    def __init__(self):
        self.api_key = settings.gemini_api_key
        self.base_url = settings.gemini_base_url
        self.chat_model = settings.gemini_chat_model
        self.embedding_model = settings.gemini_embedding_model
        self.embedding_dimension = settings.gemini_embedding_dimension
        # Rate limiter: 10 concurrent requests matching Spring Semaphore
        self.rate_limiter = asyncio.Semaphore(10)

    async def _post_with_retry(self, client: httpx.AsyncClient, url: str, body: dict, timeout: float) -> httpx.Response:
        attempts = 3
        wait_duration = 2.0
        for attempt in range(attempts):
            try:
                response = await client.post(url, json=body, timeout=timeout)
                response.raise_for_status()
                return response
            except Exception as e:
                if attempt == attempts - 1:
                    raise e
                logger.warning(
                    f"Gemini API attempt {attempt + 1} failed: {str(e)}. "
                    f"Retrying in {wait_duration}s..."
                )
                await asyncio.sleep(wait_duration)
                wait_duration *= 2
        raise RuntimeError("Request failed after max retries")

    async def generate_embedding(self, text: str) -> List[float]:
        if not self.api_key:
            logger.error("Gemini API key is not configured.")
            return [0.0] * self.embedding_dimension

        async with self.rate_limiter:
            url = f"{self.base_url}/models/{self.embedding_model}:embedContent?key={self.api_key}"
            body = {
                "content": {
                    "parts": [{"text": text}]
                },
                "outputDimensionality": self.embedding_dimension
            }
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await self._post_with_retry(client, url, body, timeout=30.0)
                    data = response.json()
                    values = data.get("embedding", {}).get("values", [])
                    return [float(v) for v in values]
            except Exception as e:
                logger.error(f"Circuit breaker: embedding fallback triggered. Error: {str(e)}")
                # Fallback: empty float array
                return [0.0] * self.embedding_dimension

    def _build_chat_body(self, system_prompt: str, user_message: str, history: List[str]) -> dict:
        body = {}
        contents = []

        # System instructions
        if system_prompt and system_prompt.strip():
            body["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        # History: alternate user and model roles.
        # Spring logs: i % 2 == 0 ? "user" : "model"
        for i, text in enumerate(history):
            role = "user" if i % 2 == 0 else "model"
            contents.append({
                "role": role,
                "parts": [{"text": text}]
            })

        # Current user message
        contents.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        body["contents"] = contents
        body["generationConfig"] = {
            "temperature": 0.7,
            "topP": 0.95,
            "maxOutputTokens": 2048
        }
        return body

    async def generate_answer(self, system_prompt: str, user_message: str, history: List[str]) -> str:
        if not self.api_key:
            return "Gemini API key is not configured."

        async with self.rate_limiter:
            url = f"{self.base_url}/models/{self.chat_model}:generateContent?key={self.api_key}"
            body = self._build_chat_body(system_prompt, user_message, history)
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await self._post_with_retry(client, url, body, timeout=60.0)
                    data = response.json()
                    
                    # Extract text: root.path("candidates").get(0).path("content").path("parts").get(0).path("text")
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                    return ""
            except Exception as e:
                logger.error(f"Circuit breaker: chat fallback triggered. Error: {str(e)}")
                return "I'm currently experiencing high load. Please try again in a moment."

    async def stream_answer(self, system_prompt: str, user_message: str, history: List[str]) -> AsyncGenerator[str, None]:
        """
        Asynchronously streams chunk responses from Gemini using SSE.
        Returns an AsyncGenerator yielding tokens.
        """
        if not self.api_key:
            yield "Gemini API key is not configured."
            return

        url = f"{self.base_url}/models/{self.chat_model}:streamGenerateContent?key={self.api_key}&alt=sse"
        body = self._build_chat_body(system_prompt, user_message, history)

        # Semaphore rate limiting is applied for streaming as well
        async with self.rate_limiter:
            try:
                async with httpx.AsyncClient() as client:
                    # Stream POST request
                    async with client.stream("POST", url, json=body, timeout=120.0) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Parse Server-Sent Events prefix
                            if line.startswith("data: "):
                                json_str = line[6:].strip()
                            else:
                                json_str = line
                            
                            if not json_str or json_str == "[DONE]":
                                continue
                            
                            try:
                                chunk_data = json.loads(json_str)
                                candidates = chunk_data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    if parts:
                                        text = parts[0].get("text", "")
                                        if text:
                                            yield text
                            except Exception:
                                # Skip malformed SSE chunks matching doOnError / logging debug in Java
                                continue
            except Exception as e:
                logger.error(f"SSE stream error: {str(e)}")
                yield "\n[Stream Error: Connection failed or timed out]"

# Instantiate service singleton
gemini_service = GeminiService()
