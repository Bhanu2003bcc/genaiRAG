import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.utils.chunker import TextChunker
from app.utils.parser import parse_document
from app.security import hash_password, verify_password, generate_token
from app.services.gemini_service import gemini_service

class TestRAGComponents(unittest.TestCase):
    
    def test_text_chunker(self):
        chunker = TextChunker()
        
        # Test short text (should remain 1 chunk)
        short_text = "Hello world from the Python RAG system backend."
        chunks = chunker.chunk(short_text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Hello world from the Python RAG system backend.")

        # Test spacing normalization
        spaced_text = "Word1     Word2\n\nWord3"
        chunks = chunker.chunk(spaced_text)
        self.assertEqual(chunks[0], "Word1 Word2 Word3")

        # Test window chunking (512 words, 64 overlap)
        long_words = ["word"] * 600
        long_text = " ".join(long_words)
        chunks = chunker.chunk(long_text)
        self.assertEqual(len(chunks), 2)
        # First chunk should have 512 words
        self.assertEqual(len(chunks[0].split()), 512)
        # Second chunk should have: remaining words + overlap
        # Remaining = 600 - 512 = 88. Overlap = 64. Total = 152.
        self.assertEqual(len(chunks[1].split()), 152)

    def test_security_utilities(self):
        password = "secure_password_123"
        hashed = hash_password(password)
        
        # Verify hash match
        self.assertTrue(verify_password(password, hashed))
        self.assertFalse(verify_password("wrong_password", hashed))

        # Verify JWT Token format and parsing
        import uuid
        user_id = uuid.uuid4()
        username = "rag_test_user"
        role = "ADMIN"
        
        token = generate_token(user_id, username, role)
        self.assertIsNotNone(token)
        self.assertTrue(isinstance(token, str))

    def test_document_parser(self):
        # Plain text
        text_bytes = b"Hello standard text parsing"
        self.assertEqual(parse_document("text/plain", text_bytes), "Hello standard text parsing")
        
        # HTML content
        html_bytes = b"<html><body><div>Some text</div><style>css-code</style></body></html>"
        parsed_html = parse_document("text/html", html_bytes)
        self.assertIn("Some text", parsed_html)
        self.assertNotIn("css-code", parsed_html)  # Script/style should be cleaned

    @patch("httpx.AsyncClient.post")
    def test_gemini_embedding(self, mock_post):
        # Inject API key for test
        gemini_service.api_key = "dummy_key"
        
        # Use real httpx.Response to mock responses accurately
        import httpx
        res = httpx.Response(
            status_code=200,
            json={
                "embedding": {
                    "values": [0.1, 0.2, 0.3, 0.4]
                }
            }
        )
        res.request = httpx.Request("POST", "http://test")
        mock_post.return_value = res

        # Execute
        embedding = asyncio.run(gemini_service.generate_embedding("sample text"))
        
        self.assertEqual(len(embedding), 4)
        self.assertEqual(embedding, [0.1, 0.2, 0.3, 0.4])

    @patch("httpx.AsyncClient.post")
    def test_gemini_generate_answer(self, mock_post):
        # Inject API key for test
        gemini_service.api_key = "dummy_key"
        
        # Use real httpx.Response to mock responses accurately
        import httpx
        res = httpx.Response(
            status_code=200,
            json={
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Gemini generated answer text"}]
                    }
                }]
            }
        )
        res.request = httpx.Request("POST", "http://test")
        mock_post.return_value = res

        # Execute
        answer = asyncio.run(
            gemini_service.generate_answer("System context", "User question", ["history user", "history model"])
        )
        
        self.assertEqual(answer, "Gemini generated answer text")

    def test_gemini_build_chat_body_role_mapping(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "ASSISTANT", "content": "how can I help?"}
        ]
        body = gemini_service._build_chat_body(
            system_prompt="system instructions",
            user_message="current question",
            history=history
        )
        contents = body.get("contents", [])
        self.assertEqual(len(contents), 4)
        
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[1]["role"], "model")
        self.assertEqual(contents[2]["role"], "model")
        self.assertEqual(contents[3]["role"], "user") # user_message gets user role

    def test_metrics_collector(self):
        from app.utils.metrics import MetricsCollector
        collector = MetricsCollector()
        
        collector.increment_active_requests()
        self.assertEqual(collector.active_requests, 1)
        
        collector.decrement_active_requests()
        self.assertEqual(collector.active_requests, 0)
        
        collector.record_request("GET", "/api/documents", 200, 0.15)
        collector.record_request("GET", "/api/documents", 200, 0.25)
        
        stats = collector.get_stats()
        self.assertEqual(stats["http"]["total_requests"], 2)
        endpoint_stats = stats["http"]["requests_by_endpoint"][0]
        self.assertEqual(endpoint_stats["endpoint"], "GET /api/documents 200")
        self.assertEqual(endpoint_stats["count"], 2)
        self.assertEqual(endpoint_stats["avg_duration_sec"], 0.20)
        
        # Caches
        collector.record_embedding_cache(hit=True)
        collector.record_embedding_cache(hit=False)
        self.assertEqual(collector.embedding_cache_hits, 1)
        self.assertEqual(collector.embedding_cache_misses, 1)

    def test_router_cache_ttl_expiration(self):
        from app.utils.router_cache import RouterCache
        cache = RouterCache(default_ttl=120)
        cache.set("key1", "data1", tags=["tag1"], ttl=-1) # expires immediately
        self.assertIsNone(cache.get("key1"))

    def test_router_cache_maxsize_eviction(self):
        from app.utils.router_cache import RouterCache
        cache = RouterCache(maxsize=2, default_ttl=120)
        cache.set("k1", "d1", tags=["t1"])
        cache.set("k2", "d2", tags=["t2"])
        cache.set("k3", "d3", tags=["t3"])
        
        # k1 should have been evicted
        self.assertIsNone(cache.get("k1"))
        self.assertEqual(cache.get("k2"), "d2")
        self.assertEqual(cache.get("k3"), "d3")

    def test_router_cache_tag_invalidation(self):
        from app.utils.router_cache import RouterCache
        cache = RouterCache(default_ttl=120)
        cache.set("k1", "d1", tags=["t1", "shared"])
        cache.set("k2", "d2", tags=["t2", "shared"])
        cache.set("k3", "d3", tags=["t3"])
        
        cache.invalidate_tag("shared")
        self.assertIsNone(cache.get("k1"))
        self.assertIsNone(cache.get("k2"))
        self.assertEqual(cache.get("k3"), "d3")

    @patch("app.services.gemini_service.GeminiService._post_with_retry")
    def test_gemini_re_ranking(self, mock_post):
        gemini_service.api_key = "dummy_key"
        import httpx
        res = httpx.Response(
            status_code=200,
            json={
                "candidates": [{
                    "content": {
                        "parts": [{"text": '[{"id": "chunk1", "score": 0.9}, {"id": "chunk2", "score": 0.2}]'}]
                    }
                }]
            }
        )
        res.request = httpx.Request("POST", "http://test")
        mock_post.return_value = res
        
        scores = asyncio.run(
            gemini_service.re_rank_chunks(
                query="test query",
                chunks=[{"id": "chunk1", "content": "text1"}, {"id": "chunk2", "content": "text2"}]
            )
        )
        self.assertEqual(scores.get("chunk1"), 0.9)
        self.assertEqual(scores.get("chunk2"), 0.2)

if __name__ == "__main__":
    unittest.main()
