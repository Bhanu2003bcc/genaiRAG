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
        username = "rag_test_user"
        role = "ADMIN"
        
        token = generate_token(username, role)
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

if __name__ == "__main__":
    unittest.main()
