import re
from typing import List

class TextChunker:
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        
        # Clean whitespace (replace multiple whitespaces/newlines with a single space)
        cleaned = re.sub(r"\s+", " ", text).strip()
        words = cleaned.split(" ")
        
        if len(words) <= self.CHUNK_SIZE:
            return [cleaned]
        
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.CHUNK_SIZE, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()
            chunks.append(chunk_text)
            
            if end == len(words):
                break
            
            # Slide window back by overlap
            start = end - self.CHUNK_OVERLAP
            
        return chunks
