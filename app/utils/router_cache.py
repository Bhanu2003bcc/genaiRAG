import time
from typing import Dict, Any, Optional, Set, List
from app.utils.metrics import metrics_collector

class RouterCache:
    def __init__(self, maxsize: int = 500, default_ttl: int = 120):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        # Key: cache_key -> Value: (expire_time, response_data)
        self.cache: Dict[str, tuple[float, Any]] = {}
        # Tag: tag_name -> Set of cache_keys
        self.tags: Dict[str, Set[str]] = {}

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            metrics_collector.record_router_cache(hit=False)
            return None
        
        expire_time, data = self.cache[key]
        if time.time() > expire_time:
            self.invalidate_key(key)
            metrics_collector.record_router_cache(hit=False)
            return None
        
        metrics_collector.record_router_cache(hit=True)
        return data

    def set(self, key: str, data: Any, tags: List[str], ttl: Optional[int] = None) -> None:
        if len(self.cache) >= self.maxsize:
            # First, clean any expired keys
            expired_keys = [k for k, (exp, _) in self.cache.items() if time.time() > exp]
            if expired_keys:
                for k in expired_keys:
                    self.invalidate_key(k)
            # If still full, evict the oldest key (FIFO approximation since Python dict keeps insertion order)
            if len(self.cache) >= self.maxsize:
                oldest_key = next(iter(self.cache))
                self.invalidate_key(oldest_key)

        cache_ttl = ttl if ttl is not None else self.default_ttl
        expire_time = time.time() + cache_ttl
        self.cache[key] = (expire_time, data)

        for tag in tags:
            if tag not in self.tags:
                self.tags[tag] = set()
            self.tags[tag].add(key)

    def invalidate_tag(self, tag: str) -> None:
        if tag not in self.tags:
            return
        # Create a list copy of keys to avoid modification issues during iteration
        keys_to_remove = list(self.tags[tag])
        for key in keys_to_remove:
            self.invalidate_key(key)
        self.tags.pop(tag, None)

    def invalidate_key(self, key: str) -> None:
        self.cache.pop(key, None)
        # Clean up tags that referenced this key
        for tag in list(self.tags.keys()):
            if key in self.tags[tag]:
                self.tags[tag].remove(key)
                if not self.tags[tag]:
                    self.tags.pop(tag, None)

    def clear(self) -> None:
        self.cache.clear()
        self.tags.clear()

# Global router cache singleton
router_cache = RouterCache()
