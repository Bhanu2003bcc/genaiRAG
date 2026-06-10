import time
from typing import Dict, Any, List

class MetricsCollector:
    def __init__(self):
        # HTTP Request stats
        # Key: "METHOD PATH STATUS_CODE" -> count
        self.request_count: Dict[str, int] = {}
        # Key: "METHOD PATH STATUS_CODE" -> total_duration_seconds
        self.request_durations: Dict[str, float] = {}
        self.active_requests: int = 0

        # Cache stats
        self.embedding_cache_hits: int = 0
        self.embedding_cache_misses: int = 0
        self.router_cache_hits: int = 0
        self.router_cache_misses: int = 0

        # RAG Pipeline stats
        self.llm_generations: int = 0
        self.llm_generation_time: float = 0.0
        self.rerank_calls: int = 0
        self.rerank_time: float = 0.0

    def record_request(self, method: str, path: str, status_code: int, duration: float):
        key = f"{method} {path} {status_code}"
        self.request_count[key] = self.request_count.get(key, 0) + 1
        self.request_durations[key] = self.request_durations.get(key, 0.0) + duration

    def increment_active_requests(self):
        self.active_requests += 1

    def decrement_active_requests(self):
        self.active_requests = max(0, self.active_requests - 1)

    def record_embedding_cache(self, hit: bool):
        if hit:
            self.embedding_cache_hits += 1
        else:
            self.embedding_cache_misses += 1

    def record_router_cache(self, hit: bool):
        if hit:
            self.router_cache_hits += 1
        else:
            self.router_cache_misses += 1

    def record_llm_generation(self, duration: float):
        self.llm_generations += 1
        self.llm_generation_time += duration

    def record_rerank(self, duration: float):
        self.rerank_calls += 1
        self.rerank_time += duration

    def get_db_pool_stats(self) -> Dict[str, int]:
        try:
            from app.database import engine
            if hasattr(engine, "pool"):
                pool = engine.pool
                return {
                    "size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow() if hasattr(pool, "overflow") else 0
                }
        except Exception:
            pass
        return {"size": 0, "checked_in": 0, "checked_out": 0, "overflow": 0}

    def get_stats(self) -> Dict[str, Any]:
        total_requests = sum(self.request_count.values())
        avg_durations = {
            k: (self.request_durations[k] / self.request_count[k]) if self.request_count[k] > 0 else 0.0
            for k in self.request_count
        }

        total_emb = self.embedding_cache_hits + self.embedding_cache_misses
        emb_hit_rate = round(self.embedding_cache_hits / total_emb, 4) if total_emb > 0 else 0.0

        total_router = self.router_cache_hits + self.router_cache_misses
        router_hit_rate = round(self.router_cache_hits / total_router, 4) if total_router > 0 else 0.0

        return {
            "http": {
                "active_requests": self.active_requests,
                "total_requests": total_requests,
                "requests_by_endpoint": [
                    {
                        "endpoint": k,
                        "count": v,
                        "avg_duration_sec": round(avg_durations[k], 4)
                    }
                    for k, v in self.request_count.items()
                ]
            },
            "caches": {
                "embedding_cache": {
                    "hits": self.embedding_cache_hits,
                    "misses": self.embedding_cache_misses,
                    "hit_rate": emb_hit_rate
                },
                "router_cache": {
                    "hits": self.router_cache_hits,
                    "misses": self.router_cache_misses,
                    "hit_rate": router_hit_rate
                }
            },
            "rag": {
                "llm_generations": self.llm_generations,
                "avg_llm_generation_time_sec": round(self.llm_generation_time / max(1, self.llm_generations), 4),
                "rerank_calls": self.rerank_calls,
                "avg_rerank_time_sec": round(self.rerank_time / max(1, self.rerank_calls), 4)
            },
            "database_pool": self.get_db_pool_stats()
        }

    def get_prometheus_metrics(self) -> str:
        lines = []

        # Active requests
        lines.append("# HELP http_active_requests The number of currently active HTTP requests.")
        lines.append("# TYPE http_active_requests gauge")
        lines.append(f"http_active_requests {self.active_requests}")

        # Total HTTP requests count and duration
        lines.append("# HELP http_requests_total Total number of HTTP requests.")
        lines.append("# TYPE http_requests_total counter")
        for k, count in self.request_count.items():
            parts = k.split(" ", 2)
            method, path, status = parts[0], parts[1], parts[2]
            lines.append(f'http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}')

        lines.append("# HELP http_request_duration_seconds_sum Total HTTP request latency in seconds.")
        lines.append("# TYPE http_request_duration_seconds_sum counter")
        for k, duration in self.request_durations.items():
            parts = k.split(" ", 2)
            method, path, status = parts[0], parts[1], parts[2]
            lines.append(f'http_request_duration_seconds_sum{{method="{method}",path="{path}",status="{status}"}} {duration}')

        # Caches
        lines.append("# HELP cache_hits_total Total number of cache hits.")
        lines.append("# TYPE cache_hits_total counter")
        lines.append(f'cache_hits_total{{cache="embedding"}} {self.embedding_cache_hits}')
        lines.append(f'cache_hits_total{{cache="router"}} {self.router_cache_hits}')

        lines.append("# HELP cache_misses_total Total number of cache misses.")
        lines.append("# TYPE cache_misses_total counter")
        lines.append(f'cache_misses_total{{cache="embedding"}} {self.embedding_cache_misses}')
        lines.append(f'cache_misses_total{{cache="router"}} {self.router_cache_misses}')

        # RAG pipeline
        lines.append("# HELP rag_llm_generations_total Total number of LLM response generations.")
        lines.append("# TYPE rag_llm_generations_total counter")
        lines.append(f"rag_llm_generations_total {self.llm_generations}")

        lines.append("# HELP rag_llm_generation_duration_seconds_sum Cumulative LLM generation time in seconds.")
        lines.append("# TYPE rag_llm_generation_duration_seconds_sum counter")
        lines.append(f"rag_llm_generation_duration_seconds_sum {self.llm_generation_time}")

        lines.append("# HELP rag_rerank_calls_total Total number of LLM chunk re-ranking calls.")
        lines.append("# TYPE rag_rerank_calls_total counter")
        lines.append(f"rag_rerank_calls_total {self.rerank_calls}")

        lines.append("# HELP rag_rerank_duration_seconds_sum Cumulative LLM re-ranking duration in seconds.")
        lines.append("# TYPE rag_rerank_duration_seconds_sum counter")
        lines.append(f"rag_rerank_duration_seconds_sum {self.rerank_time}")

        # DB pool
        stats = self.get_db_pool_stats()
        lines.append("# HELP db_pool_size Total database connection pool size.")
        lines.append("# TYPE db_pool_size gauge")
        lines.append(f"db_pool_size {stats['size']}")

        lines.append("# HELP db_pool_connections_checked_in Database connections checked in (idle).")
        lines.append("# TYPE db_pool_connections_checked_in gauge")
        lines.append(f"db_pool_connections_checked_in {stats['checked_in']}")

        lines.append("# HELP db_pool_connections_checked_out Database connections checked out (active).")
        lines.append("# TYPE db_pool_connections_checked_out gauge")
        lines.append(f"db_pool_connections_checked_out {stats['checked_out']}")

        lines.append("# HELP db_pool_overflow Database connections in overflow status.")
        lines.append("# TYPE db_pool_overflow gauge")
        lines.append(f"db_pool_overflow {stats['overflow']}")

        return "\n".join(lines) + "\n"

# Global collector singleton
metrics_collector = MetricsCollector()
