"""
Shared slowapi rate limiter instance.

Import this singleton in:
  - app/main.py       to attach it to app.state and register the error handler
  - app/routers/auth.py  to decorate auth endpoints

Key design notes:
  - key_func=get_remote_address   → limits per client IP (works behind reverse
    proxies that set X-Forwarded-For; configure trusted proxies in production)
  - Storage defaults to in-memory (suitable for single-node).
    For multi-node deployments, swap storage to Redis:
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        limiter = Limiter(key_func=get_remote_address, storage_uri="redis://...")
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
