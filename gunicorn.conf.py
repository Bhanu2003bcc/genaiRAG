"""
Gunicorn production configuration.

Connection budget (CRITICAL — must satisfy for Supabase):
  workers × (DB_POOL_SIZE + DB_MAX_OVERFLOW) ≤ 14
  Default: 4 × (2 + 1) = 12 connections  ✓  (leaves 3 for migrations/admin)

Env overrides:
  GUNICORN_WORKERS  — number of worker processes (default: 4)
  PORT              — bind port (default: 8080)
"""

import os

# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

workers = int(os.environ.get("GUNICORN_WORKERS", 4))
worker_class = "uvicorn.workers.UvicornWorker"

# Each Uvicorn worker is single-threaded with a shared async event loop;
# worker_connections controls the maximum number of simultaneous connections
# in the accept backlog per worker.
worker_connections = 1000

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

# Worker timeout — LLM + embedding calls can be slow; use 120s.
timeout = 120

# Graceful shutdown: allow in-flight requests to finish before SIGKILL.
graceful_timeout = 30

# Keep-alive timeout for idle HTTP/1.1 connections.
keepalive = 5

# ---------------------------------------------------------------------------
# Worker recycling — prevents memory leaks from accumulating
# ---------------------------------------------------------------------------

# Recycle each worker after handling this many requests.
max_requests = 1000

# Add jitter so workers don't all recycle at the same time.
max_requests_jitter = 100

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# "-" routes logs to stdout/stderr (captured by Docker / cloud log agents).
accesslog = "-"
errorlog = "-"
loglevel = "info"

# ---------------------------------------------------------------------------
# Preload
# ---------------------------------------------------------------------------

# Keep False with UvicornWorker — each worker initialises its own event loop
# and lifespan. Preloading with async workers can cause forked loop issues.
preload_app = False
