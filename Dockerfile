# syntax=docker/dockerfile:1.7
# Single image: the React build is served by the FastAPI app (one origin, no CORS, one container to run).

# ---- 1. build the frontend ---------------------------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. python runtime -------------------------------------------------------------------------
FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FRONTEND_DIST=/app/static
WORKDIR /app

# Install dependencies first (cached layer), pinned to the versions the test suite ran against.
COPY backend/pyproject.toml backend/constraints.txt backend/README.md ./backend/
RUN mkdir -p backend/src/policyqa && touch backend/src/policyqa/__init__.py \
 && pip install -c backend/constraints.txt "./backend[anthropic,openai]" \
 && pip uninstall -y policyqa

COPY backend/ ./backend/
RUN pip install --no-deps ./backend
COPY data/ ./data/
COPY --from=web /web/dist ./static

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2).status == 200 else 1)"

# init-db is idempotent; the app also retries schema creation and reports "degraded" until the DB is up.
CMD ["sh", "-c", "policyqa init-db && policyqa serve --host 0.0.0.0 --port 8000"]
