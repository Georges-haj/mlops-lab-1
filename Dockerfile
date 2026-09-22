# ---- Stage 1: builder ---------------------------------------------------
# Has uv + full build tooling. Nothing here ends up in the final image.
FROM python:3.12-slim AS builder

# uv's official static binary - avoids needing pip/pipx inside the image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy ONLY the dependency manifests first. As long as these two files don't
# change, Docker reuses the cached result of `uv sync` on every rebuild -
# editing serve.py below won't invalidate this (expensive) layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Now bring in the actual source code and install the project itself.
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# ---- Stage 2: runtime -----------------------------------------------------
# Clean slim base - none of the builder stage's build tools/cache come along.
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy just the finished virtual environment and the source code, nothing else.
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
