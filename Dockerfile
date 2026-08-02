# Multi-stage: copy uv binary from the official image
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked

# Copy application code and data
COPY . .

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

CMD ["streamlit", "run", "src/metascholar/app/app.py"]
