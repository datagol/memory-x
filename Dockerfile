# 1) Build the web dashboard
FROM node:22-slim AS web
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0
RUN corepack enable
WORKDIR /src/web
COPY web/package.json web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY web/ ./
RUN pnpm build   # writes /src/memx_api/static

# 2) API + worker image
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml ./
COPY memx ./memx
COPY memx_api ./memx_api
COPY --from=web /src/memx_api/static ./memx_api/static
RUN pip install --no-cache-dir .
RUN useradd --create-home memx
USER memx
EXPOSE 8000
CMD ["uvicorn", "memx_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
