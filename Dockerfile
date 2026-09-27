# --- Frontend build stage (React + TypeScript + Vite) ---------------------
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build   # outputs to /web/dist (see vite.config.ts outDir)

# --- Runtime stage ---------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY data ./data
COPY --from=frontend /web/dist ./web/dist
RUN adduser --disabled-password --uid 10001 app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
