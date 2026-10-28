FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md alembic.ini ./
COPY services ./services
COPY packages ./packages
COPY migrations ./migrations

RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app/services:/app/packages

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
