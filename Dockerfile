# Single-service deployment image: FastAPI serves both the API and the
# static frontend from one process/one origin. This is the image used for
# the deployed demo (see README > Deployment). Local development still
# uses docker-compose.yml (backend + nginx frontend as two containers) -
# this file doesn't change that at all.
FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# model needs to exist at build time - train it if it wasn't checked in
RUN python -m app.ml.train_model

# static frontend, served by app/main.py via StaticFiles when this
# directory is present (see FRONTEND_STATIC_DIR in app/main.py)
COPY frontend/ ./frontend_static/

ENV DATABASE_URL=sqlite:///./data_volume/reflow.db
ENV CORS_ALLOW_ORIGINS=*
ENV FRONTEND_STATIC_DIR=/app/frontend_static

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
