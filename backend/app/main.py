import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.db import Base, engine
from app.routers import transactions, retries, dashboard, insights
from app.seed import seed_if_empty

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Reflow - Payment Recovery Engine",
    description="Decides how (and whether) to retry failed Razorpay payments.",
    version="0.1.0",
)

# demo/dev defaults to wide open so the frontend just works out of the box;
# set CORS_ALLOW_ORIGINS to a comma-separated list before deploying this anywhere real
_allow_origins = os.environ.get("CORS_ALLOW_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_origins == "*" else [o.strip() for o in _allow_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transactions.router)
app.include_router(retries.router)
app.include_router(dashboard.router)
app.include_router(insights.router)


@app.get("/health")
def health():
    return {"status": "ok"}


# auto-seed demo data on boot if the DB is empty - keeps the deployed demo
# (or a fresh docker volume, or a judge's own clone) self-sufficient instead
# of depending on someone's local Codespace state or a manual seed step.
# No-op if there's already data (real activity or a previous seed).
if os.environ.get("DEMO_SEED_ON_STARTUP", "1") != "0":
    seed_if_empty()


# single-service deployment: if a built frontend is present alongside the
# backend (see the repo-root Dockerfile), serve it at "/" from this same
# FastAPI process so a judge gets one URL with no CORS/mixed-origin setup
# required. Mounted last so it never shadows the API routes above - Starlette
# matches routes in registration order, and this is a catch-all for whatever
# isn't already handled. In local dev (docker-compose, two containers) this
# directory doesn't exist in the backend image, so the mount is skipped and
# nothing changes.
_frontend_dir = os.environ.get("FRONTEND_STATIC_DIR", "/app/frontend_static")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
