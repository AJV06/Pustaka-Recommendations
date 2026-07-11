from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.recommend import router

app = FastAPI(
    title="Pustaka Recommendation API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Pustaka Recommendation API",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

app.include_router(router)
