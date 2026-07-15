from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from app.routes import router
from app.recommender import engine
from app.utils import setup_logging
import logging

# Setup structured logging
setup_logging()
logger = logging.getLogger("api_logger")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Models
    logger.info("Starting up FastAPI Backend...")
    start_time = time.time()
    try:
        engine.load_all()
        elapsed = time.time() - start_time
        logger.info(f"Startup complete in {elapsed:.2f} seconds.")
    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        # In production, you might choose to sys.exit(1) here if models are required
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI Backend...")

app = FastAPI(
    title="Pustaka Recommender API",
    description="Production-ready recommender backend.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler for unexpected 500s
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected Exception during {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )

# Middleware for timing and logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    formatted_process_time = '{0:.2f}'.format(process_time)
    
    logger.info(f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {formatted_process_time}ms")
    
    # Add timing header (useful for monitoring the <100ms goal)
    response.headers["X-Process-Time-Ms"] = str(formatted_process_time)
    return response

app.include_router(router)
