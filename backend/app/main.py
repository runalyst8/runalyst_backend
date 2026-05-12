import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.session import db_ping
from app.api.router import api_router as main_router
from app.core.logging_config import setup_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    from app.db.base import Base
    from app.db.session import engine

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Runalyst Backend Refactor is starting up...")

    import app.models  # ensure all models are registered before create_all

    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created/verified")
    print("✓ Application started")

    
    yield
    
    # Shutdown
    print("✓ Application shutting down")


setup_logging()
app = FastAPI(
    title="Runalyst API",
    description="User authentication and authorization API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/documentation",
    redoc_url=None
)

# CORS middleware - configure based on your needs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health():
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.get("/health/db", tags=["health"])
def health_db():
    """Database health check endpoint."""
    db_status = "ok" if db_ping() else "down"
    return {"database": db_status}


# Include routers
app.include_router(main_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Runalyst API"}