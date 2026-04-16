from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.core.database import init_db
from src.api.v1.routes.admin import router as admin_router
from src.api.v1.routes.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="LangGraph RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(admin_router, prefix="/api/v1/admin")
app.include_router(query_router, prefix="/api/v1")

