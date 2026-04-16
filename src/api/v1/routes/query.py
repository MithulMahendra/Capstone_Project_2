from fastapi import APIRouter, HTTPException
from typing import Union
from src.api.v1.services.query_service import query_documents
from src.api.v1.schemas.query_schema import QueryRequest, DocumentQueryResponse, SQLQueryResponse

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=Union[DocumentQueryResponse, SQLQueryResponse])
def query_endpoint(request: QueryRequest):
    try:
        return query_documents(
            query=request.query,
            max_iterations=request.max_iterations,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
