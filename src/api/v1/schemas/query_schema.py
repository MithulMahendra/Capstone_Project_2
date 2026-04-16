from pydantic import BaseModel, Field
from typing import Optional, Union


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Max rephrase-and-retry attempts if no relevant information is found",
    )


class ChunkResult(BaseModel):
    chunk_id: str
    content: str
    source_page: Optional[int] = Field(
        default=None,
        description="Source page number of the chunk"
    )
    document_name: str = Field(
        default="N/A",
        description="Source document name for this chunk"
    )
    chunk_type: str = Field(
        default="text",
        description="Chunk type: text / table / image_caption"
    )
    relevance_score: Optional[float] = Field(
        default=None,
        description="Relevance score assigned during retrieval/reranking"
    )


class DocumentQueryResponse(BaseModel):
    """Response schema for the document retrieval route."""
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Synthesised answer from documents")
    relevant_chunks: list[ChunkResult] = Field(
        default_factory=list,
        description="Top reranked chunks used to generate the answer",
    )
    iterations: int = Field(..., description="Number of retrieval attempts made")
    search_type: str = Field(..., description="Search strategy used (vector / fts / hybrid)")
    policy_citations: str = Field(
        default="",
        description="Inline citations extracted from the answer"
    )


class SQLQueryResponse(BaseModel):
    """Response schema for the SQL / time-bounded data route."""
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Synthesised answer from database results")
    iterations: int = Field(..., description="Number of SQL attempts made")
    database_name: str = Field(default="", description="Database name that was queried")
    sql_query_executed: str = Field(default="", description="The SQL query that was run")


QueryResponse = Union[DocumentQueryResponse, SQLQueryResponse]

class SummaryResult(BaseModel):
    answer_found: bool = Field(
        default=False,
        description="Whether the answer is present in the provided context"
    )
    answer: str = Field(
        default="The provided documents do not contain enough information to answer this question.",
        description="Final grounded answer"
    )

