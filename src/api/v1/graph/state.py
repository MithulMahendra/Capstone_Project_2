from typing import TypedDict, NotRequired, Literal, Any


class ChunkData(TypedDict, total=False):
    chunk_id: str
    content: str
    chunk_type: Literal["text", "table", "image_caption"]
    source_page: int | None
    document_name: str
    metadata: dict[str, Any]
    relevance_score: NotRequired[float]


class GraphState(TypedDict, total=False):
    query: str
    rephrased_query: str
    route: Literal["document", "sql"]
    iteration: int
    max_iterations: int

    # Document retrieval route
    search_type: str
    raw_chunks: list[ChunkData]
    reranked_chunks: list[ChunkData]
    retrieval_status: str

    # SQL route
    sql_result: str
    database_name: str
    sql_query_executed: str
    sql_success: bool

    # Final response
    answer_found: bool
    answer: str
    policy_citations: str