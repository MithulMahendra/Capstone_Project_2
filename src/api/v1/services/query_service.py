from src.api.v1.graph.graph_builder import compiled_graph
from src.api.v1.schemas.query_schema import ChunkResult, DocumentQueryResponse, SQLQueryResponse


def query_documents(query: str, max_iterations: int = 3) -> DocumentQueryResponse | SQLQueryResponse:
    initial_state = {
        "query": query,
        "rephrased_query": query,
        "route": "",
        "iteration": 1,
        "max_iterations": max_iterations,
        "search_type": "",
        "raw_chunks": [],
        "reranked_chunks": [],
        "sql_result": "",
        "database_name": "",
        "sql_query_executed": "",
        "answer_found": False,
        "answer": "",
        "policy_citations": "",
    }

    result = compiled_graph.invoke(initial_state)

    if result.get("route") == "sql":
        return SQLQueryResponse(
            query=query,
            answer=result.get("answer", ""),
            iterations=result.get("iteration", 1),
            database_name=result.get("database_name", ""),
            sql_query_executed=result.get("sql_query_executed", ""),
        )

    chunks = [
        ChunkResult(
            chunk_id=str(c.get("chunk_id", "")),
            content=str(c.get("content", "")),
            source_page=c.get("source_page"),
            document_name=str(c.get("document_name", "N/A")),
            chunk_type=str(c.get("chunk_type", "text")),
            relevance_score=c.get("relevance_score"),
        )
        for c in result.get("reranked_chunks", [])
    ]

    return DocumentQueryResponse(
        query=query,
        answer=result.get("answer", ""),
        relevant_chunks=chunks,
        iterations=result.get("iteration", 1),
        search_type=result.get("search_type", ""),
        policy_citations=result.get("policy_citations", ""),
    )

