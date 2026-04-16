import os
import json
import re
from typing import Any, Dict, List, Optional
import cohere
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from src.api.v1.schemas.query_schema import SummaryResult
from src.core.helper import get_llm, get_sql_database
from src.api.v1.graph.state import GraphState
from src.api.v1.tools.vector_search_tool import vector_search as _vector_search
from src.api.v1.tools.fts_search_tool import fts_search as _fts_search
from src.api.v1.tools.hybrid_search_tool import hybrid_search as _hybrid_search
from src.core.helper import (
    _safe_text,
    _current_query,
    _clean_sql,
    _chunk_to_searchable_text,
    _is_safe_select_query,
    _extract_json_object,
    _looks_like_has_data,
)

load_dotenv(override=True)

_COHERE_API_KEY = os.getenv("COHERE_API_KEY")
_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
_SQL_DB_NAME = os.getenv("SQL_DB_NAME", "agentic_rag_db")
_MAX_SQL_ROWS = int(os.getenv("MAX_SQL_ROWS", "50"))

@tool
def vector_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Semantic vector search against the knowledge base.
    Best for conceptual, explanatory, natural-language questions.
    Example: "What is the maternity leave policy?"
    """
    return _vector_search(query=query, k=k)


@tool
def fts_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Full-text keyword search against the knowledge base.
    Best for exact names, IDs, abbreviations, codes, policy numbers.
    Example: "POL-2024-HR-007"
    """
    return _fts_search(query=query, k=k)


@tool
def hybrid_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Hybrid search combining semantic + lexical retrieval.
    Best for short queries, ambiguous intent, or mixed keyword/concept questions.
    Example: "bonus policy", "salary slip"
    """
    return _hybrid_search(query=query, k=k)


_TOOLS = [vector_search, fts_search, hybrid_search]
_TOOL_MAP = {t.name: t for t in _TOOLS}

