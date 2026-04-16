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

# Intent route
def intent_router_node(state: GraphState) -> GraphState:
    query = state["query"]

    _INTENT_PROMPT = """\
        You are a query intent classifier for an enterprise assistant.

        Classify the user query into exactly one category:
        - sql
        - document

        Choose sql ONLY if the user is clearly asking for structured database output such as:
        - records
        - lists
        - counts
        - totals
        - averages
        - comparisons
        - metrics
        - rows filtered by date, department, employee, etc.

        Choose document for:
        - policies
        - FAQs
        - rules
        - definitions
        - explanations
        - conceptual questions
        - product information from documents/brochures/manuals
        - benefits / leave / claims guidance
        - ambiguous or mixed queries
        - any question that does not clearly require database rows or calculations

        Important rules:
        1. If unsure, choose document.
        2. Product/card/brochure/manual/fee-sheet style questions are document unless the user explicitly asks for database output.
        3. Reply with exactly one word: sql OR document

        Examples:
        Q: How many employees joined last month? -> sql
        Q: Show salary of employee 1042 -> sql
        Q: Explain maternity leave policy -> document
        Q: What does payslip deduction mean? -> document
        Q: What is the annual fee for NorthStar Platinum card? -> document

        User query:
        {query}
    """

    llm = get_llm()
    response = llm.invoke([
        HumanMessage(content=_INTENT_PROMPT.format(query=query))
    ])

    route = _safe_text(response.content).lower()
    if route not in {"sql", "document"}:
        route = "document"

    print(f"[intent_router] route='{route}'")

    return {
        **state,
        "route": route,
    }


def route_after_router(state: GraphState) -> str:
    return state["route"]

# Agent Retrieve Node
def agent_retrieve(state: GraphState) -> GraphState:
    query = _current_query(state)

    _AGENT_SYSTEM_PROMPT = """\
        You are a retrieval-routing agent for a document knowledge base.

        Your job is ONLY to choose the best retrieval tool and call exactly ONE tool.

        Available tools:
        1. fts_search
        - best for exact terms, product names, policy codes, IDs, abbreviations, names, short literal phrases
        2. vector_search
        - best for conceptual, explanatory, natural-language questions
        3. hybrid_search
        - best for short queries, product queries, ambiguous phrasing, or mixed lexical + semantic intent

        Rules:
        1. You must call exactly one tool.
        2. Do NOT answer the question.
        3. Do NOT add conversational text.
        4. Prefer:
        - fts_search for exact codes / IDs / proper nouns / exact product names
        - vector_search for explanatory questions
        - hybrid_search for short, product-centric, fee/table, or ambiguous queries
        5. If uncertain, choose hybrid_search.
        """

    llm = get_llm()
    llm_with_tools = llm.bind_tools(_TOOLS, tool_choice="any")

    response = llm_with_tools.invoke([
        SystemMessage(content=_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ])

    if not getattr(response, "tool_calls", None):
        print("[agent_retrieve] No tool call -> fallback hybrid_search")
        chunks = _hybrid_search(query=query, k=10)
        return {
            **state,
            "raw_chunks": chunks,
            "search_type": "hybrid_fallback_no_tool_call",
        }

    tool_call = response.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call.get("args", {}) or {}

    if "query" not in tool_args or not str(tool_args["query"]).strip():
        tool_args["query"] = query
    if "k" not in tool_args:
        tool_args["k"] = 10

    print(f"[agent_retrieve] selected_tool='{tool_name}' args={tool_args}")

    selected_tool = _TOOL_MAP.get(tool_name)
    if selected_tool is None:
        print(f"[agent_retrieve] Unknown tool '{tool_name}' -> fallback hybrid_search")
        chunks = _hybrid_search(query=query, k=10)
        return {
            **state,
            "raw_chunks": chunks,
            "search_type": "hybrid_fallback_unknown_tool",
        }

    try:
        chunks = selected_tool.invoke(tool_args)
    except Exception as exc:
        print(f"[agent_retrieve] Tool error: {exc} -> fallback hybrid_search")
        chunks = _hybrid_search(query=query, k=10)
        return {
            **state,
            "raw_chunks": chunks,
            "search_type": "hybrid_fallback_tool_error",
        }

    if not chunks and tool_name != "hybrid_search":
        print("[agent_retrieve] Empty retrieval -> fallback hybrid_search")
        chunks = _hybrid_search(query=query, k=10)
        return {
            **state,
            "raw_chunks": chunks,
            "search_type": "hybrid_fallback_empty",
        }

    print(f"[agent_retrieve] retrieved={len(chunks)} chunks")
    return {
        **state,
        "raw_chunks": chunks,
        "search_type": tool_name,
    }

# Rerank Node
def rerank(state: GraphState) -> GraphState:
    query = _current_query(state)
    chunks = state.get("raw_chunks", []) or []

    if not chunks:
        print("[rerank] No raw chunks available")
        return {
            **state,
            "reranked_chunks": [],
            "retrieval_status": "no_chunks",
        }

    documents = [_chunk_to_searchable_text(c) for c in chunks]

    try:
        co = cohere.Client(_COHERE_API_KEY)
        results = co.rerank(
            query=query,
            documents=documents,
            top_n=min(8, len(documents)),
            model=_RERANK_MODEL,
        )

        reranked = [
            {
                **chunks[r.index],
                "relevance_score": round(float(r.relevance_score), 4),
            }
            for r in results.results
        ]
    except Exception as exc:
        print(f"[rerank] Cohere rerank failed: {exc}")
        reranked = [{**c, "relevance_score": 0.0} for c in chunks[:8]]

    print(f"[rerank] reranked_count={len(reranked)}")

    for idx, c in enumerate(reranked[:5], start=1):
        print(
            f"[rerank] top{idx} chunk_type={c.get('chunk_type')} "
            f"doc={c.get('document_name')} "
            f"score={c.get('relevance_score')}"
        )

    return {
        **state,
        "reranked_chunks": reranked,
        "retrieval_status": "ok" if reranked else "no_chunks",
    }
