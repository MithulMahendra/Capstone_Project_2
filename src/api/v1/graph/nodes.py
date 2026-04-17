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
_SQL_DB_NAME = os.getenv("SQL_DB_NAME", "default_db")


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

        Your task is to classify the user query into exactly ONE of these categories:
        - sql
        - document
        - out_of_scope

        Definitions:

        1) sql
        Choose sql ONLY when the user is asking for data that must come from structured records, transaction tables, account tables, employee tables, or operational databases.

        Typical sql queries involve:
        - listing records or transactions
        - filtering rows by account, customer, date, status, department, employee, etc.
        - counts, totals, averages, trends, comparisons, or metrics computed from live or stored structured data
        - customer-specific or account-specific data retrieval
        - examples: “show”, “list”, “get records”, “count”, “sum”, “average”, “top 10”, “transactions for account X”, “spend last month”, “customers with overdue payment”

        Important:
        If the user asks for actual transaction history, actual account data, or computed analytics from stored records, choose sql.

        2) document
        Choose document when the answer should come from enterprise documents such as policy manuals, product guides, SOPs, FAQs, HR handbooks, claims rules, brochures, knowledge articles, or training documents.

        This includes:
        - policy/rule lookup
        - definitions and explanations
        - eligibility criteria
        - fee structures
        - product features
        - billing rules
        - rewards rules
        - spend category definitions
        - threshold/limit explanations
        - process guidance
        - conceptual or procedural questions
        - tabular lookups from documents
        - comparisons between product variants described in documents
        - ambiguous or mixed enterprise questions

        Important:
        Choose document even if the answer contains numbers, limits, percentages, slabs, dates, fees, rewards, or comparisons — as long as the answer is stated in a document rather than requiring database retrieval.

        3) out_of_scope
        Choose out_of_scope ONLY if the query is not related to the enterprise domain or cannot reasonably be answered using enterprise documents or enterprise databases.

        Examples:
        - jokes, chit-chat, casual conversation
        - politics, movies, sports, celebrities, general trivia
        - personal advice unrelated to enterprise context
        - random non-business questions

        Critical rules:
        1. If the answer can be found in a policy, guide, manual, or FAQ → choose document.
        2. If the answer requires actual records, rows, transactions, balances, or analytics from structured data → choose sql.
        3. If unsure between sql and document → choose document.
        4. Do NOT choose sql just because the query mentions numbers, fees, percentages, categories, thresholds, or comparisons.
        5. Respond with EXACTLY one word:
        sql
        document
        out_of_scope

        Examples:
        - “What is the annual fee for NorthStar Platinum?” -> document
        - “Compare Signature vs Platinum lounge access” -> document
        - “What is the reward rate for international spends on Signature?” -> document
        - “Explain how billing cycle and due date work” -> document
        - “Show all transactions above Rs. 50,000 for account CC-881001 in March 2026” -> sql
        - “What is the total spend by category for account CC-881001 this month?” -> sql
        - “How many customers missed payment in the last 30 days?” -> sql
        - “Who won yesterday’s IPL match?” -> out_of_scope

        User query:
        {query}
    """

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=_INTENT_PROMPT.format(query=query))])

    route = _safe_text(response.content).strip().lower()

    if route not in {"sql", "document", "out_of_scope"}:
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

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    )

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
            top_n=min(5, len(documents)),
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


# Check Relevance
def check_relevance(state: GraphState) -> str:
    chunks = state.get("reranked_chunks", []) or []
    iteration = state["iteration"]
    max_iter = state["max_iterations"]

    if chunks:
        return "summarize"

    if iteration < max_iter:
        return "rephrase"

    return "no_answer"


# Summarize Answer Node
def summarize_answer(state: GraphState) -> GraphState:
    query = state["query"]
    chunks = state.get("reranked_chunks", []) or []

    if not chunks:
        print("[summarize] No reranked chunks available")
        return {
            **state,
            "answer_found": False,
            "answer": "The provided documents do not contain enough information to answer this question.",
            "policy_citations": "",
        }

    context_blocks: List[str] = []
    for i, c in enumerate(chunks[:8], start=1):
        block = f"[Chunk {i}]\n{_chunk_to_searchable_text(c)}"
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    structured_prompt = """\
        You are a grounded enterprise document assistant.

        Answer the user's question using ONLY the provided context.

        Rules:
        1. Use only the provided context.
        2. Answers may appear in normal text OR inside tables.
        3. If the answer is in a table, identify the correct row/entity and extract the exact value.
        4. For product/card questions, match the exact product name exactly if present.
        5. If the context contains the answer in a table, DO NOT say information is missing.
        6. If the context truly does not contain enough information:
        - answer_found = false
        - answer = "The provided documents do not contain enough information to answer this question."
        7. Do not hallucinate.
        8. Do not mention document names, file names, page numbers, chunk numbers, or source references in the answer.
        9. Be concise, exact, and professional.
        10. If a numeric fee/price/value is present, return the exact value as stated.

        User question:
        {query}

        Context:
        {context}
    """

    llm = get_llm()

    answer_found = False
    final_answer = "The provided documents do not contain enough information to answer this question."

    try:
        structured_llm = llm.with_structured_output(SummaryResult)
        result = structured_llm.invoke(
            structured_prompt.format(query=query, context=context)
        )

        print(f"[summarize] structured_result={result}")

        if isinstance(result, SummaryResult):
            answer_found = bool(result.answer_found)
            final_answer = result.answer.strip() or final_answer
        elif isinstance(result, dict):
            answer_found = bool(result.get("answer_found", False))
            final_answer = str(result.get("answer", "")).strip() or final_answer
        else:
            print("[summarize] Unexpected structured result type, falling back...")
            raise ValueError("Unexpected structured output type")

    except Exception as exc:
        print(f"[summarize] Structured output failed: {exc}")

        fallback_prompt = """\
            You are a grounded enterprise document assistant.

            Answer the user's question using ONLY the provided context.

            Return valid JSON with exactly this schema:
            {
            "answer_found": true,
            "answer": "string"
            }

            Rules:
            1. Use only the provided context.
            2. Answers may appear in normal text OR inside tables.
            3. If the answer is in a table, identify the correct row/entity and extract the exact value.
            4. For product/card questions, match the exact product name if present.
            5. If the context contains the answer in a table, DO NOT say information is missing.
            6. If the context truly does not contain enough information, return:
            {
            "answer_found": false,
            "answer": "The provided documents do not contain enough information to answer this question."
            }
            7. Do not hallucinate.
            8. Do not mention document names, file names, page numbers, chunk numbers, or source references in the answer.
            9. Be concise, exact, and professional.
            10. If a numeric fee/price/value is present, return the exact value as stated.
            11. Return ONLY JSON.

            User question:
            {query}

            Context:
            {context}
        """

        response = llm.invoke(
            [HumanMessage(content=fallback_prompt.format(query=query, context=context))]
        )

        raw_text = _safe_text(response.content)
        print(f"[summarize] raw_output_repr={raw_text!r}")

        parsed = _extract_json_object(raw_text)
        print(f"[summarize] parsed={parsed}")

        if parsed:
            raw_answer_found = parsed.get("answer_found", False)
            if isinstance(raw_answer_found, str):
                answer_found = raw_answer_found.strip().lower() == "true"
            else:
                answer_found = bool(raw_answer_found)

            final_answer = str(parsed.get("answer", "")).strip() or final_answer
        else:
            # Very last fallback heuristic
            if '"answer_found": true' in raw_text.lower():
                answer_found = True

                answer_match = re.search(
                    r'"answer"\s*:\s*"([^"]+)"', raw_text, re.DOTALL
                )
                if answer_match:
                    final_answer = answer_match.group(1).strip()

    print(f"[summarize] final answer_found={answer_found}, answer={final_answer!r}")

    seen: List[str] = []
    for c in chunks:
        document_name = str(c.get("document_name", "")).strip()
        if document_name and document_name not in seen:
            seen.append(document_name)

    return {
        **state,
        "answer_found": answer_found,
        "answer": final_answer,
        "policy_citations": ", ".join(seen),
    }


def route_after_summary(state: GraphState) -> str:
    print(
        f"[route_after_summary] answer_found={state.get('answer_found')} "
        f"iteration={state.get('iteration')} "
        f"answer={state.get('answer')!r}"
    )

    if state.get("answer_found", False):
        return "end"

    if state["iteration"] < state["max_iterations"]:
        return "rephrase"

    return "no_answer"


# NL2SQL Node
def nl2sql_node(state: GraphState) -> GraphState:
    query = _current_query(state)
    llm = get_llm()
    db = get_sql_database()

    schema_info = db.get_table_info()
    MAX_SQL_ROWS = 50
    _NL2SQL_SYSTEM = f"""\
        You are an expert PostgreSQL query generator for a NorthStar Bank credit-card analytics database.

        Convert the user's question into exactly ONE safe SQL query.

        Return ONLY raw SQL.
        No explanation.
        No markdown.
        No backticks.
        No comments.

        RULES
        1. Generate exactly one read-only query.
        2. Allowed: SELECT or WITH ... SELECT only.
        3. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, MERGE, GRANT, or REVOKE.
        4. Use ONLY tables and columns from the schema.
        5. Never invent columns or tables.
        6. Prefer explicit column names instead of SELECT *.
        7. For non-aggregate queries, include LIMIT {MAX_SQL_ROWS}.
        8. Use PostgreSQL syntax.

        DATABASE SCHEMA
        {schema_info}

        TABLE USAGE
        - customers: customer identity
        - credit_cards: card-level snapshot data (limits, outstanding_amt, min_due, reward_points, due_date, statement_date)
        - card_transactions: transaction-level analysis (txn_date, txn_type, amount, merchant_name, category_name, is_international, reward_pts_earned)
        - reward_transactions: reward ledger over time
        - billing_statements: statement-level monthly summaries

        IMPORTANT DOMAIN LOGIC
        - purchase = spend
        - payment = repayment, not spend
        - refund = negative reversal already stored as negative amount
        - fee = charged fee
        - emi_instalment = EMI installment charge
        - total spend usually means SUM(amount) over purchase transactions
        - net spend means SUM(amount) including refunds if relevant
        - do not subtract refunds twice
        - use billing_statements for statement month summaries
        - use card_transactions for merchant/category/transaction detail
        - use reward_transactions for reward history
        - use credit_cards.reward_points for current reward balance snapshot
        - use billing_statements.reward_pts_earned for statement-cycle earned points

        COLUMN MAPPING
        - customer name -> customers.full_name
        - card -> credit_cards.card_id
        - card variant -> credit_cards.card_variant
        - transaction date -> card_transactions.txn_date
        - posting date -> card_transactions.posting_date
        - category -> card_transactions.category_name
        - merchant -> card_transactions.merchant_name
        - reward points earned -> card_transactions.reward_pts_earned or billing_statements.reward_pts_earned depending on question
        - reward balance -> credit_cards.reward_points
        - outstanding balance -> credit_cards.outstanding_amt
        - available limit -> credit_cards.available_limit
        - minimum due -> credit_cards.min_due or billing_statements.min_amount_due depending on question
        - statement month -> billing_statements.billing_month
        - statement due date -> billing_statements.due_date

        SPECIAL DATASET MAPPING
        - "primary card" or "primary account" means card_id = 'CC-881001'

        TIME RULES
        - If user asks for a statement month like "March 2026 statement" or "March 2026 billing cycle", prefer billing_statements.billing_month = '2026-03'
        - If user asks for transaction details in a billing cycle, use the correct transaction date range or derive it from billing_statements
        - If user asks for recent/latest/current, infer from the latest dates available in the dataset

        MATCHING RULES
        - Use ILIKE for customer names, merchant names, category names, and card variant names when text matching is needed
        - Avoid overly strict exact equality on merchant names unless user supplied the full exact name
        - For merchant family matching such as Amazon, use ILIKE '%amazon%'

        QUERY STRATEGY
        - Determine whether the answer is card snapshot, transaction detail, reward ledger, or statement summary
        - Use joins only when needed
        - Use aggregation for totals/counts/averages/comparisons
        - Use CTEs when useful for complex analytics
        - Use NULLIF for safe division

        FAILURE AVOIDANCE
        Do NOT use nonexistent columns such as:
        - transaction_date
        - reward_points_earned
        - statement_month
        - customer_name
        - transaction_category
        - available_credit

        Use these instead:
        - txn_date
        - reward_pts_earned
        - billing_month
        - full_name
        - category_name
        - available_limit

        Return ONLY SQL.
    """

    sql_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _NL2SQL_SYSTEM),
            ("human", "User question: {question}"),
        ]
    )

    raw_sql_response = (sql_prompt | llm).invoke(
        {
            "schema": schema_info,
            "question": query,
        }
    )

    sql_query = _clean_sql(_safe_text(raw_sql_response.content))
    print(f"[nl2sql] Generated SQL:\n{sql_query}")

    if not _is_safe_select_query(sql_query):
        print("[nl2sql] Unsafe or invalid SQL generated")
        return {
            **state,
            "sql_query_executed": sql_query,
            "sql_result": "SQL execution error: generated query was unsafe or invalid.",
            "answer": "",
            "database_name": _SQL_DB_NAME,
            "sql_success": False,
        }

    try:
        sql_result = db.run(sql_query)
        sql_result = str(sql_result)
    except Exception as exc:
        sql_result = f"SQL execution error: {exc}"

    print(f"[nl2sql] Result (truncated): {sql_result[:300]}")

    has_data = _looks_like_has_data(sql_result)

    answer = ""
    if has_data:
        _SQL_ANSWER_SYSTEM = """\
            You are a precise financial data analyst.

            Answer the user's question using ONLY the SQL result provided.

            Rules:
            1. Use only the SQL result.
            2. Be accurate, concise, and direct.
            3. If the result contains multiple rows, summarize the relevant insight clearly.
            4. If the question asks for comparison, ranking, top category, top merchant, or trend, interpret the rows correctly.
            5. If the result is empty, say: No relevant data was found for this request.
            6. If the result only partially answers the question, answer only what is supported.
            7. Do not hallucinate.
            8. Do not mention SQL unless necessary.
        """

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SQL_ANSWER_SYSTEM),
                (
                    "human",
                    "User question:\n{query}\n\nSQL query:\n{sql}\n\nSQL result:\n{result}",
                ),
            ]
        )

        ans_response = (answer_prompt | llm).invoke(
            {
                "query": state["query"],
                "sql": sql_query,
                "result": sql_result,
            }
        )

        answer = _safe_text(ans_response.content)
        print("[nl2sql] Answer generated successfully")

    return {
        **state,
        "sql_query_executed": sql_query,
        "sql_result": sql_result,
        "answer": answer,
        "database_name": _SQL_DB_NAME,
        "sql_success": has_data,
    }


# Check SQL Result Edge
def check_sql_result(state: GraphState) -> str:
    result = state.get("sql_result", "")
    has_data = _looks_like_has_data(result)

    if has_data:
        return "end"

    if state["iteration"] < state["max_iterations"]:
        return "rephrase"

    return "no_answer"


# Rephrase Query Node
def rephrase_query(state: GraphState) -> GraphState:
    query = state["query"]
    failed_query = _current_query(state)
    route = state.get("route", "document")
    iteration = state["iteration"]

    _REPHRASE_PROMPT = """\
        You are an expert query reformulation assistant for a hybrid SQL/document retrieval system.

        The previous attempt failed or returned no useful answer.
        Rewrite the query so the next attempt has a better chance of succeeding.

        Return ONLY the rewritten query text.
        No explanations.
        No bullets.
        No quotes.

        Rules:
        1. Preserve the original user intent exactly.
        2. Do not repeat the failed query wording exactly.
        3. Make the next query clearer, more explicit, and easier to retrieve or convert.
        4. Keep it concise.

        If route is "sql", improve the query by:
        - making the entity explicit (customer, card, card ID, statement, transactions, rewards, merchant, category)
        - making the metric explicit (total spend, net spend, count, fees, refunds, points earned, reward balance, closing balance, minimum due)
        - making the time scope explicit (month, billing cycle, latest, recent, date range)
        - clarifying whether the user wants transaction-level detail, statement-level summary, or card snapshot data
        - broadening overly strict wording for names/merchants/categories
        - clarifying whether spend should include refunds, fees, or only purchases
        - clarifying whether reward refers to earned, redeemed, expired, or current balance

        If route is "document", improve the query by:
        - emphasizing product name, policy wording, benefit name, fee-sheet wording, headings, and synonyms

        Original query:
        {query}

        Last failed query:
        {failed_query}

        Route:
        {route}

        Attempt number:
        {iteration}

        Rewritten query:
    """

    llm = get_llm()
    response = llm.invoke(
        [
            HumanMessage(
                content=_REPHRASE_PROMPT.format(
                    query=query,
                    failed_query=failed_query,
                    route=route,
                    iteration=iteration,
                )
            )
        ]
    )

    rephrased = _safe_text(response.content)

    if not rephrased:
        rephrased = query

    print(f"[rephrase] iteration={iteration} -> '{rephrased}'")

    return {
        **state,
        "rephrased_query": rephrased,
        "iteration": iteration + 1,
    }


def route_after_rephrase(state: GraphState) -> str:
    return state["route"]


# No Answer Node
def no_answer_node(state: GraphState) -> GraphState:
    route = state.get("route", "")
    search_type = state.get("search_type", "")
    sql_query = state.get("sql_query_executed", "N/A")

    return {
        **state,
        "answer": (
            f"I could not find a reliable answer after {state['iteration']} attempt(s). "
            f"Please try rephrasing your question more specifically."
        ),
        "policy_citations": "",
        "sql_query_executed": sql_query if route == "sql" else "N/A",
        "search_type": search_type,
        "answer_found": False,
    }
