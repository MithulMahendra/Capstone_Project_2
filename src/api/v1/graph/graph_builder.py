from langgraph.graph import StateGraph, END
from src.api.v1.graph.state import GraphState
from langchain_core.runnables.graph import MermaidDrawMethod
from src.api.v1.graph.nodes import (intent_router_node, route_after_router, agent_retrieve, rerank, check_relevance, rephrase_query, route_after_rephrase, summarize_answer,
                                    nl2sql_node, check_sql_result, no_answer_node, route_after_summary)


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", intent_router_node)

    graph.add_node("agent_retrieve", agent_retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("summarize", summarize_answer)

    graph.add_node("nl2sql", nl2sql_node)

    graph.add_node("rephrase", rephrase_query)
    graph.add_node("no_answer", no_answer_node)

    # Entry point
    graph.set_entry_point("router")

    
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "document": "agent_retrieve",
            "sql": "nl2sql",
            "out_of_scope": END,
        },
    )


    # Document route
    graph.add_edge("agent_retrieve", "rerank")

    graph.add_conditional_edges(
        "rerank",
        check_relevance,
        {
            "summarize": "summarize",
            "rephrase": "rephrase",
            "no_answer": "no_answer",
        },
    )

    graph.add_conditional_edges(
        "summarize",
        route_after_summary,
        {
            "end": END,
            "rephrase": "rephrase",
            "no_answer": "no_answer",
        },
    )

    # SQL route
    graph.add_conditional_edges(
        "nl2sql",
        check_sql_result,
        {
            "end": END,
            "rephrase": "rephrase",
            "no_answer": "no_answer",
        },
    )

    # Rephrase loops back to the correct route
    graph.add_conditional_edges(
        "rephrase",
        route_after_rephrase,
        {
            "document": "agent_retrieve",
            "sql": "nl2sql",
        },
    )

    graph.add_edge("no_answer", END)

    return graph.compile()


compiled_graph = build_graph()

graph_image = compiled_graph.get_graph().draw_mermaid_png()

with open(r"src\api\v1\graph\graph_builder.png", "wb") as f:
    f.write(graph_image)