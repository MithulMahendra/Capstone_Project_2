import streamlit as st
import requests
from src.api.v1.schemas.query_schema import DocumentQueryResponse, SQLQueryResponse
import time

# Page config
st.set_page_config(
    page_title="Document Q&A",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em {
        color: #e0e0e0 !important;
    }

    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: rgba(255, 255, 255, 0.07) !important;
    }

    .meta-pill {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.1) !important;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #cfcfcf !important;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }

    [data-testid="stExpander"] {
        background-color: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px;
    }

    .streamlit-expanderHeader {
        font-weight: 600;
        color: #ffffff !important;
    }

    div.stButton > button {
        border-radius: 8px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
        border: 1px solid rgba(255, 255, 255, 0.4);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "page" not in st.session_state:
    st.session_state.page = "Chat"

# Sidebar
with st.sidebar:
    st.markdown("### Document AI")
    st.caption(
        "Searches from knowledge base or query from database to generate appropriate answer"
    )

    st.markdown(
        "<hr style='margin: 1em 0; border: none; border-top: 1px solid rgba(255, 255, 255, 0.2);'>",
        unsafe_allow_html=True,
    )

    selected_page = st.radio(
        label="**Navigation**",
        options=["Chat", "Admin"],
        index=0 if st.session_state.page == "Chat" else 1,
        horizontal=True,
    )
    st.session_state.page = selected_page

    st.markdown(
        "<hr style='margin: 1em 0; border: none; border-top: 1px solid rgba(255, 255, 255, 0.2);'>",
        unsafe_allow_html=True,
    )

    if st.session_state.page == "Chat":
        if st.button("＋ New Chat", use_container_width=True, type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown(
            "<hr style='margin: 1em 0; border: none; border-top: 1px solid rgba(255, 255, 255, 0.2);'>",
            unsafe_allow_html=True,
        )

        max_iterations = st.slider(
            "⚙️ **Max Iterations**",
            min_value=1,
            max_value=5,
            value=3,
            help="Max rephrase-and-retry attempts if no relevant info is found.",
        )
    else:
        max_iterations = 3

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Powered by LangGraph & Streamlit")


# Response renderers
def render_document_response(response: DocumentQueryResponse):
    # st.markdown(response.answer)
    def stream_data():
        for word in response.answer.split(" "):
            yield word + " "
            time.sleep(0.02)  # Adjust speed as needed

    st.write_stream(stream_data)

    meta_parts = []
    if response.search_type:
        meta_parts.append(f"<b>Search:</b> {response.search_type}")
    meta_parts.append(f"<b>Iterations:</b> {response.iterations}")
    if response.policy_citations:
        meta_parts.append(f"<b>Citations:</b> {response.policy_citations}")

    meta_html = f"<div class='meta-pill'>{' &nbsp;|&nbsp; '.join(meta_parts)}</div>"
    st.markdown(meta_html, unsafe_allow_html=True)

    if response.relevant_chunks:
        with st.expander(
            f"📄 Source Chunks ({len(response.relevant_chunks)})", expanded=False
        ):
            chunks_json = [
                {
                    "document_name": c.document_name,
                    "chunk_id": c.chunk_id,
                    "source_page": c.source_page,
                    "chunk_type": c.chunk_type,
                    "relevance_score": c.relevance_score,
                    "content": c.content,
                }
                for c in response.relevant_chunks
            ]
            st.json(chunks_json)


def render_sql_response(response: SQLQueryResponse):
    st.markdown(response.answer if response.answer else "_No data returned._")

    meta_parts = [f"<b>Iterations:</b> {response.iterations}"]
    if response.database_name:
        meta_parts.append(f"<b>Database:</b> {response.database_name}")

    meta_html = f"<div class='meta-pill'>{' &nbsp;|&nbsp; '.join(meta_parts)}</div>"
    st.markdown(meta_html, unsafe_allow_html=True)

    if response.sql_query_executed and response.sql_query_executed != "N/A":
        with st.expander("🗄️ SQL Query Executed", expanded=False):
            st.code(response.sql_query_executed, language="sql")


def render_response(response):
    if isinstance(response, DocumentQueryResponse):
        render_document_response(response)
    elif isinstance(response, SQLQueryResponse):
        render_sql_response(response)


# Admin Page
def admin_page():
    st.title("🛠️ Admin Dashboard")
    st.caption("Manage document ingestion into the knowledge base.")
    st.write("")

    with st.container(border=True):
        st.markdown("### 📂 Ingest PDF")
        uploaded_file = st.file_uploader(
            "Upload a PDF to ingest",
            type=["pdf"],
            help="The PDF will be sent to the backend API to be parsed, chunked, and embedded.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            st.info(
                f"**{uploaded_file.name}** selected ({uploaded_file.size / 1024:.1f} KB)"
            )

            if st.button("⬆️ Ingest Document", type="primary"):
                with st.spinner(
                    f"Ingesting **{uploaded_file.name}** — this may take a few minutes…"
                ):
                    try:
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        }
                        api_url = "http://localhost:8000/api/v1/admin/upload"
                        response = requests.post(api_url, files=files)

                        if response.status_code == 200:
                            data = response.json()
                            doc_id = data.get("document_id", "Unknown")
                            st.success(
                                f"✅ Ingested successfully!  \n**Document ID:** `{doc_id}`"
                            )
                        else:
                            st.error(
                                f"❌ Backend API Error: {response.status_code} - {response.text}"
                            )

                    except requests.exceptions.ConnectionError:
                        st.error(
                            "❌ Could not connect to the backend API. Is your FastAPI server running?"
                        )
                    except Exception as e:
                        st.error(f"❌ An error occurred: {e}")


# Chat Page
def chat_page():
    st.title("🔍 Document Q&A")

    if not st.session_state.chat_history:
        st.info("👋 Hello! Start by asking a question about your documents below.")

    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["query"])
        with st.chat_message("assistant"):
            render_response(entry["response"])

    query = st.chat_input("Ask a question…")

    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    api_url = "http://localhost:8000/api/v1/query"
                    payload = {"query": query, "max_iterations": max_iterations}
                    api_response = requests.post(api_url, json=payload)

                    if api_response.status_code == 200:
                        response_data = api_response.json()
                        if response_data.get("relevant_chunks") is not None:
                            response = DocumentQueryResponse(**response_data)
                        else:
                            response = SQLQueryResponse(**response_data)
                        render_response(response)
                        st.session_state.chat_history.append(
                            {
                                "query": query,
                                "response": response,
                            }
                        )
                    else:
                        st.error(
                            f"❌ API Error: {api_response.status_code} - {api_response.text}"
                        )

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Could not connect to the backend API. Is your FastAPI server running?"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")


# Router
if st.session_state.page == "Admin":
    admin_page()
else:
    chat_page()
