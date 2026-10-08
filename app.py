from pathlib import Path

import streamlit as st

from src import config
from src.rag import RAG

st.set_page_config(page_title="Sales Call Intelligence", layout="wide")

EXAMPLE_QUESTIONS = [
    "What are the most common reasons prospects hesitate to buy?",
    "Which customers mentioned integrations with Salesforce?",
    "What concerns did customers raise about pricing?",
    "Which competitors were mentioned, and in what context?",
    "What did Acme Corp say about their current workflow?",
    "Why did customers who left decide to leave?",
    "Which cold calls produced a follow-up?",
    "Has anyone from the vendor given inconsistent answers about the product?",
]


@st.cache_resource(show_spinner="Loading models (first run only)...")
def get_rag() -> RAG:
    return RAG()


def render_sources(documents: list) -> None:
    with st.expander(f"Retrieved excerpts ({len(documents)})"):
        for i, doc in enumerate(documents, start=1):
            st.markdown(f"**{i}. {doc.metadata.get('file_name', 'unknown')}**")
            st.text(doc.page_content)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("documents"):
            render_sources(message["documents"])


st.title("Sales Call Intelligence")
st.caption(
    "Ask questions about the sales call transcripts. Answers cite the call file and timestamp; "
    "the retrieved excerpts are shown under each answer so you can verify them. "
    "Each question is answered independently (no conversation memory)."
)

if not Path(config.QDRANT_PATH).exists():
    st.error("No vector index found. Build it first by running `python -m scripts.ingest` from the project root.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Example questions")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, use_container_width=True):
            st.session_state.pending_question = example
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    render_message(message)

question = st.chat_input("Ask a question about the calls...") or st.session_state.pop("pending_question", None)

if question:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the calls..."):
                result = get_rag().rag_chain(question)
        except Exception as exc:
            st.error(f"Something went wrong while answering: {exc}")
        else:
            st.markdown(result["answer"])
            render_sources(result["documents"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "documents": result["documents"]}
            )


