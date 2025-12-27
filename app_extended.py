import os
import json
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Optional retrievers / compressors (all open source)
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# Cross-encoder reranking (open source, runs locally)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
QUERY_LOG = LOG_DIR / "queries.jsonl"

st.set_page_config(page_title="📄 PDF RAG Playground (Local)", page_icon="🤖", layout="wide")

st.title("📄 PDF RAG Playground (FAISS + Optional Hybrid/Rerank/Rewrite)")
st.write(
    "Upload a PDF, configure retrieval strategies (hybrid BM25+vector, multi-query rewriting, cross-encoder reranking), "
    "and ask questions with grounded answers + sources."
)

# -----------------------------
# Sidebar: Controls
# -----------------------------
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    st.divider()
    st.header("Chunking")
    chunk_size = st.slider("Chunk size (characters)", min_value=400, max_value=4000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", min_value=0, max_value=800, value=200, step=50)

    st.divider()
    st.header("LLM")
    model_name = st.selectbox("Model", options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.divider()
    st.header("Retrieval")
    vector_search_type = st.selectbox("Vector search type", options=["similarity", "mmr"], index=0)
    vector_k = st.slider("Vector top-k", min_value=2, max_value=40, value=8, step=1)

    use_bm25 = st.checkbox("Enable BM25 keyword retrieval (hybrid)", value=True)
    bm25_k = st.slider("BM25 top-k", min_value=2, max_value=60, value=12, step=1, disabled=not use_bm25)
    bm25_weight = st.slider("BM25 weight", min_value=0.0, max_value=1.0, value=0.45, step=0.05, disabled=not use_bm25)
    vector_weight = 1.0 - bm25_weight if use_bm25 else 1.0

    st.divider()
    st.header("Advanced")
    use_multiquery = st.checkbox("Enable multi-query rewriting (LLM generates alternate queries)", value=False)
    max_query_variants = st.slider("Max rewritten queries", min_value=2, max_value=8, value=4, step=1, disabled=not use_multiquery)

    use_rerank = st.checkbox("Enable cross-encoder reranking (local)", value=False)
    rerank_top_n = st.slider("Rerank to top-N", min_value=2, max_value=15, value=6, step=1, disabled=not use_rerank)
    reranker_model = st.selectbox(
        "Reranker model",
        options=[
            "BAAI/bge-reranker-base",
            "BAAI/bge-reranker-large",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ],
        index=0,
        disabled=not use_rerank,
        help="These models run locally. You may need: pip install sentence-transformers torch",
    )

    st.divider()
    st.header("Answer policy")
    require_citations = st.checkbox("Require citations in the answer", value=True)
    allow_abstain = st.checkbox("Allow abstain if evidence is insufficient", value=True)

    st.divider()
    st.header("Debug / Observability")
    show_retrieval_debug = st.checkbox("Show retrieved chunks + scores", value=False)
    log_queries = st.checkbox("Log queries to logs/queries.jsonl", value=True)

# -----------------------------
# Helpers
# -----------------------------
def safe_write_jsonl(path: Path, obj: dict) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        # Avoid breaking the app on logging failure
        pass


def build_prompt(require_citations: bool, allow_abstain: bool) -> PromptTemplate:
    # We ask for citations using the source labels we add in formatting below: [pX].
    # RetrievalQA "stuff" chain will inject {context}.
    policy_bits = []
    if require_citations:
        policy_bits.append(
            "Cite your sources inline using bracketed page markers like [p3]. "
            "Every substantive claim should be supported by at least one citation."
        )
    if allow_abstain:
        policy_bits.append(
            "If the provided context does not contain enough evidence to answer, say so clearly and ask a focused clarifying question. "
            "Do not guess."
        )
    else:
        policy_bits.append("Answer using the provided context. Do not invent facts not present in the context.")

    policy = " ".join(policy_bits)

    template = f"""You are a careful assistant answering questions about an uploaded PDF.
Use ONLY the provided context to answer.

{policy}

Context:
{{context}}

Question:
{{question}}

Answer:"""

    return PromptTemplate(input_variables=["context", "question"], template=template)


def format_docs_with_page_markers(docs):
    # Attach page markers to each chunk so citations are easy.
    # PyPDFLoader uses metadata like {"page": <int>, "source": <path>}
    lines = []
    for d in docs:
        page = d.metadata.get("page", None)
        if page is None:
            marker = "[p?]"
        else:
            # page is 0-indexed in many loaders; present 1-indexed to users
            marker = f"[p{int(page) + 1}]"
        lines.append(f"{marker} {d.page_content}")
    return "\n\n".join(lines)


# -----------------------------
# Main
# -----------------------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # Embeddings + FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Base vector retriever
    search_kwargs = {"k": vector_k}
    if vector_search_type == "mmr":
        # MMR needs fetch_k and lambda_mult; keep sane defaults
        search_kwargs = {"k": vector_k, "fetch_k": max(20, vector_k * 4), "lambda_mult": 0.5}

    vector_retriever = vectorstore.as_retriever(search_type=vector_search_type, search_kwargs=search_kwargs)

    # Optional BM25
    if use_bm25:
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = bm25_k
        base_retriever = EnsembleRetriever(
            retrievers=[bm25, vector_retriever],
            weights=[bm25_weight, vector_weight],
        )
    else:
        base_retriever = vector_retriever

    # LLM
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    # Optional MultiQuery (query rewriting)
    # Note: MultiQueryRetriever doesn't expose "max queries" directly in all versions.
    # We pass a custom prompt that encourages a small number of variants.
    if use_multiquery:
        mq_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "You are a search assistant. Generate up to {n} alternative search queries that help retrieve relevant passages. "
                "Return each query on a new line without numbering.\n\n"
                "Original question: {question}\n\n"
                "Alternative queries:"
            ).replace("{n}", str(max_query_variants)),
        )
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            prompt=mq_prompt,
            include_original=True,
        )
    else:
        retriever = base_retriever

    # Optional Cross-encoder reranking (ContextualCompressionRetriever wraps a base retriever)
    if use_rerank:
        cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=rerank_top_n)
        retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)

    # RetrievalQA with custom prompt + sources
    prompt = build_prompt(require_citations=require_citations, allow_abstain=allow_abstain)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    st.success("✅ PDF processed! Configure settings on the left and ask questions.")

    col1, col2 = st.columns([2, 3])

    with col1:
        query = st.text_input("Ask a question:")
        run_btn = st.button("Run", type="primary", use_container_width=True)

    with col2:
        st.subheader("Answer")
        if (query and run_btn) or (query and not run_btn):
            with st.spinner("Retrieving and answering..."):
                result = qa_chain({"query": query})
                answer = result.get("result", "")
                source_docs = result.get("source_documents", []) or []

            # Log query + settings + sources
            if log_queries:
                safe_write_jsonl(
                    QUERY_LOG,
                    {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "query": query,
                        "settings": {
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "model": model_name,
                            "temperature": temperature,
                            "vector_search_type": vector_search_type,
                            "vector_k": vector_k,
                            "use_bm25": use_bm25,
                            "bm25_k": bm25_k if use_bm25 else None,
                            "bm25_weight": bm25_weight if use_bm25 else None,
                            "use_multiquery": use_multiquery,
                            "max_query_variants": max_query_variants if use_multiquery else None,
                            "use_rerank": use_rerank,
                            "rerank_top_n": rerank_top_n if use_rerank else None,
                            "reranker_model": reranker_model if use_rerank else None,
                            "require_citations": require_citations,
                            "allow_abstain": allow_abstain,
                        },
                        "sources": [
                            {
                                "page": int(d.metadata.get("page")) + 1 if d.metadata.get("page") is not None else None,
                                "source": d.metadata.get("source"),
                            }
                            for d in source_docs[:20]
                        ],
                        "answer_preview": answer[:4000],
                    },
                )

            st.markdown(answer)

            # Sources display
            if source_docs:
                st.divider()
                st.subheader("Sources")
                # Show a compact list of citations first
                pages = []
                for d in source_docs:
                    p = d.metadata.get("page", None)
                    if p is not None:
                        pages.append(int(p) + 1)
                if pages:
                    st.write("Pages referenced (approx.): " + ", ".join(map(str, sorted(set(pages)))))

                with st.expander("Show retrieved passages"):
                    st.markdown(format_docs_with_page_markers(source_docs))

            # Retrieval debug
            if show_retrieval_debug:
                st.divider()
                st.subheader("Debug")
                st.caption(
                    "If you enable BM25 / MultiQuery / Rerank, the underlying retriever may not expose raw scores. "
                    "This section is therefore limited to showing retrieved chunk metadata."
                )
                for i, d in enumerate(source_docs[:15], start=1):
                    st.write(
                        {
                            "rank": i,
                            "page": int(d.metadata.get("page")) + 1 if d.metadata.get("page") is not None else None,
                            "source": d.metadata.get("source"),
                            "chunk_preview": d.page_content[:220].replace("\n", " "),
                        }
                    )
else:
    st.info("Upload a PDF from the sidebar to begin.")

st.caption(
    "Notes: For cross-encoder reranking, install local dependencies: "
    "`pip install sentence-transformers torch` (and optionally `accelerate`). "
    "Reranking runs on your machine; CPU works but GPU is faster."
)
