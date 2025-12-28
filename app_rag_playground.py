import os
import json
import math
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Optional open-source components
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


# -----------------------------
# Constants / Paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
QUERY_LOG_PATH = LOG_DIR / "queries.jsonl"


# -----------------------------
# Utilities
# -----------------------------
def safe_write_jsonl(path: Path, obj: dict) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


def minmax_norm(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    mn, mx = float(min(scores)), float(max(scores))
    if math.isclose(mn, mx):
        return [1.0 for _ in scores]
    return [(float(s) - mn) / (mx - mn) for s in scores]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def tokenize_for_bm25(text: str) -> List[str]:
    # Simple, fast tokenizer for prototyping (BM25 is robust to this).
    return [t for t in "".join([c.lower() if c.isalnum() else " " for c in text]).split() if t]


def format_docs_with_page_markers(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page")
        page_str = f"p{int(page)+1}" if page is not None else "p?"
        lines.append(f"[{page_str}] {d.page_content}")
    return "\n\n".join(lines)


def build_answer_prompt(require_citations: bool, allow_abstain: bool) -> PromptTemplate:
    abstain_clause = ""
    if allow_abstain:
        abstain_clause = (
            "If the provided excerpts do not contain enough evidence to answer confidently, say: "
            "\"I don’t have enough evidence in the provided text to answer that. "
            "Could you point me to the relevant chapter/page or clarify what to look for?\" "
            "Do not guess.\n"
        )

    citation_clause = ""
    if require_citations:
        citation_clause = (
            "Cite the excerpts you used by including page markers like [p12] inline after relevant sentences. "
            "Only cite pages that appear in the provided excerpts.\n"
        )

    template = (
        "You are a careful assistant answering questions about an uploaded PDF.\n"
        "Use ONLY the provided excerpts as evidence.\n"
        f"{abstain_clause}"
        f"{citation_clause}"
        "\n"
        "Question:\n{question}\n\n"
        "Excerpts:\n{context}\n\n"
        "Answer:"
    )
    return PromptTemplate(input_variables=["question", "context"], template=template)


@dataclass
class Candidate:
    chunk_id: str
    doc: Document
    # provenance
    method_hits: List[Dict[str, Any]]
    # raw scores
    vector_score: Optional[float] = None  # higher is better after conversion
    bm25_score: Optional[float] = None    # higher is better
    combined_score: Optional[float] = None
    rerank_score: Optional[float] = None


# -----------------------------
# Retrieval components
# -----------------------------
def build_bm25(docs: List[Document]) -> Tuple[Any, List[List[str]]]:
    if BM25Okapi is None:
        raise RuntimeError("BM25 dependencies not installed. Run: pip install rank-bm25")
    tokenized = [tokenize_for_bm25(d.page_content) for d in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized


def faiss_similarity_candidates(
    vectorstore: FAISS,
    query: str,
    k: int,
) -> List[Tuple[Document, float]]:
    # LangChain FAISS returns (doc, score). For L2 distance, smaller is better.
    # We convert to a "higher is better" score as (-distance).
    pairs = vectorstore.similarity_search_with_score(query, k=k)
    converted = []
    for d, dist in pairs:
        # If dist is already similarity, negating won't be correct; but for LC FAISS L2 default it's a distance.
        converted.append((d, float(-dist)))
    return converted


def faiss_mmr_candidates(
    vectorstore: FAISS,
    embeddings: OpenAIEmbeddings,
    query: str,
    fetch_k: int,
    k: int,
    lambda_mult: float,
) -> List[Tuple[Document, float]]:
    """
    Implement MMR selection while still exposing a query-similarity score.
    Approach:
      1) get fetch_k candidates via similarity_search_with_score (with converted score)
      2) reconstruct vectors for those candidates from the FAISS index
      3) run MMR selection on vectors
      4) return selected docs with their query-sim score (converted)
    """
    # Step 1: candidate pool
    pool = vectorstore.similarity_search_with_score(query, k=fetch_k)
    if not pool:
        return []

    # Query vector (normalized)
    qv = np.array(embeddings.embed_query(query), dtype=np.float32)
    qv = l2_normalize(qv)

    # Map docstore ids to faiss positions
    inv = {doc_id: idx for idx, doc_id in vectorstore.index_to_docstore_id.items()}

    pool_docs: List[Document] = []
    pool_vecs: List[np.ndarray] = []
    pool_qscores: List[float] = []

    for d, dist in pool:
        doc_id = d.metadata.get("_docstore_id")
        if doc_id is None:
            # LangChain FAISS sometimes stores docstore id outside metadata; fall back to text hash.
            # In practice with LC FAISS, _docstore_id is available.
            continue
        idx = inv.get(doc_id)
        if idx is None:
            continue
        v = vectorstore.index.reconstruct(int(idx))
        v = l2_normalize(np.array(v, dtype=np.float32))
        pool_docs.append(d)
        pool_vecs.append(v)
        pool_qscores.append(float(-dist))

    if not pool_docs:
        return []

    # MMR selection
    selected: List[int] = []
    candidate_idxs = list(range(len(pool_docs)))

    # precompute sim(query, doc)
    sim_q = np.array([float(np.dot(qv, v)) for v in pool_vecs], dtype=np.float32)
    # doc-doc similarities for diversity penalty (computed on the fly)
    while candidate_idxs and len(selected) < k:
        if not selected:
            best = int(np.argmax(sim_q[candidate_idxs]))
            chosen = candidate_idxs[best]
            selected.append(chosen)
            candidate_idxs.remove(chosen)
            continue

        best_score = None
        best_idx = None
        for ci in candidate_idxs:
            relevance = sim_q[ci]
            diversity = max(float(np.dot(pool_vecs[ci], pool_vecs[s])) for s in selected)
            score = lambda_mult * relevance - (1.0 - lambda_mult) * diversity
            if best_score is None or score > best_score:
                best_score = score
                best_idx = ci

        selected.append(best_idx)
        candidate_idxs.remove(best_idx)

    # Return selected docs and their original query-score (converted distance) for transparency
    out = [(pool_docs[i], pool_qscores[i]) for i in selected]
    return out


def bm25_candidates(
    bm25: Any,
    docs: List[Document],
    query: str,
    k: int,
) -> List[Tuple[Document, float]]:
    q_tokens = tokenize_for_bm25(query)
    scores = bm25.get_scores(q_tokens)
    scores = np.array(scores, dtype=np.float32)
    if len(scores) == 0:
        return []
    top_idx = np.argsort(-scores)[:k]
    return [(docs[int(i)], float(scores[int(i)])) for i in top_idx if scores[int(i)] > 0]


def combine_candidates(
    vector_hits_by_query: Dict[str, List[Tuple[Document, float]]],
    bm25_hits_by_query: Dict[str, List[Tuple[Document, float]]],
    vector_weight: float,
    bm25_weight: float,
) -> List[Candidate]:
    """
    Merge hits into per-chunk candidates, keep provenance and compute a combined score.
    We compute min-max normalized scores within each method across the union set for stability.
    """
    # Gather all unique docs by chunk_id
    tmp: Dict[str, Candidate] = {}

    def upsert(doc: Document, method: str, query_used: str, score: float) -> None:
        cid = doc.metadata.get("chunk_id")
        if cid is None:
            # fallback stable-ish id
            cid = f"chunk::{hash(doc.page_content)}"
            doc.metadata["chunk_id"] = cid
        c = tmp.get(cid)
        if c is None:
            tmp[cid] = Candidate(
                chunk_id=cid,
                doc=doc,
                method_hits=[{"method": method, "query_used": query_used, "score": score}],
            )
        else:
            c.method_hits.append({"method": method, "query_used": query_used, "score": score})

    for q, hits in vector_hits_by_query.items():
        for d, s in hits:
            upsert(d, "vector", q, s)

    for q, hits in bm25_hits_by_query.items():
        for d, s in hits:
            upsert(d, "bm25", q, s)

    candidates = list(tmp.values())

    # Compute representative per-candidate vector and bm25 scores (take max across provenance for that method)
    for c in candidates:
        v_scores = [h["score"] for h in c.method_hits if h["method"] == "vector"]
        b_scores = [h["score"] for h in c.method_hits if h["method"] == "bm25"]
        c.vector_score = max(v_scores) if v_scores else None
        c.bm25_score = max(b_scores) if b_scores else None

    # Normalize across candidates (separately per method)
    v_list = [c.vector_score for c in candidates if c.vector_score is not None]
    b_list = [c.bm25_score for c in candidates if c.bm25_score is not None]
    v_norm_map: Dict[str, float] = {}
    b_norm_map: Dict[str, float] = {}

    if v_list:
        normed = minmax_norm([c.vector_score if c.vector_score is not None else min(v_list) for c in candidates])
        for c, n in zip(candidates, normed):
            v_norm_map[c.chunk_id] = float(n)

    if b_list:
        normed = minmax_norm([c.bm25_score if c.bm25_score is not None else min(b_list) for c in candidates])
        for c, n in zip(candidates, normed):
            b_norm_map[c.chunk_id] = float(n)

    for c in candidates:
        v = v_norm_map.get(c.chunk_id, 0.0)
        b = b_norm_map.get(c.chunk_id, 0.0)
        # If a method is disabled, its weight will be 0.
        c.combined_score = float(vector_weight * v + bm25_weight * b)

    # Sort by combined score descending
    candidates.sort(key=lambda x: x.combined_score if x.combined_score is not None else -1e9, reverse=True)
    return candidates


def rerank_with_cross_encoder(
    query: str,
    candidates: List[Candidate],
    model_name: str,
    top_n: int,
) -> List[Candidate]:
    if CrossEncoder is None:
        raise RuntimeError("Reranker dependencies not installed. Run: pip install sentence-transformers torch")
    if not candidates:
        return candidates

    # Only rerank a limited pool for latency
    pool = candidates[:top_n]
    pairs = [(query, c.doc.page_content) for c in pool]
    model = CrossEncoder(model_name)
    scores = model.predict(pairs)

    for c, s in zip(pool, scores):
        c.rerank_score = float(s)

    # Candidates beyond pool keep None rerank_score; keep them after reranked pool
    reranked_pool = sorted(pool, key=lambda x: x.rerank_score if x.rerank_score is not None else -1e9, reverse=True)
    tail = candidates[top_n:]
    return reranked_pool + tail


# -----------------------------
# Query rewriting
# -----------------------------
def rewrite_queries(llm: ChatOpenAI, question: str, n: int) -> List[str]:
    prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are a search assistant. Generate up to {n} alternative search queries that preserve the original meaning.\n"
            "- Do NOT answer the question.\n"
            "- Do NOT introduce new assumptions.\n"
            "- Keep each query short and search-oriented.\n"
            "Return one query per line, without numbering.\n\n"
            "Original question: {question}\n"
            "Alternative queries:"
        ).replace("{n}", str(n)),
    )
    txt = llm.invoke(prompt.format(question=question)).content
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    # Deduplicate while preserving order
    out = []
    seen = set()
    for l in lines:
        if l.lower() not in seen:
            out.append(l)
            seen.add(l.lower())
    return out[:n]


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG Playground (Local)", layout="wide")
st.title("RAG Playground (Local, Open Source + OpenAI API)")
st.caption(
    "Upload a PDF, experiment with hybrid retrieval, query rewriting, and cross-encoder reranking. "
    "Use Debug to inspect rewritten queries, retrieved chunks and scores, and reranking."
)

# Fail fast on key (better UX)
if not os.getenv("OPENAI_API_KEY"):
    st.warning(
        "OPENAI_API_KEY is not set. Set it in your environment or in .streamlit/secrets.toml. "
        "The app will not be able to embed or answer questions until it is set."
    )

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    st.divider()
    st.header("Chunking")
    chunk_size = st.slider("Chunk size (characters)", 400, 4000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 800, 200, 50)

    st.divider()
    st.header("LLM")
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.divider()
    st.header("Vector retrieval")
    vector_search_type = st.selectbox("Search type", ["similarity", "mmr"], index=0)
    vector_k = st.slider("Vector top-k", 2, 60, 12, 1)
    mmr_fetch_k = st.slider("MMR fetch_k", 10, 200, 60, 5, help="Candidate pool size for MMR (higher = better, slower).")
    mmr_lambda = st.slider("MMR lambda", 0.0, 1.0, 0.5, 0.05, help="Higher = prioritize relevance; lower = prioritize diversity.")

    st.divider()
    st.header("BM25 lexical retrieval")
    use_bm25 = st.checkbox("Enable BM25", value=True, help="Keyword retrieval; good for names, terms, and exact matches.")
    bm25_k = st.slider("BM25 top-k", 2, 80, 20, 1, disabled=not use_bm25)
    bm25_weight = st.slider("BM25 weight", 0.0, 1.0, 0.45, 0.05, disabled=not use_bm25)
    vector_weight = 1.0 - bm25_weight if use_bm25 else 1.0

    st.divider()
    st.header("Query rewriting")
    use_rewrite = st.checkbox("Enable query rewriting (LLM)", value=False)
    rewrite_n = st.slider("Rewrite variants", 1, 6, 3, 1, disabled=not use_rewrite)

    st.divider()
    st.header("Reranking")
    use_rerank = st.checkbox("Enable cross-encoder reranking (local)", value=False)
    reranker_model = st.selectbox(
        "Reranker model",
        ["BAAI/bge-reranker-base", "BAAI/bge-reranker-large", "cross-encoder/ms-marco-MiniLM-L-6-v2"],
        index=0,
        disabled=not use_rerank,
        help="Requires: pip install sentence-transformers torch",
    )
    rerank_pool = st.slider("Rerank pool size", 10, 200, 80, 5, disabled=not use_rerank)
    final_k = st.slider("Final context chunks", 2, 20, 8, 1)

    st.divider()
    st.header("Answer policy")
    require_citations = st.checkbox("Require citations", value=True)
    allow_abstain = st.checkbox("Allow abstain", value=True)

    st.divider()
    st.header("Debug / Observability")
    show_debug = st.checkbox("Show detailed retrieval debug", value=True)
    log_queries = st.checkbox("Log queries to logs/queries.jsonl", value=True)

tabs = st.tabs(["Ask", "Evaluate"])

# -----------------------------
# Build indexes when PDF is uploaded
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_indexes_from_pdf(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)

    # Add stable chunk ids and preserve docstore ids for FAISS debug later
    for i, d in enumerate(chunks):
        d.metadata = dict(d.metadata or {})
        d.metadata["chunk_id"] = f"c{i:06d}"
        # Helpful for display
        if "source" not in d.metadata:
            d.metadata["source"] = "uploaded_pdf"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Ensure docstore ids are accessible from metadata for MMR vector reconstruction
    # LangChain stores docstore ids in vectorstore.index_to_docstore_id; we mirror onto docs for ease.
    for faiss_pos, docstore_id in vectorstore.index_to_docstore_id.items():
        try:
            doc = vectorstore.docstore.search(docstore_id)
            if doc and isinstance(doc, Document):
                doc.metadata = dict(doc.metadata or {})
                doc.metadata["_docstore_id"] = docstore_id
        except Exception:
            continue

    bm25 = None
    if use_bm25:
        bm25, _ = build_bm25(chunks)

    return chunks, vectorstore, embeddings, bm25


with tabs[0]:
    if uploaded_file is None:
        st.info("Upload a PDF from the sidebar to begin.")
    else:
        pdf_bytes = uploaded_file.getvalue()
        with st.spinner("Indexing PDF (chunking + embeddings + FAISS, and BM25 if enabled)..."):
            try:
                docs, vectorstore, embeddings, bm25 = build_indexes_from_pdf(pdf_bytes, chunk_size, chunk_overlap)
            except Exception as e:
                st.error(f"Failed to build indexes: {e}")
                st.stop()

        st.success(f"Indexed {len(docs)} chunks.")

        question = st.text_input("Ask a question about the PDF", value="")
        ask = st.button("Ask", type="primary", disabled=not bool(question.strip()) or not os.getenv("OPENAI_API_KEY"))

        if ask and question.strip():
            llm = ChatOpenAI(model=model_name, temperature=temperature)

            # Rewrite queries
            rewrite_list: List[str] = []
            if use_rewrite:
                with st.spinner("Rewriting query..."):
                    rewrite_list = rewrite_queries(llm, question.strip(), rewrite_n)

            search_queries = [question.strip()] + rewrite_list

            # Retrieve candidates per query and method
            vector_hits_by_query: Dict[str, List[Tuple[Document, float]]] = {}
            bm25_hits_by_query: Dict[str, List[Tuple[Document, float]]] = {}

            with st.spinner("Retrieving candidates..."):
                for q in search_queries:
                    if vector_search_type == "similarity":
                        v_hits = faiss_similarity_candidates(vectorstore, q, k=vector_k)
                    else:
                        v_hits = faiss_mmr_candidates(
                            vectorstore=vectorstore,
                            embeddings=embeddings,
                            query=q,
                            fetch_k=mmr_fetch_k,
                            k=vector_k,
                            lambda_mult=mmr_lambda,
                        )
                    vector_hits_by_query[q] = v_hits

                    if use_bm25 and bm25 is not None:
                        bm25_hits_by_query[q] = bm25_candidates(bm25, docs, q, k=bm25_k)

            candidates = combine_candidates(
                vector_hits_by_query=vector_hits_by_query,
                bm25_hits_by_query=bm25_hits_by_query,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight if use_bm25 else 0.0,
            )

            # Optional rerank
            reranked = candidates
            if use_rerank:
                with st.spinner("Reranking with cross-encoder..."):
                    reranked = rerank_with_cross_encoder(question.strip(), candidates, reranker_model, top_n=rerank_pool)

            # Final selection
            top_candidates = reranked[:final_k]
            context_docs = [c.doc for c in top_candidates]

            # Generate answer
            prompt = build_answer_prompt(require_citations=require_citations, allow_abstain=allow_abstain)
            context_str = format_docs_with_page_markers(context_docs)

            with st.spinner("Generating answer..."):
                answer = llm.invoke(prompt.format(question=question.strip(), context=context_str)).content

            st.subheader("Answer")
            st.markdown(answer)

            # Sources
            if context_docs:
                st.divider()
                st.subheader("Sources")
                pages = []
                for d in context_docs:
                    p = d.metadata.get("page")
                    if p is not None:
                        pages.append(int(p) + 1)
                if pages:
                    st.write("Pages referenced (approx.): " + ", ".join(map(str, sorted(set(pages)))))

                with st.expander("Show passages used in context"):
                    st.markdown(context_str)

            # Debug views
            if show_debug:
                st.divider()
                st.subheader("Debug")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Rewritten queries**")
                    if use_rewrite and rewrite_list:
                        st.write(rewrite_list)
                    else:
                        st.caption("Rewriting disabled or no rewrites generated.")
                    st.markdown("**Search queries used**")
                    st.write(search_queries)

                with col2:
                    st.markdown("**Pipeline settings**")
                    st.json(
                        {
                            "vector_search_type": vector_search_type,
                            "vector_k": vector_k,
                            "mmr_fetch_k": mmr_fetch_k if vector_search_type == "mmr" else None,
                            "mmr_lambda": mmr_lambda if vector_search_type == "mmr" else None,
                            "use_bm25": use_bm25,
                            "bm25_k": bm25_k if use_bm25 else None,
                            "bm25_weight": bm25_weight if use_bm25 else None,
                            "vector_weight": vector_weight,
                            "use_rewrite": use_rewrite,
                            "rewrite_n": rewrite_n if use_rewrite else None,
                            "use_rerank": use_rerank,
                            "reranker_model": reranker_model if use_rerank else None,
                            "rerank_pool": rerank_pool if use_rerank else None,
                            "final_k": final_k,
                        }
                    )

                st.markdown("**Retrieved candidates (pre-rerank ordering by combined score)**")
                # Build a flat table
                rows = []
                for rank, c in enumerate(candidates[: min(100, len(candidates))], start=1):
                    page = c.doc.metadata.get("page")
                    rows.append(
                        {
                            "rank": rank,
                            "chunk_id": c.chunk_id,
                            "page": int(page) + 1 if page is not None else None,
                            "combined_score": c.combined_score,
                            "vector_score": c.vector_score,
                            "bm25_score": c.bm25_score,
                            "methods": ", ".join(sorted(set(h["method"] for h in c.method_hits))),
                            "retrieved_by": "; ".join(
                                f'{h["method"]}("{h["query_used"][:60]}{"..." if len(h["query_used"])>60 else ""}", {h["score"]:.4f})'
                                for h in c.method_hits[:4]
                            )
                            + (" ..." if len(c.method_hits) > 4 else ""),
                            "preview": c.doc.page_content[:180].replace("\n", " "),
                        }
                    )

                st.dataframe(rows, use_container_width=True, height=420)

                if use_rerank:
                    st.markdown("**Reranked candidates (top pool reranked, then remainder)**")
                    r_rows = []
                    for rank, c in enumerate(reranked[: min(100, len(reranked))], start=1):
                        page = c.doc.metadata.get("page")
                        r_rows.append(
                            {
                                "rank": rank,
                                "chunk_id": c.chunk_id,
                                "page": int(page) + 1 if page is not None else None,
                                "rerank_score": c.rerank_score,
                                "combined_score": c.combined_score,
                                "methods": ", ".join(sorted(set(h["method"] for h in c.method_hits))),
                                "preview": c.doc.page_content[:180].replace("\n", " "),
                            }
                        )
                    st.dataframe(r_rows, use_container_width=True, height=420)

                with st.expander("Inspect a candidate chunk by id"):
                    cid = st.text_input("chunk_id (e.g., c000123)", value="")
                    if cid:
                        match = next((c for c in reranked if c.chunk_id == cid), None)
                        if match is None:
                            st.warning("No such chunk_id in current candidate set.")
                        else:
                            st.write({"chunk_id": match.chunk_id, "metadata": match.doc.metadata})
                            st.markdown(match.doc.page_content)

            # Logging
            if log_queries:
                safe_write_jsonl(
                    QUERY_LOG_PATH,
                    {
                        "ts": datetime.utcnow().isoformat(),
                        "question": question.strip(),
                        "rewrites": rewrite_list,
                        "search_queries": search_queries,
                        "settings": {
                            "vector_search_type": vector_search_type,
                            "vector_k": vector_k,
                            "mmr_fetch_k": mmr_fetch_k if vector_search_type == "mmr" else None,
                            "mmr_lambda": mmr_lambda if vector_search_type == "mmr" else None,
                            "use_bm25": use_bm25,
                            "bm25_k": bm25_k if use_bm25 else None,
                            "bm25_weight": bm25_weight if use_bm25 else None,
                            "use_rewrite": use_rewrite,
                            "rewrite_n": rewrite_n if use_rewrite else None,
                            "use_rerank": use_rerank,
                            "reranker_model": reranker_model if use_rerank else None,
                            "rerank_pool": rerank_pool if use_rerank else None,
                            "final_k": final_k,
                        },
                        "top_sources": [
                            {
                                "chunk_id": c.chunk_id,
                                "page": int(c.doc.metadata.get("page")) + 1 if c.doc.metadata.get("page") is not None else None,
                                "combined_score": c.combined_score,
                                "rerank_score": c.rerank_score,
                                "methods": c.method_hits,
                            }
                            for c in top_candidates
                        ],
                        "answer_preview": answer[:500],
                    },
                )

with tabs[1]:
    st.subheader("Evaluate retrieval configurations")

    st.markdown(
        "You can evaluate retrieval quality using a simple labelled test set.\n\n"
        "**Format:** upload a JSONL file where each line has at least `question` and an `expected_pages` list.\n\n"
        "Example line:\n"
        '`{"question": "Who kills Patroclus?", "expected_pages": [12, 13]}`\n\n'
        "Metrics computed: Recall@k and MRR (based on page matches)."
    )

    eval_file = st.file_uploader("Upload JSONL test set", type=["jsonl", "json"], key="eval")
    eval_k = st.slider("k (top-k pages to check)", 1, 30, 5, 1)

    if uploaded_file is None:
        st.info("Upload a PDF in the sidebar first (indexes are built from the PDF).")
    elif eval_file is None:
        st.info("Upload a JSONL test set to run evaluation.")
    else:
        pdf_bytes = uploaded_file.getvalue()
        with st.spinner("Ensuring indexes are built..."):
            docs, vectorstore, embeddings, bm25 = build_indexes_from_pdf(pdf_bytes, chunk_size, chunk_overlap)

        # Parse test cases
        raw = eval_file.getvalue().decode("utf-8", errors="ignore").strip()
        cases = []
        if raw.startswith("["):
            try:
                cases = json.loads(raw)
            except Exception:
                cases = []
        else:
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(json.loads(line))
                except Exception:
                    continue

        if not cases:
            st.error("Could not parse any test cases from the uploaded file.")
            st.stop()

        run = st.button("Run evaluation", type="primary", key="run_eval")

        if run:
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY is required to run evaluation when query rewriting is enabled (LLM call).")
                st.stop()

            llm = ChatOpenAI(model=model_name, temperature=0.0)

            results = []
            for i, case in enumerate(cases, start=1):
                q = (case.get("question") or "").strip()
                expected_pages = case.get("expected_pages") or case.get("expected_page") or []
                expected_pages = [int(p) for p in expected_pages] if isinstance(expected_pages, list) else [int(expected_pages)]

                if not q:
                    continue

                rewrite_list = []
                if use_rewrite:
                    rewrite_list = rewrite_queries(llm, q, rewrite_n)
                search_queries = [q] + rewrite_list

                vector_hits_by_query = {}
                bm25_hits_by_query = {}
                for sq in search_queries:
                    if vector_search_type == "similarity":
                        v_hits = faiss_similarity_candidates(vectorstore, sq, k=vector_k)
                    else:
                        v_hits = faiss_mmr_candidates(vectorstore, embeddings, sq, fetch_k=mmr_fetch_k, k=vector_k, lambda_mult=mmr_lambda)
                    vector_hits_by_query[sq] = v_hits
                    if use_bm25 and bm25 is not None:
                        bm25_hits_by_query[sq] = bm25_candidates(bm25, docs, sq, k=bm25_k)

                candidates = combine_candidates(vector_hits_by_query, bm25_hits_by_query, vector_weight, bm25_weight if use_bm25 else 0.0)
                if use_rerank:
                    candidates = rerank_with_cross_encoder(q, candidates, reranker_model, top_n=rerank_pool)

                top = candidates[:final_k]

                # Evaluate pages
                retrieved_pages = []
                for c in top:
                    p = c.doc.metadata.get("page")
                    if p is not None:
                        retrieved_pages.append(int(p) + 1)
                # unique, preserve order
                seen = set()
                retrieved_pages = [p for p in retrieved_pages if not (p in seen or seen.add(p))]

                # Recall@k (page-level)
                top_pages = retrieved_pages[:eval_k]
                hit = any(p in expected_pages for p in top_pages) if expected_pages else None

                # MRR (first relevant page rank)
                rr = 0.0
                if expected_pages:
                    for rank, p in enumerate(retrieved_pages, start=1):
                        if p in expected_pages:
                            rr = 1.0 / rank
                            break

                results.append(
                    {
                        "case": i,
                        "question": q,
                        "expected_pages": expected_pages,
                        "retrieved_pages": retrieved_pages[: max(eval_k, 10)],
                        f"recall@{eval_k}": float(hit) if hit is not None else None,
                        "mrr": rr if expected_pages else None,
                    }
                )

            if not results:
                st.error("No evaluation results produced (check your test file).")
                st.stop()

            # Aggregate
            recalls = [r[f"recall@{eval_k}"] for r in results if r[f"recall@{eval_k}"] is not None]
            mrrs = [r["mrr"] for r in results if r["mrr"] is not None]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Recall@{eval_k}", f"{(sum(recalls)/len(recalls)):.3f}" if recalls else "n/a")
            with col2:
                st.metric("MRR", f"{(sum(mrrs)/len(mrrs)):.3f}" if mrrs else "n/a")
            with col3:
                st.metric("Cases", str(len(results)))

            st.dataframe(results, use_container_width=True, height=520)

st.caption(
    "Notes: BM25 requires `pip install rank-bm25`. Reranking requires `pip install sentence-transformers torch`. "
    "Vector similarity scores shown for FAISS are converted as (-distance) for transparency; their absolute values are not meaningful across indexes."
)
