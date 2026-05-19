"""Streamlit UI for RAG Code QA — hybrid retrieval, semantic chunking, multi-backend."""
import os, sys, json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import streamlit as st

st.set_page_config(page_title="RAG Code QA", page_icon="📚", layout="wide")
st.title("📚 CodeDoc RAG — Intelligent Technical Document Q&A")
st.caption("Semantic Chunking | FAISS / ChromaDB | Hybrid Retrieval + Reranker | DeepSeek")

# ---- Sidebar ----
with st.sidebar:
    st.header("📄 Document Management")

    uploaded = st.file_uploader(
        "Upload PDF / MD / TXT / Code",
        type=["pdf", "md", "txt", "py", "js", "ts", "java", "go", "rs"],
        accept_multiple_files=True,
    )

    # Chunk strategy
    st.subheader("🔪 Chunking")
    chunk_strategy = st.selectbox(
        "Strategy",
        ["fixed_size", "hierarchical", "embedding_semantic", "code"],
        index=1,
        format_func=lambda s: {
            "fixed_size": "fixed_size — equal-sized chunks",
            "hierarchical": "hierarchical — markdown-aware boundaries",
            "embedding_semantic": "embedding_semantic — AI similarity-based splits",
            "code": "code — AST-aware (function/class)",
        }[s],
        help="embedding_semantic uses the embedding model to find natural break points",
    )
    chunk_size = st.slider("Chunk size (ignored by embedding_semantic)", 200, 1500, 500, 50)

    # Vector store backend
    st.subheader("🗄️ Vector Database")
    vector_backend = st.selectbox(
        "Backend",
        ["faiss", "chroma"],
        index=0,
        format_func=lambda b: {
            "faiss": "FAISS — fast local index, no deps",
            "chroma": "ChromaDB — incremental updates, metadata filtering",
        }[b],
    )
    collection_name = st.text_input("Collection name", "default",
                                     help="ChromaDB collection. FAISS uses only 'default'.")

    col_build, col_del = st.columns(2)
    with col_build:
        if st.button("🔄 Build Index", use_container_width=True):
            if not uploaded:
                st.warning("Please upload files first")
            else:
                with st.spinner(f"Building index ({vector_backend}, {chunk_strategy})..."):
                    from langchain_community.document_loaders import PyPDFLoader, TextLoader
                    from src.ingest import split_documents, build_index
                    from src.vectorstore import delete_collection

                    # Delete old collection if rebuilding
                    try:
                        delete_collection(collection_name, backend=vector_backend)
                    except Exception:
                        pass

                    docs = []
                    save_dir = os.path.join(os.path.dirname(__file__), "docs")
                    os.makedirs(save_dir, exist_ok=True)

                    for f in uploaded:
                        path = os.path.join(save_dir, f.name)
                        with open(path, "wb") as fp:
                            fp.write(f.getbuffer())
                        try:
                            if f.name.endswith(".pdf"):
                                loaded = PyPDFLoader(path).load()
                            else:
                                loaded = TextLoader(path, encoding="utf-8").load()
                            for d in loaded:
                                d.metadata["filename"] = f.name
                            docs.extend(loaded)
                        except Exception as e:
                            st.error(f"Failed to load {f.name}: {e}")

                    if docs:
                        try:
                            vs, chunks = build_index(
                                docs, strategy=chunk_strategy,
                                backend=vector_backend,
                                collection_name=collection_name,
                            )
                            st.session_state["chunks"] = chunks
                            st.session_state["qa_pipeline"] = None
                            st.session_state["vector_backend"] = vector_backend
                            st.session_state["collection"] = collection_name
                            st.success(
                                f"Index built: {len(chunks)} chunks "
                                f"(backend={vector_backend}, strategy={chunk_strategy})"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Build failed: {e}")

    with col_del:
        if st.button("🗑️ Delete", use_container_width=True):
            from src.vectorstore import delete_collection
            try:
                delete_collection(collection_name, backend=vector_backend)
                st.session_state["qa_pipeline"] = None
                st.session_state["chunks"] = None
                st.success(f"Collection '{collection_name}' deleted")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    # Show existing collections
    from src.vectorstore import list_collections, load_vectorstore
    collections = list_collections(backend=vector_backend)
    if collections:
        st.caption(f"Existing: {', '.join(collections)}")

    st.divider()

    # Retrieval settings
    st.subheader("⚙️ Retrieval")
    use_hybrid = st.checkbox("Hybrid (Dense + BM25)", value=True)
    use_reranker = st.checkbox("CrossEncoder Reranker", value=True)
    use_rewrite = st.checkbox("Query Rewriting", value=True)
    final_k = st.slider("Documents to retrieve", 1, 10, 4)

    st.divider()

    # Eval
    st.subheader("📊 Evaluation")
    if st.button("Run RAG Evaluation", use_container_width=True):
        with st.spinner("Running..."):
            from src.eval import run_ragas_eval
            from src.qa import QAPipeline
            pipeline = QAPipeline()
            results = run_ragas_eval(pipeline)
            if results:
                st.json({k: v for k, v in results.items() if k != "results"})

    st.divider()
    st.markdown(
        f"**Tech:** {vector_backend.upper()} + BM25 + CrossEncoder + DeepSeek\n\n"
        "**Strategies:** fixed / hierarchical / embedding_semantic / code"
    )

# ---- Apply settings ----
os.environ["USE_HYBRID"] = str(use_hybrid).lower()
os.environ["USE_RERANKER"] = str(use_reranker).lower()
os.environ["ENABLE_QUERY_REWRITE"] = str(use_rewrite).lower()
os.environ["FINAL_TOP_K"] = str(final_k)
os.environ["CHUNK_SIZE"] = str(chunk_size)
os.environ["VECTOR_BACKEND"] = vector_backend

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_pipeline" not in st.session_state:
    st.session_state.qa_pipeline = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "vector_backend" not in st.session_state:
    st.session_state.vector_backend = vector_backend
if "collection" not in st.session_state:
    st.session_state.collection = collection_name

# ---- Clear chat button ----
col_chat, col_clear = st.columns([6, 1])
with col_clear:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---- Main Chat ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.caption(f"• {s}")
        if msg.get("rewritten"):
            st.caption(f"🔍 Rewritten: _{msg['rewritten']}_")
        if msg.get("decomposed"):
            st.caption(f"🧩 Decomposed into {msg['decomposed']} sub-questions")
        if msg.get("filter"):
            st.caption(f"🏷️ Metadata filter: `{msg['filter']}`")

if prompt := st.chat_input("Ask about your documents... (multi-turn & decomposition supported)"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Rewriting → Decomposing → Retrieving → Generating..."):
            try:
                from src.qa import QAPipeline
                from src.config import VALIDATION_MODEL
                import importlib, src.config as cfg
                importlib.reload(cfg)

                if st.session_state.qa_pipeline is None:
                    st.session_state.qa_pipeline = QAPipeline(
                        backend=st.session_state.vector_backend
                    )

                # Build conversation history from last N exchanges
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]  # exclude current
                    if m["role"] in ("user", "assistant")
                ][-10:]  # last 5 exchanges max

                result = st.session_state.qa_pipeline.ask(
                    prompt,
                    history=history if history else None,
                )

                st.markdown(result["answer"])
                sources = result.get("sources", [])
                rewritten = result.get("rewritten_query")
                is_decomposed = result.get("is_decomposed", False)
                sub_questions = result.get("sub_questions", [])
                sub_results = result.get("sub_results", [])

                msg_data = {"role": "assistant", "content": result["answer"],
                            "sources": sources}

                # Show rewriting result
                if rewritten:
                    msg_data["rewritten"] = rewritten
                    st.caption(f"🔍 Rewritten: _{rewritten}_")

                # Show decomposition info
                if is_decomposed and sub_questions:
                    msg_data["decomposed"] = len(sub_questions)
                    with st.expander(f"🧩 Decomposed into {len(sub_questions)} sub-questions"):
                        for i, sq in enumerate(sub_questions):
                            sub_r = sub_results[i] if i < len(sub_results) else {}
                            applied = sub_r.get("applied_filter")
                            v = sub_r.get("validation", {})
                            vs = f" score={v.get('overall_score', '?')}" if v else ""
                            f_str = f" [filter: {applied}]" if applied else ""
                            r_str = " 🔄 retried" if sub_r.get("retried") else ""
                            st.caption(f"{i+1}. {sq}{f_str}{vs}{r_str}")

                # Show validation result
                validation = result.get("validation")
                if not validation and sub_results:
                    validation = sub_results[0].get("validation")
                if validation:
                    from src.validator import ResponseValidator
                    rv = ResponseValidator()
                    with st.expander(f"🔍 Validation: {validation.get('overall_score', '?')}/100 ({validation.get('verdict', '?')})"):
                        st.markdown(rv.format_validation_badge(validation))
                    if sub_results and sub_results[0].get("retried"):
                        orig = sub_results[0].get("original_score", "?")
                        improved = sub_results[0].get("improved_score", "?")
                        st.caption(f"🔄 Auto-retried: score {orig} → {improved} (using {VALIDATION_MODEL})")

                # Show metadata filter
                if sub_results:
                    first_filter = sub_results[0].get("applied_filter")
                    if first_filter:
                        msg_data["filter"] = str(first_filter)
                        st.caption(f"🏷️ Auto metadata filter: `{first_filter}`")

                # Show sources
                if sources:
                    with st.expander(f"📎 Sources ({len(sources)})"):
                        for s in sources:
                            st.caption(f"• {s}")

                st.session_state.messages.append(msg_data)

            except Exception as e:
                err = f"Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
