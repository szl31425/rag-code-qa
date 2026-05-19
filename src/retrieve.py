"""Hybrid retrieval: Dense (FAISS/ChromaDB) + Sparse (BM25) + CrossEncoder Reranker."""
import os
import re

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    jieba = None
    HAS_JIEBA = False

from .config import (
    EMBEDDING_MODEL,
    LOCAL_MODEL_DIR,
    RERANKER_MODEL,
    DENSE_TOP_K,
    SPARSE_TOP_K,
    FINAL_TOP_K,
    USE_RERANKER,
    USE_HYBRID,
)
from .vectorstore import load_vectorstore

# Chinese character detection regex
_CJK_RE = re.compile(r'[一-鿿㐀-䶿豈-﫿]')


def _has_chinese(text: str) -> bool:
    """Check if text contains CJK characters."""
    return bool(_CJK_RE.search(text))


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 — uses jieba for Chinese, split() for others.

    Auto-detects: if text contains CJK characters, use jieba with word-level
    segmentation. Otherwise falls back to whitespace split (sufficient for
    English and most programming languages).
    """
    if HAS_JIEBA and _has_chinese(text):
        # jieba.lcut returns word-level tokens for Chinese
        # Mix: run jieba on full text — it handles mixed Chinese/English well
        tokens = []
        for word in jieba.lcut(text):
            word = word.strip()
            if word and not word.isspace():
                tokens.append(word)
        return tokens if tokens else text.split()
    return text.split()


class HybridRetriever:
    """Combines dense (FAISS or ChromaDB) and sparse (BM25) retrieval with optional reranking."""

    def __init__(self, vectorstore=None, chunks=None, backend=None):
        self._embeddings = self._create_embeddings()
        self._vectorstore = vectorstore
        self._chunks = chunks or []
        self._bm25 = None
        self._reranker = None
        self._doc_texts = []
        self._backend = backend

        if self._vectorstore is None:
            self._vectorstore = load_vectorstore(backend=backend)

        if self._vectorstore is not None:
            self._rebuild_from_vectorstore()

        if self._chunks:
            self._build_bm25()

        if USE_RERANKER:
            self._init_reranker()

    @staticmethod
    def _create_embeddings():
        if LOCAL_MODEL_DIR and os.path.isdir(LOCAL_MODEL_DIR):
            model_path = os.path.join(LOCAL_MODEL_DIR, EMBEDDING_MODEL.replace("/", "_"))
            if os.path.isdir(model_path):
                return HuggingFaceEmbeddings(model_name=model_path)
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def _rebuild_from_vectorstore(self):
        """Extract document texts from vectorstore for BM25 indexing."""
        try:
            doc_dict = self._vectorstore.docstore._dict
            self._chunks = list(doc_dict.values())
        except Exception:
            pass

        if not self._chunks:
            try:
                coll = self._vectorstore._collection
                result = coll.get(include=["documents", "metadatas"])
                if result and result.get("documents"):
                    from langchain_core.documents import Document
                    self._chunks = [
                        Document(page_content=doc, metadata=meta or {})
                        for doc, meta in zip(result["documents"], result["metadatas"])
                    ]
            except Exception:
                pass

    def _build_bm25(self):
        """Build BM25 index from document chunks with Chinese-aware tokenization."""
        if BM25Okapi is None:
            self._bm25 = None
            return
        self._doc_texts = [d.page_content for d in self._chunks]
        tokenized = [tokenize(text) for text in self._doc_texts]
        if tokenized:
            self._bm25 = BM25Okapi(tokenized)

    def _init_reranker(self):
        """Initialize the CrossEncoder reranker (Chinese-friendly BGE by default)."""
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_MODEL)
        except Exception as e:
            print(f"Reranker init failed: {e}, falling back to no reranking")
            self._reranker = None

    def _dense_search(self, query: str, k: int = DENSE_TOP_K):
        """Dense retrieval using the vector store (FAISS or ChromaDB)."""
        if self._vectorstore is None:
            return [], []
        try:
            results = self._vectorstore.similarity_search_with_score(query, k=k)
            docs = [r[0] for r in results]
            scores = [float(r[1]) for r in results]
        except Exception:
            docs = self._vectorstore.similarity_search(query, k=k)
            scores = [0.5] * len(docs)

        similarities = [1.0 / (1.0 + s) for s in scores]
        return docs, similarities

    def _sparse_search(self, query: str, k: int = SPARSE_TOP_K):
        """Sparse retrieval using BM25 with Chinese-aware tokenization."""
        if self._bm25 is None or not self._doc_texts:
            return [], []
        tokenized_query = tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        docs = [self._chunks[i] for i in top_indices if scores[i] > 0]
        doc_scores = [float(scores[i]) for i in top_indices if scores[i] > 0]
        return docs, doc_scores

    def _merge_results(self, dense_docs, dense_scores, sparse_docs, sparse_scores):
        """Merge dense and sparse results with Reciprocal Rank Fusion."""
        def normalize(scores):
            if not scores:
                return []
            s = np.array(scores)
            if s.max() == s.min():
                return [0.5] * len(s)
            return ((s - s.min()) / (s.max() - s.min())).tolist()

        dense_norm = normalize(dense_scores)
        sparse_norm = normalize(sparse_scores)

        seen = {}
        rrf_k = 60

        for rank, (doc, score) in enumerate(zip(dense_docs, dense_norm)):
            key = doc.page_content[:200]
            rrf_score = 1.0 / (rrf_k + rank + 1)
            seen[key] = (doc, max(score, rrf_score * 2))

        for rank, (doc, score) in enumerate(zip(sparse_docs, sparse_norm)):
            key = doc.page_content[:200]
            rrf_score = 1.0 / (rrf_k + rank + 1)
            if key in seen:
                _, existing = seen[key]
                seen[key] = (doc, existing + rrf_score * 2)
            else:
                seen[key] = (doc, rrf_score * 2)

        merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in merged]

    def _rerank(self, query: str, docs, top_k: int = FINAL_TOP_K):
        """Rerank documents using CrossEncoder (Chinese-friendly BGE model)."""
        if self._reranker is None or not docs:
            return docs[:top_k]
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def retrieve(self, query: str, top_k: int = None, metadata_filter: dict = None):
        """Main retrieval: hybrid search + metadata filtering + reranking + boosting.

        Args:
            query: search query
            top_k: number of results
            metadata_filter: optional dict for pre-retrieval filtering
                             e.g. {"file_type": "code"}, {"filename": "manual.pdf"}
                             Only fully supported by ChromaDB; FAISS does post-hoc filter.
        """
        if top_k is None:
            top_k = FINAL_TOP_K

        # Fetch more candidates than needed so we have room after filtering
        fetch_k = max(top_k * 3, DENSE_TOP_K)

        if USE_HYBRID:
            dense_docs, dense_scores = self._dense_search(query, k=fetch_k)
            sparse_docs, sparse_scores = self._sparse_search(query, k=fetch_k)
            merged = self._merge_results(
                dense_docs, dense_scores, sparse_docs, sparse_scores
            )
        else:
            merged, _ = self._dense_search(query, k=fetch_k)

        # Pre-retrieval / post-hoc metadata filtering
        if metadata_filter:
            merged = self._filter_by_metadata(merged, metadata_filter)

        # Post-retrieval metadata boosting
        merged = self._boost_by_metadata(merged, query)

        if USE_RERANKER and len(merged) > top_k:
            merged = self._rerank(query, merged, top_k)

        return merged[:top_k]

    # ---- Metadata filtering & boosting ----

    def _filter_by_metadata(self, docs: list, filter_dict: dict) -> list:
        """Filter documents by metadata key-value pairs (post-hoc for FAISS compat).

        ChromaDB supports native where-filtering, but FAISS does not. This works
        for both backends by post-hoc filtering. The fetch_k multiplier in
        retrieve() compensates for the filter losses.
        """
        filtered = []
        for doc in docs:
            match = True
            for key, value in filter_dict.items():
                doc_val = doc.metadata.get(key, "")
                if isinstance(value, str) and isinstance(doc_val, str):
                    if value.lower() not in doc_val.lower():
                        match = False
                        break
                elif doc_val != value:
                    match = False
                    break
            if match:
                filtered.append(doc)
        return filtered

    def _boost_by_metadata(self, docs: list, query: str) -> list:
        """Boost document scores based on metadata-to-query relevance.

        Heuristics (applied as score multipliers):
        - Code query + code doc: 1.3x boost
        - Query terms appear in filename: 1.2x boost per term
        - Higher chunk_index (more detailed content): slight boost

        Returns re-sorted list.
        """
        if not docs:
            return docs

        query_lower = query.lower()
        # Detect if query is code-related
        code_keywords = ("代码", "函数", "class", "def ", "import", "api",
                         "code", "function", "method", "example", "示例",
                         "实现", "implementation", "编程", "programming")
        is_code_query = any(kw in query_lower for kw in code_keywords)

        scored = []
        for doc in docs:
            boost = 1.0

            # Code query → prefer code files
            if is_code_query:
                file_type = doc.metadata.get("file_type", "")
                if file_type == "code":
                    boost *= 1.3

            # Query terms appear in filename
            filename = doc.metadata.get("filename", "").lower()
            query_terms = query_lower.split()
            term_matches = sum(1 for t in query_terms if t and t in filename)
            if term_matches > 0:
                boost *= 1.0 + 0.15 * term_matches

            # Higher chunk_index (later in doc) often has details
            chunk_idx = doc.metadata.get("chunk_index", -1)
            if isinstance(chunk_idx, int) and chunk_idx > 3:
                boost *= 1.05

            scored.append((doc, boost))

        # Sort by boost (stabilizes original order for equal boosts)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored]

    def retrieve_with_metadata_strategy(self, query: str, top_k: int = None) -> dict:
        """Smart retrieval that auto-selects metadata filter based on query intent.

        Returns dict with 'docs' and 'applied_filter' for UI display.
        """
        applied_filter = None

        query_lower = query.lower()
        code_keywords = ("代码", "function", "def ", "class", "import", "api",
                         "code", "函数", "方法", "示例", "example", "实现", "源码")
        if any(kw in query_lower for kw in code_keywords):
            applied_filter = {"file_type": "code"}

        docs = self.retrieve(query, top_k=top_k, metadata_filter=applied_filter)

        # If filtering returned too few, retry without filter
        if applied_filter and len(docs) < 2:
            docs = self.retrieve(query, top_k=top_k, metadata_filter=None)
            applied_filter = None

        return {"docs": docs, "applied_filter": applied_filter}

    def update_chunks(self, chunks):
        """Update the chunk list and rebuild BM25 index."""
        self._chunks = chunks
        self._build_bm25()

    @property
    def chunk_count(self):
        return len(self._chunks)

    @property
    def backend_name(self):
        if self._backend:
            return self._backend
        if self._vectorstore is None:
            return "none"
        return "chroma" if hasattr(self._vectorstore, "_collection") else "faiss"


def load_retriever(vectorstore=None, chunks=None, backend=None):
    """Factory function to create a HybridRetriever."""
    return HybridRetriever(vectorstore=vectorstore, chunks=chunks, backend=backend)
