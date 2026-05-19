"""Document ingestion with multiple chunking strategies and vector store backends."""
import os, re

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from .config import (
    DOCS_DIR,
    EMBEDDING_MODEL,
    LOCAL_MODEL_DIR,
    CHUNK_STRATEGY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)

from .vectorstore import create_vectorstore, _create_embeddings

LANG_MAP = {
    ".py": Language.PYTHON, ".js": Language.JS, ".ts": Language.TS,
    ".java": Language.JAVA, ".go": Language.GO, ".rs": Language.RUST,
    ".cpp": Language.CPP, ".c": Language.CPP, ".h": Language.CPP,
}

# On first import, try to init the SemanticChunker embedder
_semantic_embeddings = None


def _get_semantic_embeddings():
    """Lazy-load embeddings for SemanticChunker (cached)."""
    global _semantic_embeddings
    if _semantic_embeddings is None:
        _semantic_embeddings = _create_embeddings()
    return _semantic_embeddings


def _detect_file_type(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in LANG_MAP:
        return "code"
    if ext == ".md":
        return "markdown"
    return "text"


# ---- Chunking strategies ----

def _split_fixed(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Fixed-size chunking — the baseline strategy."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def _split_hierarchical(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Structure-aware chunking — splits on markdown headers, paragraphs etc."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ", "\n### ", "\n#### ",
            "\n---\n", "\n\n", "\n", ". ", " ", "",
        ],
    )
    return splitter.split_documents(docs)


def _split_code(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """AST-aware splitting — splits on function/class boundaries for code files."""
    chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        lang = LANG_MAP.get(os.path.splitext(source)[1].lower())
        if lang:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )
        chunks.extend(splitter.split_documents([doc]))
    return chunks


def _split_embedding_semantic(docs, chunk_size=None):
    """True semantic chunking — uses embedding similarity to find natural break points.

    The SemanticChunker computes embeddings for each sentence, then measures
    cosine similarity between adjacent sentences. When similarity drops below
    a threshold (interquartile range by default), it splits — creating chunks
    that are semantically coherent rather than arbitrarily sized.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        print("langchain_experimental not installed. Install with: pip install langchain-experimental")
        print("Falling back to hierarchical chunking.")
        return _split_hierarchical(docs)

    embeddings = _get_semantic_embeddings()
    # breakpoint_threshold_type: "interquartile" | "percentile" | "standard_deviation"
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="interquartile",
        buffer_size=1,       # sentences to group on each side of gradient
        add_start_index=True, # track position
    )
    chunks = splitter.split_documents(docs)
    # Add chunk index metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def split_documents(docs, strategy=None):
    """Split documents using the configured strategy.

    Strategies:
      - fixed_size:          equal-sized chunks (baseline)
      - hierarchical:        splits on markdown headers and natural boundaries
      - embedding_semantic:  true semantic chunking using embedding similarity
      - code:                AST-aware, splits on function/class boundaries
    """
    if strategy is None:
        strategy = CHUNK_STRATEGY

    splitters = {
        "fixed_size": _split_fixed,
        "hierarchical": _split_hierarchical,
        "embedding_semantic": _split_embedding_semantic,
        "code": _split_code,
    }
    # Backward compat: "semantic" maps to the new hierarchical (old behavior)
    if strategy == "semantic":
        strategy = "hierarchical"

    split_fn = splitters.get(strategy, _split_fixed)
    return split_fn(docs)


# ---- Document loading ----

def load_documents(docs_dir=None):
    """Load all documents from the docs directory."""
    if docs_dir is None:
        docs_dir = DOCS_DIR

    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        return []

    all_docs = []
    extensions = [
        ("**/*.pdf", PyPDFLoader), ("**/*.md", TextLoader),
        ("**/*.txt", TextLoader), ("**/*.py", TextLoader),
        ("**/*.js", TextLoader), ("**/*.ts", TextLoader),
        ("**/*.java", TextLoader), ("**/*.go", TextLoader),
    ]
    for ext, loader_cls in extensions:
        try:
            loader = DirectoryLoader(docs_dir, glob=ext, loader_cls=loader_cls,
                                     silent_errors=True)
            all_docs.extend(loader.load())
        except Exception:
            pass

    for doc in all_docs:
        source = doc.metadata.get("source", "")
        doc.metadata["file_type"] = _detect_file_type(source)
        doc.metadata["filename"] = os.path.basename(source)

    return all_docs


# ---- Index building ----

def build_index(docs=None, strategy=None, backend=None,
                collection_name="default", save=True):
    """Build a vector index from documents.

    Args:
        docs:             list of LangChain Documents (auto-loaded if None)
        strategy:         chunking strategy name
        backend:          "faiss" or "chroma"
        collection_name:  ChromaDB collection name
        save:             persist to disk
    """
    if docs is None:
        docs = load_documents()

    if not docs:
        print("No documents found. Add files to docs/ directory.")
        return None, []

    strategy = strategy or CHUNK_STRATEGY
    chunks = split_documents(docs, strategy)
    print(f"Strategy={strategy}: {len(docs)} docs -> {len(chunks)} chunks")

    vectorstore = create_vectorstore(chunks, backend=backend,
                                     collection_name=collection_name)
    return vectorstore, chunks


if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        sample = os.path.join(DOCS_DIR, "sample_python_guide.md")
        os.makedirs(DOCS_DIR, exist_ok=True)
        with open(sample, "w", encoding="utf-8") as f:
            f.write(
                "# Python Coding Guide\n\n"
                "## Naming Conventions\n"
                "- Variables: snake_case\n"
                "- Constants: UPPER_CASE\n"
                "- Classes: PascalCase\n\n"
                "## Error Handling\n"
                "- Use try/except for expected errors\n"
                "- Never use bare except:\n"
                "- Log exceptions with traceback\n\n"
                "## Best Practices\n"
                "- Functions should have docstrings\n"
                "- Keep functions under 50 lines\n"
                "- Use type hints for function signatures\n"
            )
        print(f"Created sample document: {sample}")
        docs = load_documents()

    for strategy in ["fixed_size", "hierarchical", "embedding_semantic", "code"]:
        print(f"\n--- Testing {strategy} ---")
        try:
            build_index(docs, strategy=strategy, save=(strategy == "fixed_size"))
        except Exception as e:
            print(f"  Error: {e}")
    print("\nDone!")
