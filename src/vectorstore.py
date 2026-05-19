"""Unified vector store abstraction: FAISS + ChromaDB, switchable via config."""
import os, shutil

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain_chroma import Chroma
    HAS_CHROMA = True
except ImportError:
    Chroma = None
    HAS_CHROMA = False

from .config import (
    DATA_DIR,
    INDEX_DIR,
    EMBEDDING_MODEL,
    LOCAL_MODEL_DIR,
    FINAL_TOP_K,
)

CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")  # "faiss" or "chroma"


def _create_embeddings():
    """Create embeddings with local model fallback."""
    if LOCAL_MODEL_DIR and os.path.isdir(LOCAL_MODEL_DIR):
        model_path = os.path.join(LOCAL_MODEL_DIR, EMBEDDING_MODEL.replace("/", "_"))
        if os.path.isdir(model_path):
            return HuggingFaceEmbeddings(model_name=model_path)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vectorstore(chunks, backend=None, collection_name="default"):
    """Create a vector store from document chunks.

    Args:
        chunks: List of LangChain Document objects
        backend: "faiss" or "chroma" (default: VECTOR_BACKEND from env)
        collection_name: ChromaDB collection name (ignored for FAISS)
    """
    if backend is None:
        backend = VECTOR_BACKEND

    embeddings = _create_embeddings()

    if backend == "chroma":
        if not HAS_CHROMA:
            print("ChromaDB not installed. Run: pip install langchain-chroma chromadb")
            print("Falling back to FAISS.")
            backend = "faiss"
        else:
            os.makedirs(CHROMA_DIR, exist_ok=True)
            vs = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
                collection_name=collection_name,
            )
            print(f"ChromaDB store created: {len(chunks)} chunks in collection '{collection_name}'")
            return vs

    if backend == "faiss":
        vs = FAISS.from_documents(chunks, embeddings)
        os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)
        vs.save_local(INDEX_DIR)
        print(f"FAISS store created: {len(chunks)} chunks at {INDEX_DIR}")
        return vs

    # Unknown backend, fall back to FAISS
    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"FAISS store created (fallback): {len(chunks)} chunks at {INDEX_DIR}")
    return vs


def load_vectorstore(backend=None, collection_name="default"):
    """Load an existing vector store.

    Returns None if the store doesn't exist yet.
    """
    if backend is None:
        backend = VECTOR_BACKEND

    embeddings = _create_embeddings()

    if backend == "chroma":
        if not HAS_CHROMA:
            print("ChromaDB not installed, falling back to FAISS.")
            backend = "faiss"
        else:
            if not os.path.exists(CHROMA_DIR):
                return None
            try:
                vs = Chroma(
                    embedding_function=embeddings,
                    persist_directory=CHROMA_DIR,
                    collection_name=collection_name,
                )
                if vs._collection.count() > 0:
                    print(f"ChromaDB loaded: {vs._collection.count()} docs in '{collection_name}'")
                    return vs
                return None
            except Exception:
                return None

    if backend == "faiss" or True:
        if not os.path.exists(INDEX_DIR):
            return None
        try:
            vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            print(f"FAISS loaded from {INDEX_DIR}")
            return vs
        except Exception:
            return None


def list_collections(backend=None):
    """List available collections / indices."""
    if backend is None:
        backend = VECTOR_BACKEND

    if backend == "chroma":
        if not os.path.exists(CHROMA_DIR):
            return []
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            return [c.name for c in client.list_collections()]
        except Exception:
            return []
    else:
        if os.path.exists(INDEX_DIR):
            return ["default"]
        return []


def delete_collection(collection_name="default", backend=None):
    """Delete a collection / index."""
    if backend is None:
        backend = VECTOR_BACKEND

    if backend == "chroma":
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            try:
                client.delete_collection(collection_name)
                print(f"ChromaDB collection '{collection_name}' deleted")
            except Exception:
                pass
        except Exception:
            # Fallback: delete directory
            path = os.path.join(CHROMA_DIR, collection_name)
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
            print("FAISS index deleted")


def as_retriever(vectorstore, k=FINAL_TOP_K, search_type="similarity"):
    """Get a retriever from a vector store, regardless of backend."""
    return vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": k})


def similarity_search(vectorstore, query: str, k=FINAL_TOP_K):
    """Search a vector store, returning (docs, scores) regardless of backend."""
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
        docs = [r[0] for r in results]
        scores = [float(r[1]) for r in results]
        return docs, scores
    except Exception:
        # Fallback for stores that don't support scores
        docs = vectorstore.similarity_search(query, k=k)
        return docs, [0.5] * len(docs)
