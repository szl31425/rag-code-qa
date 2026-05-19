"""Configuration for RAG Code QA system."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

# Embedding & models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# For Chinese documents, set in .env: EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
# Then run: python download_models.py --local ./models
# Set LOCAL_MODEL_DIR to a pre-downloaded path like ./models/ to skip HF download
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# For English-only: set RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 in .env
LLM_MODEL = "deepseek-chat"

# Chunking strategies: "fixed_size", "semantic", "code"
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "fixed_size")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Hybrid retrieval
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "10"))
SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", "10"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "4"))
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
USE_HYBRID = os.getenv("USE_HYBRID", "true").lower() == "true"

# Query rewriting
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"

# DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Dual-model setup for Response Validation
# Fast/cheap model for answer generation (e.g., deepseek-chat / deepseek-v4-flash)
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "deepseek-chat")
# Powerful model for validation/evaluation (e.g., deepseek-reasoner / deepseek-v4-pro)
VALIDATION_MODEL = os.getenv("VALIDATION_MODEL", "deepseek-chat")

# Validation settings
ENABLE_VALIDATION = os.getenv("ENABLE_VALIDATION", "true").lower() == "true"
VALIDATION_THRESHOLD = float(os.getenv("VALIDATION_THRESHOLD", "60"))
# If validation score < threshold, retry with the validation model
RETRY_WITH_PRO_MODEL = os.getenv("RETRY_WITH_PRO_MODEL", "true").lower() == "true"

# HuggingFace mirror for China (set before any HF imports)
HF_ENDPOINT = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["HF_HUB_ENDPOINT"] = HF_ENDPOINT
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
