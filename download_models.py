"""Pre-download embedding and reranker models for offline use.

Usage:
    python download_models.py                    # download to default cache
    python download_models.py --local ./models   # download to local directory
"""
import os, sys, argparse

# Set mirror before anything else
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer, CrossEncoder

MODELS = {
    "embedding": "BAAI/bge-small-zh-v1.5",  # Chinese embedding model
    # English-only fallback: set EMBEDDING_MODEL=all-MiniLM-L6-v2 in .env
    "reranker": "BAAI/bge-reranker-v2-m3",  # Chinese-friendly multilingual reranker
}


def download_to_cache():
    """Download models to the default HuggingFace cache."""
    print("Downloading embedding model:", MODELS["embedding"])
    SentenceTransformer(MODELS["embedding"])
    print("OK")

    print("Downloading reranker model:", MODELS["reranker"])
    CrossEncoder(MODELS["reranker"])
    print("OK")


def download_to_local(target_dir: str):
    """Download models and save to a local directory."""
    os.makedirs(target_dir, exist_ok=True)

    for name, model_id in MODELS.items():
        safe_name = model_id.replace("/", "_")
        save_path = os.path.join(target_dir, safe_name)

        if os.path.isdir(save_path):
            print(f"{name}: already exists at {save_path}, skipping")
            continue

        print(f"Downloading {name} model: {model_id} -> {save_path}")
        if name == "embedding":
            model = SentenceTransformer(model_id)
        else:
            model = CrossEncoder(model_id)
        model.save(save_path)
        print(f"  Saved to {save_path}")

    print(f"\nDone! Set LOCAL_MODEL_DIR={target_dir} in .env or environment.")
    print(f"Example: echo 'LOCAL_MODEL_DIR={target_dir}' >> .env")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-download models for RAG Code QA")
    parser.add_argument("--local", "-l", help="Download to local directory instead of HF cache")
    args = parser.parse_args()

    if args.local:
        download_to_local(os.path.abspath(args.local))
    else:
        download_to_cache()
        print("\nModels cached by HuggingFace. Set HF_ENDPOINT for future use if needed.")
