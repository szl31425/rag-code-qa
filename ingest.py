"""Backward-compatible wrapper — delegates to src.ingest."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from src.ingest import load_documents, build_index, DOCS_DIR, INDEX_DIR

if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        sample = os.path.join(DOCS_DIR, "sample_python_guide.md")
        os.makedirs(DOCS_DIR, exist_ok=True)
        with open(sample, "w", encoding="utf-8") as sf:
            sf.write(
                "# Python Coding Guide\n\n"
                "- Variables: snake_case\n"
                "- Constants: UPPER_CASE\n"
                "- Functions with docstrings\n"
                "- try/except for error handling\n"
            )
        docs = load_documents()
    build_index(docs)
    print("Done!")
