"""Backward-compatible wrapper — delegates to src.qa and src.retrieve."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "faiss_index")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

SYSTEM_PROMPT = """You are a code knowledge base assistant. Answer based on the retrieved context below.

Rules:
1. If context has relevant info, base your answer on it
2. If not enough info, say "This information is not in the current knowledge base"
3. Give specific code examples when relevant
4. Keep answers concise and accurate

Retrieved context:
{context}"""


def load_vectorstore():
    if not os.path.exists(INDEX_DIR):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


def create_qa_chain(vectorstore=None):
    from src.qa import QAPipeline
    from src.retrieve import HybridRetriever

    retriever = HybridRetriever(vectorstore=vectorstore)
    pipeline = QAPipeline(retriever=retriever)

    class LegacyWrapper:
        def invoke(self, inputs):
            result = pipeline.ask(inputs.get("input") or inputs.get("question"))
            return {
                "answer": result["answer"],
                "context": result.get("context", []),
                "sources": result.get("sources", []),
            }

    return LegacyWrapper()


def ask(question, chain=None):
    if chain is None:
        chain = create_qa_chain()
    result = chain.invoke({"input": question})
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    if result.get("sources"):
        print(f"\nSources ({len(result['sources'])}):")
        for i, s in enumerate(result["sources"]):
            print(f"  [{i+1}] {s}")
    return result


if __name__ == "__main__":
    chain = create_qa_chain()
    ask("What are Python naming conventions?", chain)
