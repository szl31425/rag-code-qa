"""RAG evaluation pipeline using RAGAS metrics."""
import os, json
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL
from .qa import QAPipeline


EVAL_DATASET = [
    {
        "question": "What are the Python naming conventions for variables?",
        "reference_answer": "Python variables should use snake_case naming, with lowercase letters and underscores between words.",
    },
    {
        "question": "How should errors be handled in Python?",
        "reference_answer": "Use try/except blocks for expected errors, avoid bare except clauses, and log exceptions with traceback information.",
    },
    {
        "question": "What are the best practices for writing functions?",
        "reference_answer": "Functions should have docstrings, be kept under 50 lines, and use type hints for function signatures.",
    },
    {
        "question": "How should constants be named?",
        "reference_answer": "Constants should use UPPER_CASE naming with underscores between words.",
    },
    {
        "question": "How should classes be named in Python?",
        "reference_answer": "Python classes should use PascalCase naming convention, with each word capitalized and no underscores.",
    },
]


def run_ragas_eval(pipeline=None, dataset=None):
    """Evaluate the RAG pipeline using RAGAS metrics."""
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError:
        print("RAGAS not installed. Run: pip install ragas")
        return _run_simple_eval(pipeline, dataset)

    if pipeline is None:
        pipeline = QAPipeline()
    if dataset is None:
        dataset = EVAL_DATASET

    eval_data = {"question": [], "answer": [], "contexts": [], "reference": []}

    print(f"Evaluating {len(dataset)} questions...")
    for item in dataset:
        result = pipeline.ask(item["question"])
        contexts = [doc.page_content for doc in result.get("context", [])]

        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append(contexts)
        eval_data["reference"].append(item["reference_answer"])
        print(f"  Q: {item['question'][:60]}...")

    try:
        ds = EvaluationDataset.from_dict(eval_data)
        results = evaluate(ds, metrics=[faithfulness, answer_relevancy,
                                         context_precision, context_recall])
        print("\n=== RAGAS Evaluation Results ===")
        for metric, score in results.items():
            print(f"  {metric}: {score:.4f}")
        return results
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        return _run_simple_eval(pipeline, dataset)


def _run_simple_eval(pipeline=None, dataset=None):
    """Fallback: simple keyword-overlap evaluation."""
    if pipeline is None:
        pipeline = QAPipeline()
    if dataset is None:
        dataset = EVAL_DATASET

    results = []
    print(f"\nRunning simple evaluation on {len(dataset)} questions...")

    for item in dataset:
        result = pipeline.ask(item["question"])
        answer = result["answer"].lower()
        reference = item["reference_answer"].lower()

        # Simple word overlap score
        ref_words = set(reference.split())
        ans_words = set(answer.split())
        overlap = len(ref_words & ans_words) / len(ref_words) if ref_words else 0

        # Source coverage
        has_sources = len(result.get("sources", [])) > 0
        doc_count = len(result.get("context", []))

        results.append({
            "question": item["question"],
            "answer": answer[:200],
            "reference": reference[:200],
            "word_overlap": round(overlap, 3),
            "has_sources": has_sources,
            "docs_retrieved": doc_count,
        })
        print(f"  Q: {item['question'][:60]}... overlap={overlap:.3f} sources={has_sources} docs={doc_count}")

    # Aggregate
    avg_overlap = sum(r["word_overlap"] for r in results) / len(results)
    source_rate = sum(1 for r in results if r["has_sources"]) / len(results)
    avg_docs = sum(r["docs_retrieved"] for r in results) / len(results)

    print(f"\n=== Evaluation Summary ===")
    print(f"  Avg Word Overlap:  {avg_overlap:.3f}")
    print(f"  Source Coverage:   {source_rate:.1%}")
    print(f"  Avg Docs Retrieved: {avg_docs:.1f}")

    return {
        "results": results,
        "summary": {
            "avg_word_overlap": round(avg_overlap, 3),
            "source_coverage": source_rate,
            "avg_docs_retrieved": round(avg_docs, 1),
        },
    }


def generate_test_questions(num_questions=5):
    """Use LLM to generate evaluation questions from the knowledge base."""
    from openai import OpenAI
    from .retrieve import HybridRetriever

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    retriever = HybridRetriever()

    # Sample some document chunks
    test_query = "key concepts and features"
    docs = retriever.retrieve(test_query, top_k=6)

    if not docs:
        print("No documents in index. Run ingest.py first.")
        return []

    context = "\n\n".join([d.page_content[:500] for d in docs])

    prompt = f"""Based on the following document excerpts, generate {num_questions} diverse test questions
that a user might ask about this content. For each question, also write the expected answer.

Output JSON array:
[
  {{"question": "...", "reference_answer": "..."}},
  ...
]

Document excerpts:
{context}"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    try:
        questions = json.loads(response.choices[0].message.content)
        if isinstance(questions, dict):
            questions = questions.get("questions", [])
        print(f"Generated {len(questions)} test questions")
        return questions
    except Exception as e:
        print(f"Failed to generate questions: {e}")
        return []


if __name__ == "__main__":
    pipeline = QAPipeline()
    run_ragas_eval(pipeline)
