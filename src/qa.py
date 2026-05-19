"""QA pipeline — decomposition, multi-turn history, metadata-aware retrieval."""
import os, json

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from openai import OpenAI

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    GENERATION_MODEL,
    VALIDATION_MODEL,
    ENABLE_QUERY_REWRITE,
    ENABLE_VALIDATION,
    FINAL_TOP_K,
)
from .retrieve import HybridRetriever
from .prompts import build_chat_messages
from .validator import ResponseValidator


def _format_history(history: list[dict]) -> str:
    """Format conversation history for prompt injection.

    Args:
        history: list of {"role": "user"/"assistant", "content": "..."}

    Returns:
        formatted string like:
        User: question 1
        Assistant: answer 1
        User: question 2
    """
    if not history:
        return "(no previous turns)"
    lines = []
    for turn in history[-6:]:  # last 3 exchanges max
        role = "User" if turn["role"] == "user" else "Assistant"
        content = turn["content"]
        # Truncate long messages
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class QAPipeline:
    """End-to-end QA pipeline: decomposition, multi-turn rewriting,
    hybrid retrieval with metadata strategy, and answer generation."""

    def __init__(self, retriever=None, backend=None):
        self._gen_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self._val_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self._retriever = retriever or HybridRetriever(backend=backend)
        self._validator = ResponseValidator() if ENABLE_VALIDATION else None

    # ---- Question decomposition ----

    def decompose_question(self, query: str) -> list[str]:
        """Decompose a complex question into simpler sub-questions.

        Uses the decompose prompt template. If the question is already simple,
        returns a single-element list.
        """
        try:
            messages = build_chat_messages("decompose", query=query)
            response = self._gen_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            sub_questions = result.get("sub_questions", [query])
            if not sub_questions:
                return [query]
            return sub_questions
        except Exception:
            return [query]

    # ---- Query rewriting (with multi-turn history) ----

    def rewrite_query(self, query: str, history: list[dict] = None) -> str:
        """Rewrite user query for better retrieval, using conversation history.

        When history is provided, uses it to resolve ambiguous references:
        "how to configure it" + history about AD9959 → "AD9959 SPI configuration"
        """
        if not ENABLE_QUERY_REWRITE:
            return query

        history_str = _format_history(history) if history else "(no previous turns)"

        try:
            messages = build_chat_messages("rewrite", query=query, history=history_str)
            response = self._gen_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=200,
            )
            rewritten = response.choices[0].message.content.strip()
            if rewritten and len(rewritten) > 3:
                return rewritten
        except Exception:
            pass
        return query

    # ---- Answer generation ----

    def generate_answer(self, query: str, docs) -> dict:
        """Generate an answer from retrieved documents."""
        if not docs:
            return {
                "answer": "No relevant documents found. Upload documents and build an index first.",
                "sources": [],
            }

        context_parts = []
        sources = []
        seen = set()
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", doc.metadata.get("filename", "unknown"))
            if source not in seen:
                seen.add(source)
                sources.append(source)
            context_parts.append(f"[Doc{i+1} from {source}]:\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        try:
            messages = build_chat_messages("qa", query=query, context=context)
            response = self._gen_client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=messages,
                temperature=0.1,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"

        return {"answer": answer, "sources": sources, "context": docs}

    def _synthesize_answers(self, query: str, sub_results: list[dict]) -> str:
        """Synthesize multiple sub-answers into one cohesive response."""
        if len(sub_results) == 1:
            return sub_results[0]["answer"]

        parts = []
        for i, r in enumerate(sub_results):
            parts.append(f"## {r['sub_question']}\n\n{r['answer']}")

        combined = "\n\n---\n\n".join(parts)

        # Collect all unique sources
        all_sources = []
        seen = set()
        for r in sub_results:
            for s in r.get("sources", []):
                if s not in seen:
                    seen.add(s)
                    all_sources.append(s)

        return combined, all_sources

    # ---- Main API (simple + decomposed + multi-turn) ----

    def _retry_generate(self, question: str, docs) -> str:
        """Re-generate answer using the validation (pro) model for higher quality."""
        if not docs:
            return ""
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", doc.metadata.get("filename", "unknown"))
            context_parts.append(f"[Doc{i+1} from {source}]:\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        try:
            messages = build_chat_messages("qa", query=question, context=context)
            response = self._val_client.chat.completions.create(
                model=VALIDATION_MODEL,
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    def ask(self, query: str, top_k: int = None,
            history: list[dict] = None) -> dict:
        """Full pipeline: rewrite → decompose → retrieve → generate → validate → synthesize.

        Dual-model flow:
        1. GENERATION_MODEL (fast): produces draft answer
        2. VALIDATION_MODEL (pro): evaluates faithfulness/hallucination/completeness
        3. If score < VALIDATION_THRESHOLD, retry with VALIDATION_MODEL

        Args:
            query: user question
            top_k: number of documents to retrieve per sub-question
            history: conversation history for de-referencing
        """
        if top_k is None:
            top_k = FINAL_TOP_K

        # Step 1: Rewrite with history context (de-reference pronouns etc.)
        rewritten = self.rewrite_query(query, history=history)

        # Step 2: Decompose complex question into sub-questions
        sub_questions = self.decompose_question(rewritten)

        # Step 3: For each sub-question, retrieve + generate + validate
        sub_results = []
        for sq in sub_questions:
            retrieval = self._retriever.retrieve_with_metadata_strategy(sq, top_k=top_k)
            docs = retrieval["docs"]
            gen = self.generate_answer(sq, docs)

            # Step 3.5: Validate the generated answer
            validation = None
            retried = False
            if self._validator and docs:
                validated = self._validator.validate_and_optionally_retry(
                    sq, gen["answer"], docs,
                    retry_fn=lambda q, d: self._retry_generate(q, d),
                )
                gen["answer"] = validated["answer"]
                validation = validated["validation"]
                retried = validated["retried"]

            gen["sub_question"] = sq
            gen["applied_filter"] = retrieval["applied_filter"]
            gen["validation"] = validation
            gen["retried"] = retried
            sub_results.append(gen)

        # Step 4: Synthesize
        if len(sub_results) == 1:
            result = sub_results[0]
            result["rewritten_query"] = rewritten if rewritten != query else None
            result["is_decomposed"] = False
            return result

        answer, all_sources = self._synthesize_answers(query, sub_results)
        return {
            "answer": answer,
            "sources": all_sources,
            "rewritten_query": rewritten if rewritten != query else None,
            "is_decomposed": True,
            "sub_questions": sub_questions,
            "sub_results": sub_results,
        }

    def retrieve_only(self, query: str, top_k: int = None,
                      history: list[dict] = None):
        """Retrieve documents without generating an answer (for debugging)."""
        if top_k is None:
            top_k = FINAL_TOP_K
        rewritten = self.rewrite_query(query, history=history)
        retrieval = self._retriever.retrieve_with_metadata_strategy(rewritten, top_k=top_k)
        return {
            "docs": retrieval["docs"],
            "rewritten_query": rewritten if rewritten != query else None,
            "applied_filter": retrieval["applied_filter"],
        }


def create_qa_pipeline(retriever=None):
    return QAPipeline(retriever=retriever)


if __name__ == "__main__":
    pipeline = QAPipeline()

    # Test simple
    print("=== Simple question ===")
    result = pipeline.ask("Python命名规范是什么")
    print(f"Decomposed: {result.get('is_decomposed', False)}")
    print(f"Rewritten: {result.get('rewritten_query', '(none)')}")
    print(f"A: {result['answer'][:200]}")

    # Test complex
    print("\n=== Complex question ===")
    result = pipeline.ask("Python装饰器和上下文管理器有什么区别，分别怎么使用")
    print(f"Decomposed: {result.get('is_decomposed', False)}")
    if result.get("sub_questions"):
        for sq in result["sub_questions"]:
            print(f"  - {sq}")

    # Test multi-turn
    print("\n=== Multi-turn ===")
    history = [
        {"role": "user", "content": "AD9959芯片支持哪些配置模式"},
        {"role": "assistant", "content": "AD9959支持单音模式、多音模式和线性调频模式。"},
    ]
    result = pipeline.ask("怎么配置它", history=history)
    print(f"Rewritten: {result.get('rewritten_query', '(none)')}")
    print(f"A: {result['answer'][:200]}")
