"""Response Validation — dual-model: fast model generates, pro model evaluates."""
import os, json, re
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from openai import OpenAI

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    GENERATION_MODEL,
    VALIDATION_MODEL,
    VALIDATION_THRESHOLD,
    RETRY_WITH_PRO_MODEL,
)
from .prompts import build_chat_messages


class ResponseValidator:
    """Validates generated answers against retrieved context using a second LLM.

    Architecture:
      Generation Model (GENERATION_MODEL, e.g. deepseek-chat):
        Fast, cost-effective → produces draft answers

      Validation Model (VALIDATION_MODEL, e.g. deepseek-reasoner):
        More capable, slower → audits answers for faithfulness/hallucination

    If validation fails and RETRY_WITH_PRO_MODEL=True, re-generates with
    the validation model for higher quality.
    """

    def __init__(self):
        # Two separate clients for clarity — they can point to the same API
        # with different model parameters
        self._gen_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self._val_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def validate(self, question: str, answer: str, context_docs: list) -> dict:
        """Evaluate an answer's quality against the context.

        Returns a dict with overall_score, faithfulness, completeness, relevance,
        verdict (pass/fail/uncertain), hallucinations, missing_aspects, suggestion.
        """
        # Build compact context string
        context_parts = []
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get("source", doc.metadata.get("filename", "unknown"))
            context_parts.append(f"[Doc{i+1} from {source}]:\n{doc.page_content[:800]}")
        context = "\n\n".join(context_parts)

        try:
            messages = build_chat_messages("validate",
                                           question=question,
                                           answer=answer,
                                           context=context)
            response = self._val_client.chat.completions.create(
                model=VALIDATION_MODEL,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            return {
                "overall_score": 0,
                "faithfulness": 0,
                "completeness": 0,
                "relevance": 0,
                "verdict": "uncertain",
                "hallucinations": [],
                "missing_aspects": [],
                "suggestion": f"Validation error: {str(e)}",
            }

    def validate_and_optionally_retry(
        self, question: str, answer: str, context_docs: list,
        retry_fn=None,
    ) -> dict:
        """Validate answer, optionally retry with pro model if score is low.

        Args:
            question: original user question
            answer: generated answer to validate
            context_docs: retrieved documents used for generation
            retry_fn: callable(question, context_docs) -> answer string.
                      If None, retries are skipped.

        Returns:
            {
                "answer": final answer (original or retried),
                "validation": {...},
                "retried": bool,
            }
        """
        validation = self.validate(question, answer, context_docs)
        score = validation.get("overall_score", 0)

        result = {
            "answer": answer,
            "validation": validation,
            "retried": False,
        }

        # If answer is poor and we can retry with the pro model
        if (RETRY_WITH_PRO_MODEL and retry_fn is not None
                and score < VALIDATION_THRESHOLD):
            try:
                improved = retry_fn(question, context_docs)
                if improved and improved != answer:
                    # Re-validate the improved answer
                    improved_validation = self.validate(question, improved, context_docs)
                    improved_score = improved_validation.get("overall_score", 0)

                    # Only accept if actually better
                    if improved_score > score:
                        result["answer"] = improved
                        result["validation"] = improved_validation
                        result["retried"] = True
                        result["original_score"] = score
                        result["improved_score"] = improved_score
            except Exception:
                pass

        return result

    def format_validation_badge(self, validation: dict) -> str:
        """Generate a human-readable validation summary for UI display."""
        score = validation.get("overall_score", 0)
        verdict = validation.get("verdict", "uncertain")
        hallucinations = validation.get("hallucinations", [])
        missing = validation.get("missing_aspects", [])

        emoji = {"pass": "✅", "fail": "❌", "uncertain": "⚠️"}.get(verdict, "⚠️")

        lines = [
            f"{emoji} **Validation**: {score}/100 ({verdict})",
            f"- Faithfulness: {validation.get('faithfulness', 0)}/100",
            f"- Completeness: {validation.get('completeness', 0)}/100",
            f"- Relevance: {validation.get('relevance', 0)}/100",
        ]

        if hallucinations:
            lines.append(f"- ⚠️ Potential hallucinations: {len(hallucinations)}")
            for h in hallucinations[:3]:
                lines.append(f"  • {h}")

        if missing:
            lines.append(f"- 📝 Missing aspects: {len(missing)}")
            for m in missing[:3]:
                lines.append(f"  • {m}")

        return "\n".join(lines)
