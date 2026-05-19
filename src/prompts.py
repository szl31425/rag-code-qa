"""Prompt loader — reads YAML templates and renders with variables + few-shot."""
import os
import yaml

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


def _load_yaml(name: str) -> dict:
    """Load a YAML prompt file."""
    path = os.path.join(PROMPTS_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_prompt(name: str, **variables) -> dict:
    """Render a prompt template with variables and return messages for LLM API.

    Args:
        name: prompt file name without .yaml extension (e.g., "rewrite", "qa")
        **variables: template variables to substitute (e.g., query="...", context="...")

    Returns:
        dict with keys:
          - messages: list of {"role": str, "content": str} for OpenAI API
          - system: system prompt string
          - user: user message string
    """
    template = _load_yaml(name)

    # Render system prompt
    system = template.get("system", "")

    # Build messages with few-shot examples
    messages = []
    for example in template.get("few_shot", []):
        messages.append({"role": "user", "content": example["user"].strip()})

        assistant_content = example["assistant"].strip()
        # Some few-shot examples include context in the assistant view — skip if not
        messages.append({"role": "assistant", "content": assistant_content})

    # Render the final user message
    user_template = template.get("user_template", "{query}")
    user_message = user_template.format(**variables)

    # Handle the special {context} variable for QA — inject into system prompt
    if "context" in variables:
        system = system.replace("{context}", variables["context"])

    return {
        "messages": messages,
        "system": system,
        "user": user_message,
    }


def build_chat_messages(name: str, **variables) -> list[dict]:
    """Build a complete list of chat messages for OpenAI/DeepSeek API.

    Returns a list of {"role": ..., "content": ...} ready to pass to the API.
    """
    rendered = render_prompt(name, **variables)

    messages = [{"role": "system", "content": rendered["system"]}]
    messages.extend(rendered["messages"])  # few-shot examples
    messages.append({"role": "user", "content": rendered["user"]})

    return messages


def list_templates():
    """List available prompt templates."""
    if not os.path.exists(PROMPTS_DIR):
        return []
    return sorted(
        f.replace(".yaml", "")
        for f in os.listdir(PROMPTS_DIR)
        if f.endswith(".yaml")
    )
