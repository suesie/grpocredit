"""Dataset loaders: GSM8K, MATH, AIME24, OlympiadBench, MATH-500.

Every loader returns a list of `PromptRecord` — uniform schema keeps downstream
cascade / advantage code oblivious to which benchmark a prompt came from.

Prompt templating lives in `format_prompt`. Qwen2.5-Math-Instruct models were
trained with the Qwen chat template, so for a from-scratch RL sanity run we
stick to the chat template by default. The `math_instruct` template mirrors
VinePPO's prompt format for GRPO-style reasoning.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class PromptRecord:
    prompt_id: str
    question: str
    ground_truth_answer: str  # normalised extract: "42", "\\frac{1}{2}", etc.
    ground_truth_raw: str = ""  # original solution string for context
    source: str = ""
    level: str = ""  # MATH difficulty level if available
    subject: str = ""  # MATH subject if available
    extra: dict[str, Any] = field(default_factory=dict)


# ─── chat / prompt templates ─────────────────────────────────────────────
SYSTEM_MATH = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def _apply_qwen_chat_template(tokenizer: Any, question: str) -> str:
    """Use the model's native chat template if the tokenizer exposes it."""
    messages = [
        {"role": "system", "content": SYSTEM_MATH},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for tokenizers without chat templates
        return f"System: {SYSTEM_MATH}\n\nUser: {question}\n\nAssistant: "


def _apply_vineppo_math_task(tokenizer: Any, question: str) -> str:
    """VinePPO's SFT-time template for rho-1b-sft-* and deepseekmath-7b-sft-*.

    Verbatim from VinePPO's `configs/prompt_library/generic_{GSM8K,MATH}_
    step_by_step.jsonnet` and the matching SFT configs. Must be paired with
    VinePPO's eval-time sampling (`temperature: 0.35, top_p: 0.9, stop:
    ["\\n\\n\\nProblem:"]`) or pass@1 collapses — see runbook §2.2. The
    tokenizer arg is unused (base LM, no chat template) and kept only for
    `TEMPLATES` signature uniformity.
    """
    del tokenizer
    return f"[MATH_TASK] Problem:\n{question}\n\nSolution:"


TEMPLATES = {
    "math_instruct": _apply_qwen_chat_template,
    "vineppo_math_task": _apply_vineppo_math_task,
}


def format_prompt(question: str, tokenizer: Any, template: str = "math_instruct") -> str:
    fn = TEMPLATES.get(template)
    if fn is None:
        raise ValueError(f"Unknown template {template}; known: {list(TEMPLATES)}")
    return fn(tokenizer, question)


# ─── answer extraction from solution strings ────────────────────────────
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_GSM8K_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


def _extract_boxed(text: str) -> str:
    """Return the contents of the last \\boxed{...} in `text`, or ''."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""


def _extract_gsm8k_answer(solution: str) -> str:
    m = _GSM8K_ANSWER_RE.search(solution)
    return m.group(1).replace(",", "").strip() if m else ""


# ─── loaders ─────────────────────────────────────────────────────────────
def _load_hf(name: str, config: str | None = None, split: str = "train") -> Any:
    from datasets import load_dataset

    return load_dataset(name, config) if config else load_dataset(name, split=split)


def load_gsm8k(split: str = "train", n: int | None = None) -> list[PromptRecord]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    records: list[PromptRecord] = []
    for i, row in enumerate(ds):
        ans = _extract_gsm8k_answer(row["answer"])
        records.append(
            PromptRecord(
                prompt_id=f"gsm8k/{split}/{i}",
                question=row["question"],
                ground_truth_answer=ans,
                ground_truth_raw=row["answer"],
                source="gsm8k",
            )
        )
        if n is not None and len(records) >= n:
            break
    return records


def load_math(split: str = "train", n: int | None = None) -> list[PromptRecord]:
    """Hendrycks MATH. Uses HuggingFaceH4's mirror which is stable.

    Falls back to `lighteval/MATH` if the primary source errors.
    """
    from datasets import load_dataset

    for name in ("HuggingFaceH4/MATH", "lighteval/MATH", "EleutherAI/hendrycks_math"):
        try:
            ds = load_dataset(name, split=split)
            break
        except Exception as e:  # pragma: no cover
            log.warning("load_math: %s failed (%s); trying next mirror", name, e)
    else:
        raise RuntimeError("No MATH mirror could be loaded")

    records: list[PromptRecord] = []
    for i, row in enumerate(ds):
        q = row.get("problem") or row.get("question") or ""
        sol = row.get("solution") or row.get("answer") or ""
        ans = _extract_boxed(sol) or _extract_boxed(row.get("answer", ""))
        records.append(
            PromptRecord(
                prompt_id=f"math/{split}/{i}",
                question=q,
                ground_truth_answer=ans,
                ground_truth_raw=sol,
                source="math",
                level=str(row.get("level", "")),
                subject=str(row.get("type", row.get("subject", ""))),
            )
        )
        if n is not None and len(records) >= n:
            break
    return records


def load_math500(n: int | None = None) -> list[PromptRecord]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    records: list[PromptRecord] = []
    for i, row in enumerate(ds):
        q = row["problem"]
        ans = row.get("answer") or _extract_boxed(row.get("solution", ""))
        records.append(
            PromptRecord(
                prompt_id=f"math500/test/{i}",
                question=q,
                ground_truth_answer=str(ans).strip(),
                ground_truth_raw=row.get("solution", ""),
                source="math500",
                level=str(row.get("level", "")),
                subject=str(row.get("subject", "")),
            )
        )
        if n is not None and len(records) >= n:
            break
    return records


def load_aime24(n: int | None = None) -> list[PromptRecord]:
    from datasets import load_dataset

    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    records: list[PromptRecord] = []
    for i, row in enumerate(ds):
        q = row["problem"]
        ans = str(row["answer"]).strip()
        records.append(
            PromptRecord(
                prompt_id=f"aime24/{i}",
                question=q,
                ground_truth_answer=ans,
                source="aime24",
            )
        )
        if n is not None and len(records) >= n:
            break
    return records


def load_olympiadbench(n: int | None = None) -> list[PromptRecord]:
    """OlympiadBench: English text-only subset with numeric/closed-form answers.

    Filters to `maths` subject and `free-form` answer type for verifier compatibility.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train"
    )
    records: list[PromptRecord] = []
    for i, row in enumerate(ds):
        q = row["question"]
        ans = row.get("final_answer")
        if isinstance(ans, list) and ans:
            ans = ans[0]
        if not ans:
            continue
        records.append(
            PromptRecord(
                prompt_id=f"olympiad/{i}",
                question=q,
                ground_truth_answer=str(ans).strip(),
                source="olympiadbench",
            )
        )
        if n is not None and len(records) >= n:
            break
    return records


# ─── dispatcher ─────────────────────────────────────────────────────────
_LOADERS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "math500": load_math500,
    "aime24": load_aime24,
    "olympiadbench": load_olympiadbench,
}


def load_prompts(
    name: str,
    split: str = "train",
    n: int | None = None,
) -> list[PromptRecord]:
    """Dispatch to the appropriate loader by dataset name."""
    name = name.lower()
    fn = _LOADERS.get(name)
    if fn is None:
        raise ValueError(f"Unknown dataset {name}; known: {list(_LOADERS)}")
    # Dispatch based on whether the loader accepts `split`
    if name in {"gsm8k", "math"}:
        return fn(split=split, n=n)
    return fn(n=n)
