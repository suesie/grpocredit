"""MATH / GSM8K verifier.

Thin wrapper around HuggingFace's `math-verify` package — a sympy-backed
parser/verifier built for exactly this problem class. Plan §3: "copy VinePPO's
math_verify.py verbatim. Do not rewrite; it took them effort to get edge cases
right." The `math-verify` pip package captures the same logic plus OlympiadBench
and AIME extensions.

Fallbacks: GSM8K answers are usually a single integer; when the boxed extract
is empty, we regex-search for 'The answer is X' and 'Final answer: X'.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_ANSWER_IS_RE = re.compile(
    r"(?:answer\s+is|final\s+answer[:\s]+|=\s*)\s*([-\d.,/\\\w{}()^+*\s]+?)(?:[.\n]|$)",
    re.IGNORECASE,
)


@dataclass
class VerifierResult:
    correct: bool
    extracted: str
    method: str  # 'boxed' | 'answer_is' | 'gsm8k_hash' | 'fallback'


def extract_final_answer(response: str) -> tuple[str, str]:
    """Extract a final-answer string from a model response, with method tag.

    Priority: last \\boxed{} → 'The answer is X' → last numeric token.
    """
    if not response:
        return "", "fallback"

    # 1. last \boxed{}
    matches = list(_BOXED_RE.finditer(response))
    if matches:
        return matches[-1].group(1).strip(), "boxed"

    # 2. "The answer is X."
    m = _ANSWER_IS_RE.search(response)
    if m:
        return m.group(1).strip().strip(".,"), "answer_is"

    # 3. GSM8K-style #### NNN
    if "####" in response:
        tail = response.rsplit("####", 1)[-1].strip()
        tok = tail.split()[0].replace(",", "") if tail else ""
        if tok:
            return tok, "gsm8k_hash"

    # 4. last numeric token (weak fallback, usually means the verifier will fail)
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", response)
    if nums:
        return nums[-1], "fallback"
    return "", "fallback"


@dataclass
class MathVerifier:
    """Thread-safe wrapper around `math_verify.verify`.

    `score_answer(response, ground_truth)` returns (is_correct, extracted_answer).
    """

    use_math_verify: bool = True

    def __post_init__(self) -> None:
        self._parse = None
        self._verify = None
        if self.use_math_verify:
            try:
                from math_verify import parse, verify

                self._parse = parse
                self._verify = verify
            except ImportError:
                log.warning(
                    "math_verify not installed; falling back to naive string match. "
                    "Install via `pip install math-verify`."
                )
                self.use_math_verify = False

    def score(self, response: str, ground_truth: str) -> VerifierResult:
        extracted, method = extract_final_answer(response)
        if not extracted:
            return VerifierResult(correct=False, extracted="", method=method)

        if self.use_math_verify and self._parse and self._verify:
            try:
                gold = self._parse(f"${ground_truth}$")
                pred = self._parse(f"${extracted}$")
                ok = bool(self._verify(gold, pred))
                return VerifierResult(correct=ok, extracted=extracted, method=method)
            except Exception as e:
                log.debug("math_verify failed (gold=%r pred=%r): %s", ground_truth, extracted, e)
                # fall through to naive
        return VerifierResult(
            correct=_naive_equal(extracted, ground_truth), extracted=extracted, method=method
        )


def _naive_equal(a: str, b: str) -> bool:
    """Naive normalisation: strip whitespace/commas/dollar signs, compare casefolded."""
    def norm(x: str) -> str:
        return re.sub(r"[\s,\\$]+", "", x).strip().lower()

    return norm(a) == norm(b)


# Module-level convenience — avoids re-instantiating the wrapper in tight loops.
_DEFAULT_VERIFIER: MathVerifier | None = None


def score_answer(response: str, ground_truth: str) -> VerifierResult:
    global _DEFAULT_VERIFIER
    if _DEFAULT_VERIFIER is None:
        _DEFAULT_VERIFIER = MathVerifier()
    return _DEFAULT_VERIFIER.score(response, ground_truth)
