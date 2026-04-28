"""MATH / GSM8K verifier.

Thin wrapper around HuggingFace's `math-verify` package — a sympy-backed
parser/verifier built for exactly this problem class. Plan §3: "copy VinePPO's
math_verify.py verbatim. Do not rewrite; it took them effort to get edge cases
right." The `math-verify` pip package captures the same logic plus OlympiadBench
and AIME extensions.

Extraction strategy
-------------------

`extract_final_answer` runs a **priority-ordered registry of extractors**,
returning the first one that produces a non-empty extract along with a
method tag that identifies which extractor fired. Each extractor is a
pure `(response: str) -> str` callable; adding a new output convention
(e.g., for a new model family) is five lines — write the extractor, append
it to `_EXTRACTORS` at the right priority, add a test.

Current extractors, in priority order (most-authoritative first):

    1. `gsm8k_hash`      — last `#### X` (canonical GSM8K final-answer marker)
    2. `answer_tag`      — last `<answer>X</answer>`  (DeepSeek-R1 convention)
    3. `boxed`           — last `\\boxed{X}` (canonical MATH marker)
    4. `answer_is`       — "(the )?(final )?answer (is|:|=) X" prose heuristic
    5. `fallback`        — last numeric token (weak; method tag is the signal
                           to downstream code that this one is untrustworthy)

Not covered — would require dedicated extractors, not tweaks to these:

    * multiple-choice benchmarks (MMLU, ARC, GPQA)  — answer is a letter
    * code-generation (HumanEval, MBPP)             — answer is executable
    * Lean / symbolic proofs without \\boxed{}       — needs a proof checker

For those, write a new `MathVerifier`-alike (e.g. `MultipleChoiceVerifier`)
rather than bolting onto this registry. The oracle pipeline only assumes
`verifier.score(response, ground_truth) -> VerifierResult`, so swapping is
one config knob.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE
)
# Scoped to "(the )?(final )?answer (is|:|=) …" phrasing only. An earlier
# version also matched bare `=\s*<value>`, which fired on every intermediate
# CoT step ("48/2 = 24 …") and mis-graded rho-1b-sft-GSM8K — see runbook §2.4.
_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*"
    r"([-\d.,/\\\w{}()^+*\s]+?)(?:[.\n]|$)",
    re.IGNORECASE,
)
_GSM8K_HASH_NUM_RE = re.compile(r"-?\d+(?:[.,/]\d+)*")


@dataclass
class VerifierResult:
    correct: bool
    extracted: str
    # Tag of the extractor that fired. Keep in sync with `_EXTRACTORS` below.
    # Downstream aggregators use this to diagnose grader drift — e.g. a
    # GSM8K-config rollout distribution suddenly dominated by `answer_is`
    # instead of `gsm8k_hash` is the signature of a template mismatch.
    method: str


# ─── Extractors ─────────────────────────────────────────────────────────
#
# Each extractor is a pure function `response -> extracted_str`. Returning
# the empty string means "this extractor doesn't match; try the next one".
# The registry (`_EXTRACTORS`) is an ordered list of (method_tag, extractor)
# pairs; `extract_final_answer` runs them in order and returns the first
# non-empty extract along with its method tag.
#
# Add a new extractor:
#   1. Write `_extract_<name>(response: str) -> str`.
#   2. Insert `("method_tag", _extract_<name>)` into `_EXTRACTORS` at the
#      correct priority (higher = more authoritative).
#   3. Add a test in `tests/test_verifier.py` that freezes a verbatim
#      response matching your format.
#   4. Update `VerifierResult.method` docstring and SERVER2_RUNBOOK.md §2.4.


def _extract_gsm8k_hash(response: str) -> str:
    """Last `#### X` — canonical GSM8K final-answer marker.

    Authoritative when present. Parses the first number-like token after
    the last `####`, so trailing prose ("#### 72 clips.") or whitespace
    doesn't clobber the extract.
    """
    if "####" not in response:
        return ""
    tail = response.rsplit("####", 1)[-1].strip()
    if not tail:
        return ""
    m = _GSM8K_HASH_NUM_RE.search(tail)
    if m:
        return m.group(0).replace(",", "")
    tok = tail.split()[0].replace(",", "")
    return tok


def _extract_answer_tag(response: str) -> str:
    """Last `<answer>X</answer>` — DeepSeek-R1 convention.

    R1-style models emit reasoning inside `<think>...</think>` (which
    often contains intermediate `\\boxed{}` values we do NOT want to
    grab) and the final answer inside `<answer>...</answer>`. Priority
    must be above `boxed` for this model family.
    """
    matches = list(_ANSWER_TAG_RE.finditer(response))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def _extract_boxed(response: str) -> str:
    """Last `\\boxed{X}` — canonical MATH final-answer marker."""
    matches = list(_BOXED_RE.finditer(response))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def _extract_answer_is(response: str) -> str:
    """'(the )?(final )?answer (is|:|=) X' prose heuristic.

    Deliberately narrow: does NOT match bare `= X` anywhere in the
    response. See the _ANSWER_IS_RE comment above for why.
    """
    m = _ANSWER_IS_RE.search(response)
    if not m:
        return ""
    return m.group(1).strip().strip(".,")


def _extract_last_numeric(response: str) -> str:
    """Last numeric token — weak fallback. Reliable only when the
    response is very short and obviously numeric."""
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", response)
    return nums[-1] if nums else ""


_EXTRACTORS: list[tuple[str, Callable[[str], str]]] = [
    ("gsm8k_hash", _extract_gsm8k_hash),
    ("answer_tag", _extract_answer_tag),
    ("boxed", _extract_boxed),
    ("answer_is", _extract_answer_is),
    ("fallback", _extract_last_numeric),
]


def extract_final_answer(response: str) -> tuple[str, str]:
    """Extract a final-answer string from a model response, with method tag.

    Runs `_EXTRACTORS` in priority order and returns the first non-empty
    extract. See the module docstring for the list of registered
    extractors and how to add a new one.
    """
    if not response:
        return "", "fallback"
    for method, extractor in _EXTRACTORS:
        extracted = extractor(response)
        if extracted:
            return extracted, method
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
