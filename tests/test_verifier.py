"""Verifier cases that the plan requires to work out of the box."""

from __future__ import annotations

from grpocredit.rollout.verifier import MathVerifier, extract_final_answer


def test_extract_boxed_last_match() -> None:
    text = r"We compute $\boxed{1}$ but actually the answer is $\boxed{42}$."
    ans, method = extract_final_answer(text)
    assert ans == "42"
    assert method == "boxed"


def test_extract_the_answer_is() -> None:
    text = "So after simplifying, the answer is 7."
    ans, method = extract_final_answer(text)
    assert "7" in ans
    assert method == "answer_is"


def test_extract_gsm8k_hash() -> None:
    text = "Some reasoning here.\n#### 31"
    ans, method = extract_final_answer(text)
    assert ans == "31"
    assert method == "gsm8k_hash"


def test_verifier_integer_match() -> None:
    v = MathVerifier()
    result = v.score(r"The answer is $\boxed{42}$.", "42")
    assert result.correct is True


def test_verifier_wrong_answer() -> None:
    v = MathVerifier()
    result = v.score(r"$\boxed{41}$", "42")
    assert result.correct is False
