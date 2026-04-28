"""Verifier cases that the plan requires to work out of the box."""

from __future__ import annotations

from grpocredit.rollout.verifier import MathVerifier, extract_final_answer


def test_extract_boxed_last_match() -> None:
    # No ####, so `\boxed{}` wins — last-match beats first-match.
    text = r"We compute $\boxed{1}$ but actually $\boxed{42}$."
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


# ─── Regression: rho-1b-sft-GSM8K CoT mis-extraction (SERVER2_RUNBOOK §2.4) ──
# These are verbatim responses from `experiments/oracle/rho1b_sft_gsm8k/
# day1_rollouts.jsonl` where the old verifier mis-extracted an intermediate
# arithmetic step as the final answer. Freezes both the priority order
# (#### wins over `=` in CoT) and the `_ANSWER_IS_RE` narrowing (no longer
# latches onto bare `=` on a CoT line).
def test_gsm8k_hash_beats_intermediate_equation_natalia() -> None:
    """Traj 0 from the rho-1b regression. Pre-fix, this returned
    '24 clips in May' (from '= 24 clips in May' on the intermediate step)."""
    resp = (
        "\nNatalia sold 48 / 2 = 24 clips in May."
        "\nNatalia sold 48 + 24 = 72 clips altogether in April and May."
        "\n#### 72\n"
    )
    ans, method = extract_final_answer(resp)
    assert method == "gsm8k_hash"
    assert ans == "72"
    assert MathVerifier().score(resp, "72").correct is True


def test_gsm8k_hash_beats_intermediate_equation_julie() -> None:
    """Traj 3 — longer CoT with multiple `=` steps."""
    resp = (
        "\nThe number of pages that Julie read yesterday is 12 pages."
        "\nThe number of pages that she read today is 2 x 12 = 24 pages."
        "\nThe total number of pages that she read is 12 + 24 = 36 pages."
        "\nThe total number of pages that are remaining is 120 - 36 = 84 pages."
        "\nThe number of pages that she should read tomorrow is 84/2 = 42 pages."
        "\n#### 42\n"
    )
    ans, method = extract_final_answer(resp)
    assert method == "gsm8k_hash"
    assert ans == "42"
    assert MathVerifier().score(resp, "42").correct is True


def test_gsm8k_hash_beats_intermediate_equation_james() -> None:
    """Traj 4 — short CoT but three `=` signs before the final ####."""
    resp = (
        "\nHe writes each friend 3*2=6 pages a week"
        "\nSo he writes 6*2=12 pages every week"
        "\nThat means he writes 12*52=624 pages a year"
        "\n#### 624\n"
    )
    ans, method = extract_final_answer(resp)
    assert method == "gsm8k_hash"
    assert ans == "624"
    assert MathVerifier().score(resp, "624").correct is True


def test_answer_is_no_longer_matches_bare_equation() -> None:
    """A response with `= N` on a CoT line but no `####` and no `\\boxed{}`
    and no 'answer is' phrasing must not latch onto the `=` alone — that
    was the core footgun. Should fall through to the last-numeric-token
    fallback (which is still an imperfect heuristic, but is correctly
    labelled as `fallback` so downstream code can decide how much to trust
    it)."""
    resp = "We compute 2 + 2 = 4 in the first step. Some prose here."
    ans, method = extract_final_answer(resp)
    assert method == "fallback"
    assert ans == "4"


def test_gsm8k_hash_robust_to_trailing_prose() -> None:
    """A few rho-1b outputs append a period or 'clips' after the number."""
    for tail_suffix in ("", ".", " clips.", "\n", " ", " clips in May."):
        resp = f"some reasoning\n#### 72{tail_suffix}"
        ans, method = extract_final_answer(resp)
        assert method == "gsm8k_hash"
        assert ans == "72", (resp, ans)


def test_gsm8k_hash_negative_and_fraction() -> None:
    for raw, expect in [
        ("reasoning\n#### -7\n", "-7"),
        ("reasoning\n#### 3/4\n", "3/4"),
        ("reasoning\n#### 1,000\n", "1000"),  # verifier normalises commas
    ]:
        ans, method = extract_final_answer(raw)
        assert method == "gsm8k_hash"
        assert ans == expect, (raw, ans)


def test_priority_gsm8k_hash_beats_boxed() -> None:
    """If both #### and \\boxed{} are present (rare but possible after a
    template/training mismatch), #### wins — it's the authoritative
    final-answer marker, placed as the last thing by convention."""
    resp = r"We get $\boxed{1}$ as an intermediate step." "\n#### 42\n"
    ans, method = extract_final_answer(resp)
    assert method == "gsm8k_hash"
    assert ans == "42"


# ─── <answer>X</answer> tag (DeepSeek-R1 convention) ─────────────────────
def test_extract_answer_tag_basic() -> None:
    resp = "<answer>42</answer>"
    ans, method = extract_final_answer(resp)
    assert method == "answer_tag"
    assert ans == "42"


def test_answer_tag_beats_boxed_inside_think() -> None:
    """The R1 failure mode this protects against: intermediate `\\boxed{}`
    inside a `<think>` block would have been picked up by the `boxed`
    extractor. The final `<answer>` tag must win."""
    resp = (
        "<think>\n"
        r"Let me compute step-by-step. We have $\boxed{12}$ as a mid-step,"
        " then\n"
        r"further reduction gives $\boxed{7}$.\n"
        "</think>\n"
        "<answer>42</answer>"
    )
    ans, method = extract_final_answer(resp)
    assert method == "answer_tag"
    assert ans == "42"


def test_answer_tag_with_latex_content() -> None:
    """R1-style responses sometimes wrap the \\boxed{} *inside* the answer
    tag. We return the raw content; math_verify handles the LaTeX parse."""
    resp = r"<answer>\boxed{\frac{1}{2}}</answer>"
    ans, method = extract_final_answer(resp)
    assert method == "answer_tag"
    assert ans == r"\boxed{\frac{1}{2}}"


def test_answer_tag_last_match_wins() -> None:
    resp = "<answer>1</answer> then reconsider. <answer>42</answer>"
    ans, method = extract_final_answer(resp)
    assert method == "answer_tag"
    assert ans == "42"


def test_answer_tag_ignored_if_empty() -> None:
    """Empty answer tag must fall through to the next extractor."""
    resp = "<answer></answer>\nAfter reconsidering, the answer is 42."
    ans, method = extract_final_answer(resp)
    # Empty answer tag → falls through to answer_is heuristic.
    assert method == "answer_is"
    assert "42" in ans


def test_answer_tag_does_not_match_when_absent() -> None:
    """Plain GSM8K response — no <answer>, just ####. Must still pick #### .
    This is the rho-1b regression; the new extractor must not interfere
    with it."""
    resp = "reasoning\n#### 72\n"
    ans, method = extract_final_answer(resp)
    assert method == "gsm8k_hash"
    assert ans == "72"


def test_priority_answer_tag_beats_boxed_beats_answer_is() -> None:
    """All three markers present — answer_tag (priority 2) wins."""
    resp = (
        r"The answer is 100 in the intermediate. Or $\boxed{50}$. "
        "<answer>42</answer>"
    )
    ans, method = extract_final_answer(resp)
    assert method == "answer_tag"
    assert ans == "42"


# ─── Registry contract (adding an extractor shouldn't break existing) ───
def test_extractor_registry_order() -> None:
    """Freeze the priority order. Changing this order is a behavior change
    that must be justified in the commit message + SERVER2_RUNBOOK.md."""
    from grpocredit.rollout.verifier import _EXTRACTORS

    methods = [m for m, _ in _EXTRACTORS]
    assert methods == [
        "gsm8k_hash",
        "answer_tag",
        "boxed",
        "answer_is",
        "fallback",
    ]
