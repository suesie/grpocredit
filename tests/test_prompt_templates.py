"""Tests for the prompt templates in `grpocredit.rollout.datasets`.

The `vineppo_math_task` template is load-bearing for three oracle configs
(`rho1b_sft_gsm8k`, `deepseek_math_sft`, `deepseek_math_sft_gsm8k`) — if
the template string drifts from what VinePPO SFT'd those checkpoints on,
pass@1 collapses 5-6× and the §5 group-variance gate trips (see
SERVER2_RUNBOOK.md §2.2). Pin it here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grpocredit.common.config import load_config
from grpocredit.rollout.datasets import (
    TEMPLATES,
    _apply_vineppo_math_task,
    format_prompt,
)


class _StubTokenizer:
    """A tokenizer that raises on chat-template calls, to simulate a base LM."""

    def apply_chat_template(self, *_a, **_kw):
        raise RuntimeError("base LMs have no chat template")


def test_vineppo_template_is_registered():
    assert "vineppo_math_task" in TEMPLATES
    assert TEMPLATES["vineppo_math_task"] is _apply_vineppo_math_task


def test_vineppo_template_exact_string():
    # The format rho-1b / deepseekmath-7b-sft were actually trained on,
    # lifted verbatim from
    # VinePPO-grpo/configs/prompt_library/generic_{GSM8K,MATH}_step_by_step.jsonnet
    # and VinePPO-grpo/configs/sft_rho1b_for_gsm8k.jsonnet.
    q = "What is 2 + 2?"
    expected = "[MATH_TASK] Problem:\nWhat is 2 + 2?\n\nSolution:"
    assert _apply_vineppo_math_task(None, q) == expected
    # Dispatch path
    assert (
        format_prompt(q, _StubTokenizer(), template="vineppo_math_task") == expected
    )


def test_vineppo_template_ignores_tokenizer():
    # Must NOT touch the tokenizer — VinePPO's format is plain string
    # concat, no chat template. This also means the fallback System/User
    # wrapper from math_instruct must never leak in.
    q = "Prove that sqrt(2) is irrational."
    t1 = _apply_vineppo_math_task(None, q)
    t2 = _apply_vineppo_math_task(_StubTokenizer(), q)
    t3 = _apply_vineppo_math_task("anything_not_a_tokenizer", q)
    assert t1 == t2 == t3
    assert "System:" not in t1
    assert "User:" not in t1
    assert "Assistant:" not in t1


def test_vineppo_template_preserves_multiline_questions():
    q = "Given x = 3,\ncompute 2x + 1."
    out = _apply_vineppo_math_task(None, q)
    assert out == f"[MATH_TASK] Problem:\n{q}\n\nSolution:"
    # The two trailing \n before Solution are load-bearing (VinePPO's
    # stop-token is "\n\n\nProblem:" which relies on exactly this spacing
    # for the next-question boundary). Verify the count.
    assert out.endswith("\n\nSolution:")


def test_unknown_template_raises():
    with pytest.raises(ValueError, match="Unknown template"):
        format_prompt("q", _StubTokenizer(), template="does_not_exist")


# ─── Oracle configs that depend on this template ────────────────────────
@pytest.mark.parametrize(
    "config_path, expected_temp, expected_top_p, expected_stop",
    [
        (
            "configs/oracle/rho1b_sft_gsm8k.yaml",
            0.35,
            0.9,
            ["\n\n\nProblem:"],
        ),
        (
            "configs/oracle/deepseek_math_sft.yaml",
            0.35,
            0.9,
            ["\n\n\nProblem:"],
        ),
        (
            "configs/oracle/deepseek_math_sft_gsm8k.yaml",
            0.35,
            0.9,
            ["\n\n\nProblem:"],
        ),
    ],
)
def test_vineppo_oracle_configs_use_matching_sampling(
    config_path, expected_temp, expected_top_p, expected_stop
):
    """Each oracle config on a VinePPO-published SFT checkpoint must use
    the `vineppo_math_task` template AND VinePPO's eval-time sampling
    settings. The two halves must move together — template without
    sampling (or vice versa) still distribution-shifts the policy."""
    cfg = load_config(Path(config_path))
    assert cfg.data.prompt_template == "vineppo_math_task", (
        f"{config_path}: template must be 'vineppo_math_task' for VinePPO-"
        f"published SFT checkpoints; see SERVER2_RUNBOOK.md §2.2."
    )
    assert cfg.rollout.temperature == expected_temp, (
        f"{config_path}: expected temperature={expected_temp} "
        f"(VinePPO eval-time), got {cfg.rollout.temperature}. Do NOT "
        f"set this from the base config's 0.9 — that's Qwen-Instruct's "
        f"regime, not the right distribution for a base LM SFT."
    )
    assert cfg.rollout.top_p == expected_top_p, (
        f"{config_path}: expected top_p={expected_top_p}, "
        f"got {cfg.rollout.top_p}"
    )
    assert cfg.rollout.stop == expected_stop, (
        f"{config_path}: expected stop={expected_stop!r} "
        f"(prevents spillover into the next problem under the VinePPO "
        f"template), got {cfg.rollout.stop!r}"
    )


def test_qwen_instruct_oracle_keeps_chat_template():
    """qwen_math_instruct.yaml must NOT be switched to vineppo_math_task —
    Qwen-Instruct is a chat-tuned model and the tokenizer exposes the
    correct chat template."""
    cfg = load_config(Path("configs/oracle/qwen_math_instruct.yaml"))
    assert cfg.data.prompt_template == "math_instruct"
    # Keep base config's Qwen-regime sampling.
    assert cfg.rollout.temperature == 0.9
    assert cfg.rollout.top_p == 0.95
