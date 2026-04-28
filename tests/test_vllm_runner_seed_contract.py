"""Tests for `VLLMRolloutRunner`'s seed contract.

vLLM isn't installed in this CI env, so we monkey-patch
`grpocredit.rollout.vllm_runner._maybe_import_vllm` to return a fake vllm
module that records the `SamplingParams` kwargs it was called with. This
lets us assert the contract documented in the runner's module docstring:

* `n == 1`      → seed passed through (engine seed fallback if None)
* `n  > 1`, `deterministic_n=False`  → seed is dropped (None)
* `n  > 1`, `deterministic_n=True`   → expanded into n `n=1` requests
                                       with seeds [base, base+1, …]
"""

from __future__ import annotations

import pytest

from grpocredit.common.config import ModelConfig, RolloutConfig


class _FakeCompletionOutput:
    def __init__(self, text: str, token_ids: list[int]) -> None:
        self.text = text
        self.token_ids = token_ids
        self.logprobs = None
        self.finish_reason = "length"


class _FakeRequestOutput:
    def __init__(self, prompt_token_ids: list[int], outputs: list[_FakeCompletionOutput]) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs


class _FakeSamplingParams:
    last_calls: list[dict] = []

    def __init__(self, **kw) -> None:
        type(self).last_calls.append(dict(kw))
        self.__dict__.update(kw)


class _FakeLLM:
    def __init__(self, *_args, **_kwargs) -> None:
        self._counter = 0

    def generate(self, prompts, params, use_tqdm=False):  # noqa: ARG002
        # Produce `params.n` deterministic-but-distinct completions per prompt
        # so fan-out semantics and RolloutResult wiring both exercise.
        n = getattr(params, "n", 1)
        seed = getattr(params, "seed", None)
        outs: list[_FakeRequestOutput] = []
        for i, _ in enumerate(prompts):
            comps = [
                _FakeCompletionOutput(
                    text=f"p{i}_seed{seed}_sample{j}_call{self._counter}",
                    token_ids=[10 + j],
                )
                for j in range(n)
            ]
            outs.append(_FakeRequestOutput(prompt_token_ids=[1, 2, 3], outputs=comps))
        self._counter += 1
        return outs


class _FakeTokensPrompt:
    def __init__(self, prompt_token_ids: list[int]) -> None:
        self.prompt_token_ids = prompt_token_ids


class _FakeVllmModule:
    LLM = _FakeLLM
    SamplingParams = _FakeSamplingParams
    TokensPrompt = _FakeTokensPrompt


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
        return {"input_ids": [len(text)]}

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "<decoded>"


@pytest.fixture
def fake_runner(monkeypatch):
    _FakeSamplingParams.last_calls = []

    import grpocredit.rollout.vllm_runner as vr

    monkeypatch.setattr(vr, "_maybe_import_vllm", lambda: _FakeVllmModule)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *a, **k: _FakeTokenizer(),
    )

    model_cfg = ModelConfig(name_or_path="fake", trust_remote_code=False)
    rollout_cfg = RolloutConfig(seed=42, deterministic_n=False)

    runner = vr.VLLMRolloutRunner(model_cfg=model_cfg, rollout_cfg=rollout_cfg)
    return runner


def test_n1_passes_seed_through(fake_runner):
    # n=1 → seed must reach SamplingParams as-is.
    _FakeSamplingParams.last_calls = []
    fake_runner.generate_from_prompts(["hi"], n_per_prompt=1, seed=49)
    assert len(_FakeSamplingParams.last_calls) == 1
    assert _FakeSamplingParams.last_calls[0]["seed"] == 49
    assert _FakeSamplingParams.last_calls[0]["n"] == 1


def test_n1_none_seed_falls_back_to_config(fake_runner):
    _FakeSamplingParams.last_calls = []
    fake_runner.generate_from_prompts(["hi"], n_per_prompt=1, seed=None)
    assert _FakeSamplingParams.last_calls[0]["seed"] == 42  # rollout_cfg.seed


def test_n_gt_1_drops_seed_by_default(fake_runner):
    # This is the exact call shape that triggered the Day-1 collapse on
    # rho-1b-sft-GSM8K. After the fix the per-request seed MUST be None.
    _FakeSamplingParams.last_calls = []
    groups = fake_runner.generate_from_prompts(
        ["prompt_a", "prompt_b"], n_per_prompt=8, seed=49
    )
    assert len(_FakeSamplingParams.last_calls) == 1
    call = _FakeSamplingParams.last_calls[0]
    assert call["n"] == 8
    assert call["seed"] is None, (
        "seed MUST be dropped on n>1 calls; if this fails, the "
        "group-variance probe will collapse to duplicates again."
    )
    # Still returns 2 groups of 8 completions.
    assert len(groups) == 2
    assert all(len(g) == 8 for g in groups)


def test_n_gt_1_deterministic_expands_with_seed_rotation(fake_runner):
    # Flip the VinePPO-style mode and re-check: the single `n=8` request must
    # expand into 8 `n=1` requests with seeds [base, base+1, …, base+7].
    fake_runner.rollout_cfg.deterministic_n = True
    _FakeSamplingParams.last_calls = []

    groups = fake_runner.generate_from_prompts(
        ["prompt_a"], n_per_prompt=4, seed=100
    )

    assert len(_FakeSamplingParams.last_calls) == 4
    for i, call in enumerate(_FakeSamplingParams.last_calls):
        assert call["n"] == 1, f"fan-out must be n=1 per request, got {call['n']}"
        assert call["seed"] == 100 + i, (
            f"seed rotation broke: expected {100 + i}, got {call['seed']}"
        )

    # And we still expose 4 completions per prompt, each carrying its
    # distinct text (no silent deduplication during the fan-out rejoin).
    assert len(groups) == 1
    assert len(groups[0]) == 4
    texts = [rr.response_text for rr in groups[0]]
    assert len(set(texts)) == 4


def test_continue_from_prefixes_honors_contract(fake_runner):
    _FakeSamplingParams.last_calls = []
    fake_runner.continue_from_prefixes(
        [[1, 2, 3]], n_continuations=8, max_new_tokens=30, seed=77
    )
    assert _FakeSamplingParams.last_calls[0]["n"] == 8
    assert _FakeSamplingParams.last_calls[0]["seed"] is None


def test_forced_action_rollouts_honors_contract(fake_runner):
    _FakeSamplingParams.last_calls = []
    fake_runner.forced_action_rollouts(
        prefix_token_ids=[1, 2, 3],
        first_token_ids=[10, 20],
        n_per_action=32,
        max_new_tokens=128,
        seed=77,
    )
    # Default path: one SamplingParams(n=32, seed=None), applied to 2
    # forced-action prompts. Crucially seed is dropped.
    assert len(_FakeSamplingParams.last_calls) == 1
    assert _FakeSamplingParams.last_calls[0]["n"] == 32
    assert _FakeSamplingParams.last_calls[0]["seed"] is None


def test_forced_action_rollouts_deterministic_mode(fake_runner):
    fake_runner.rollout_cfg.deterministic_n = True
    _FakeSamplingParams.last_calls = []
    fake_runner.forced_action_rollouts(
        prefix_token_ids=[1, 2, 3],
        first_token_ids=[10, 20],
        n_per_action=4,
        max_new_tokens=128,
        seed=500,
    )
    # 4 fan-out requests, each n=1, each applied to both forced-action
    # prompts in one `generate` call, with seeds [500, 501, 502, 503].
    assert len(_FakeSamplingParams.last_calls) == 4
    assert [c["seed"] for c in _FakeSamplingParams.last_calls] == [500, 501, 502, 503]
    assert all(c["n"] == 1 for c in _FakeSamplingParams.last_calls)


def test_warn_once_only(fake_runner, caplog):
    # Log-once behaviour: calling n>1 with a fixed seed twice must only
    # emit one WARNING. Avoids log spam in the hot oracle loop.
    import logging

    _FakeSamplingParams.last_calls = []
    with caplog.at_level(logging.WARNING, logger="grpocredit.rollout.vllm_runner"):
        fake_runner.generate_from_prompts(["a"], n_per_prompt=4, seed=1)
        fake_runner.generate_from_prompts(["b"], n_per_prompt=4, seed=2)
    warnings = [r for r in caplog.records if "Dropping per-request seed" in r.getMessage()]
    assert len(warnings) == 1, (
        f"Expected exactly one 'seed dropped' warning; got {len(warnings)}"
    )
