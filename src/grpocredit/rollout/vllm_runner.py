"""vLLM-backed rollout runner with forced-first-token support for the oracle.

Three rollout modes, all served by the same LLM instance:

1. `generate_from_prompts(prompts, n_per_prompt)`
   Main GRPO-style rollouts: full continuation from the system+user prompt.

2. `continue_from_prefixes(prefix_token_ids, n_continuations)`
   Stage-2 lookaheads: given the deterministic prefix up to a boundary `s_b`,
   sample K_LA short continuations. Used for semantic-entropy clustering.

3. `forced_action_rollouts(prefix_token_ids, first_token_ids, n_per_action)`
   Oracle: force each first-action `a ∈ A_s`, then sample to terminal. Returns
   the decoded response and the policy distribution at `s_b` (so we can weight
   the `Q^π(s, a)` estimates by π(a|s)).

Windows note: vLLM doesn't install on native Windows — the `_maybe_import_vllm`
helper raises a clear error so Windows dev machines can still `import
grpocredit.rollout.vllm_runner` for static analysis. Actual runs happen on a
Linux GPU node.

Per-request `seed` contract
---------------------------

A fixed `SamplingParams.seed` combined with `n > 1` and
`enable_prefix_caching=True` collapses all samples to near-duplicates when
prompts share a prefix — the per-request RNG is seeded once, and a cached
prefill leaves the N samples on the same trajectory. That was the root
cause of `mean_group_reward_std=0.0` on `rho-1b-sft-GSM8K` (runbook §2.3).

`_sampling_params` enforces the ecosystem convention:

* `n == 1`: pass the supplied seed through (useful for `policy_probs_at`).
* `n > 1`, `rollout_cfg.deterministic_n=False` (default, verl/TRL/OpenRLHF
  convention): drop the per-request seed; the engine seed set at
  `LLM(..., seed=...)` construction is still honoured.
* `n > 1`, `rollout_cfg.deterministic_n=True` (VinePPO convention): use
  `_generate_deterministic_n` to fan out into `n` single-sample requests
  with seeds `[base, base+1, …]`. Bit-reproducible, ~N× decode passes.

All three rollout entry points route through `_sampling_params`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from grpocredit.common.config import ModelConfig, RolloutConfig
from grpocredit.common.types import RolloutResult

log = logging.getLogger(__name__)


def _maybe_import_vllm() -> Any:
    try:
        import vllm  # type: ignore

        return vllm
    except ImportError as e:
        raise ImportError(
            "vLLM is required for VLLMRolloutRunner but not installed. "
            "Install on Linux via `pip install vllm` or install the optional extra: "
            "`pip install grpocredit[vllm]`."
        ) from e


class RolloutBackend(Protocol):
    """Abstract interface so sprint scripts can swap vLLM ↔ HF transformers."""

    def generate_from_prompts(
        self,
        prompts: list[str],
        *,
        n_per_prompt: int,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> list[list[RolloutResult]]: ...

    def continue_from_prefixes(
        self,
        prefix_token_ids: list[list[int]],
        *,
        n_continuations: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        seed: int | None = None,
    ) -> list[list[RolloutResult]]: ...

    def forced_action_rollouts(
        self,
        prefix_token_ids: list[int],
        first_token_ids: list[int],
        *,
        n_per_action: int,
        max_new_tokens: int,
        temperature: float = 0.9,
        top_p: float = 0.95,
        seed: int | None = None,
    ) -> list[list[RolloutResult]]: ...

    def policy_probs_at(
        self,
        prefix_token_ids: list[int],
        *,
        top_k: int = 32,
    ) -> list[tuple[int, float]]: ...

    def tokenize(self, text: str) -> list[int]: ...

    def detokenize(self, token_ids: list[int]) -> str: ...

    @property
    def tokenizer(self) -> Any: ...


@dataclass
class VLLMRolloutRunner:
    model_cfg: ModelConfig
    rollout_cfg: RolloutConfig
    _llm: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _warned_seed_dropped: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        vllm = _maybe_import_vllm()
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.tokenizer_id,
            trust_remote_code=self.model_cfg.trust_remote_code,
        )
        self._llm = vllm.LLM(
            model=self.model_cfg.name_or_path,
            tokenizer=self.model_cfg.tokenizer_id,
            trust_remote_code=self.model_cfg.trust_remote_code,
            dtype=self.model_cfg.dtype,
            tensor_parallel_size=self.rollout_cfg.tensor_parallel_size,
            gpu_memory_utilization=self.rollout_cfg.gpu_memory_utilization,
            swap_space=self.rollout_cfg.swap_space_gb,
            enable_prefix_caching=self.rollout_cfg.enable_prefix_caching,
            seed=self.rollout_cfg.seed,
        )
        log.info("VLLMRolloutRunner initialised (%s)", self.model_cfg.name_or_path)

    # ── low-level helpers ────────────────────────────────────────────
    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def tokenize(self, text: str) -> list[int]:
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def detokenize(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def _sampling_params(
        self,
        *,
        n: int = 1,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        logprobs: int | None = None,
        stop: list[str] | None = None,
    ) -> Any:
        """Build a `vllm.SamplingParams` that obeys the seed contract.

        See the module docstring for the full rationale. Summary:
        * `n == 1`  → pass the supplied (or config) seed through.
        * `n  > 1`  → drop the per-request seed and let the engine RNG
          advance (verl / TRL / OpenRLHF convention). Callers who need
          bit-reproducibility for `n > 1` should use
          `_generate_deterministic_n` instead, which fans out into `n`
          separate `n=1` requests each with a distinct seed (VinePPO
          `num_expansion_rounds` pattern).
        """
        vllm = _maybe_import_vllm()
        effective_seed: int | None
        if n <= 1:
            effective_seed = seed if seed is not None else self.rollout_cfg.seed
        else:
            effective_seed = None
            if seed is not None and not self._warned_seed_dropped:
                # Log-once so users running with a fixed per-request seed and
                # `n > 1` get a clear signal that the seed was intentionally
                # dropped rather than silently honoured. Callers who genuinely
                # want reproducibility should flip `rollout.deterministic_n`.
                log.warning(
                    "Dropping per-request seed=%s on n=%d sampling call; "
                    "set `rollout.deterministic_n=True` for a VinePPO-style "
                    "seed-rotated fan-out, or rely on the engine-level seed "
                    "(%s) set at LLM construction.",
                    seed,
                    n,
                    self.rollout_cfg.seed,
                )
                self._warned_seed_dropped = True
        return vllm.SamplingParams(
            n=n,
            max_tokens=max_new_tokens or self.rollout_cfg.max_new_tokens,
            temperature=temperature if temperature is not None else self.rollout_cfg.temperature,
            top_p=top_p if top_p is not None else self.rollout_cfg.top_p,
            top_k=self.rollout_cfg.top_k,
            seed=effective_seed,
            logprobs=logprobs if logprobs is not None else self.rollout_cfg.logprobs,
            stop=stop or self.rollout_cfg.stop,
        )

    def _generate_deterministic_n(
        self,
        *,
        requests: list[Any],
        n: int,
        base_seed: int,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        logprobs: int | None = None,
        stop: list[str] | None = None,
    ) -> list[list[Any]]:
        """VinePPO-style fan-out: issue `n` single-sample requests per input
        with seeds `[base_seed, base_seed+1, …, base_seed+n-1]`.

        Returns a list-of-lists aligned with `requests`: for each input, a list
        of `n` per-sample vLLM completion outputs (one per seeded request).

        With `enable_prefix_caching=True` this is cheap — prefill is cached
        after the first seed, so the extra cost is ~n decode passes per input.
        Without prefix caching it is O(n) in total prefill FLOPs. Callers who
        don't need bit-reproducible per-sample provenance should use the
        seed-dropped `n > 1` path via `_sampling_params` instead.
        """
        per_request_outputs: list[list[Any]] = [[] for _ in requests]
        for sample_idx in range(n):
            params = self._sampling_params(
                n=1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=base_seed + sample_idx,
                logprobs=logprobs,
                stop=stop,
            )
            outs = self._llm.generate(requests, params, use_tqdm=False)
            for i, out in enumerate(outs):
                per_request_outputs[i].append(out.outputs[0])
        return per_request_outputs

    def _rolloutresult_from_vllm(
        self, prompt: str, prompt_ids: list[int], completion: Any
    ) -> RolloutResult:
        """Build a `RolloutResult` from a vLLM `CompletionOutput`."""
        token_ids = list(completion.token_ids)
        # Per-token logprobs: completion.logprobs is a list of {token_id: Logprob}
        # where each dict covers that generation step. Extract the sampled token's
        # logprob and approximate entropy from the top-K it returned.
        logprobs_list: list[float] = []
        entropies: list[float] = []
        if completion.logprobs:
            import math

            for step_dict, sampled_tid in zip(completion.logprobs, token_ids, strict=False):
                if step_dict is None:
                    logprobs_list.append(0.0)
                    entropies.append(0.0)
                    continue
                sampled = step_dict.get(sampled_tid)
                lp = float(getattr(sampled, "logprob", 0.0)) if sampled is not None else 0.0
                logprobs_list.append(lp)
                # Entropy estimate from top-k logprobs (partial — only for top-k
                # vLLM returns).
                ent = 0.0
                total_p = 0.0
                for _tid, lp_obj in step_dict.items():
                    p = math.exp(float(getattr(lp_obj, "logprob", 0.0)))
                    total_p += p
                    ent -= p * float(getattr(lp_obj, "logprob", 0.0))
                entropies.append(ent if total_p > 0 else 0.0)
        lp_sum = sum(logprobs_list)
        return RolloutResult(
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            response_text=completion.text,
            token_ids=token_ids,
            logprobs=logprobs_list,
            logprob_sum=lp_sum,
            token_entropies=entropies,
            finish_reason=completion.finish_reason or "",
            backend="vllm",
        )

    # ── public API ───────────────────────────────────────────────────
    def generate_from_prompts(
        self,
        prompts: list[str],
        *,
        n_per_prompt: int,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> list[list[RolloutResult]]:
        if n_per_prompt > 1 and self.rollout_cfg.deterministic_n:
            base_seed = seed if seed is not None else self.rollout_cfg.seed
            completions_by_prompt = self._generate_deterministic_n(
                requests=prompts,
                n=n_per_prompt,
                base_seed=base_seed,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            # We need `prompt_token_ids` per prompt for RolloutResult; pull it
            # off a 1-token cheap tokenize rather than a second generate call.
            results: list[list[RolloutResult]] = []
            for prompt, completions in zip(prompts, completions_by_prompt, strict=False):
                prompt_ids = self.tokenize(prompt)
                results.append(
                    [self._rolloutresult_from_vllm(prompt, prompt_ids, c) for c in completions]
                )
            return results

        params = self._sampling_params(
            n=n_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        outputs = self._llm.generate(prompts, params, use_tqdm=False)
        results = []
        for prompt, out in zip(prompts, outputs, strict=False):
            prompt_ids = list(out.prompt_token_ids)
            results.append(
                [self._rolloutresult_from_vllm(prompt, prompt_ids, c) for c in out.outputs]
            )
        return results

    def continue_from_prefixes(
        self,
        prefix_token_ids: list[list[int]],
        *,
        n_continuations: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        seed: int | None = None,
    ) -> list[list[RolloutResult]]:
        """Sample `n_continuations` short continuations from each prefix."""
        vllm = _maybe_import_vllm()
        prompts = [vllm.TokensPrompt(prompt_token_ids=list(ids)) for ids in prefix_token_ids]

        if n_continuations > 1 and self.rollout_cfg.deterministic_n:
            base_seed = seed if seed is not None else self.rollout_cfg.seed
            completions_by_prefix = self._generate_deterministic_n(
                requests=prompts,
                n=n_continuations,
                base_seed=base_seed,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            results: list[list[RolloutResult]] = []
            for ids, completions in zip(prefix_token_ids, completions_by_prefix, strict=False):
                detok = self.detokenize(ids)
                results.append(
                    [self._rolloutresult_from_vllm(detok, list(ids), c) for c in completions]
                )
            return results

        params = self._sampling_params(
            n=n_continuations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        outputs = self._llm.generate(prompts, params, use_tqdm=False)
        results = []
        for ids, out in zip(prefix_token_ids, outputs, strict=False):
            detok = self.detokenize(ids)
            results.append(
                [self._rolloutresult_from_vllm(detok, list(ids), c) for c in out.outputs]
            )
        return results

    def forced_action_rollouts(
        self,
        prefix_token_ids: list[int],
        first_token_ids: list[int],
        *,
        n_per_action: int,
        max_new_tokens: int,
        temperature: float = 0.9,
        top_p: float = 0.95,
        seed: int | None = None,
    ) -> list[list[RolloutResult]]:
        """For each first-action, roll out n_per_action continuations from
        `prefix + [first_token]`."""
        vllm = _maybe_import_vllm()
        tokens_prompts = [
            vllm.TokensPrompt(prompt_token_ids=list(prefix_token_ids) + [a])
            for a in first_token_ids
        ]
        detok_prefix = self.detokenize(list(prefix_token_ids))

        if n_per_action > 1 and self.rollout_cfg.deterministic_n:
            base_seed = seed if seed is not None else self.rollout_cfg.seed
            completions_by_action = self._generate_deterministic_n(
                requests=tokens_prompts,
                n=n_per_action,
                base_seed=base_seed,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            results: list[list[RolloutResult]] = []
            for first_tid, completions in zip(first_token_ids, completions_by_action, strict=False):
                effective_prefix = detok_prefix + self.detokenize([first_tid])
                per_action: list[RolloutResult] = []
                for c in completions:
                    rr = self._rolloutresult_from_vllm(
                        effective_prefix,
                        list(prefix_token_ids) + [first_tid],
                        c,
                    )
                    rr.extra["forced_first_token_id"] = first_tid
                    per_action.append(rr)
                results.append(per_action)
            return results

        params = self._sampling_params(
            n=n_per_action,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        outputs = self._llm.generate(tokens_prompts, params, use_tqdm=False)
        results = []
        for first_tid, out in zip(first_token_ids, outputs, strict=False):
            effective_prefix = detok_prefix + self.detokenize([first_tid])
            per_action: list[RolloutResult] = []
            for c in out.outputs:
                rr = self._rolloutresult_from_vllm(
                    effective_prefix,
                    list(prefix_token_ids) + [first_tid],
                    c,
                )
                rr.extra["forced_first_token_id"] = first_tid
                per_action.append(rr)
            results.append(per_action)
        return results

    def policy_probs_at(
        self,
        prefix_token_ids: list[int],
        *,
        top_k: int = 32,
    ) -> list[tuple[int, float]]:
        """Return π(·|s) top-k as a list of (token_id, prob), descending by prob.

        Implemented by running a 1-token generation with logprobs=top_k and
        reading the step-0 distribution. vLLM currently exposes top-k logprobs
        at each generation step; this is the cleanest way without reaching into
        the model internals.
        """
        vllm = _maybe_import_vllm()
        import math

        params = vllm.SamplingParams(
            n=1,
            max_tokens=1,
            temperature=1.0,  # actual sampling here is irrelevant — we only read logprobs
            top_p=1.0,
            logprobs=top_k,
            seed=self.rollout_cfg.seed,
        )
        prompt = vllm.TokensPrompt(prompt_token_ids=list(prefix_token_ids))
        outputs = self._llm.generate([prompt], params, use_tqdm=False)
        out = outputs[0].outputs[0]
        if not out.logprobs:
            return []
        step0 = out.logprobs[0]
        probs = [(int(tid), math.exp(float(lp_obj.logprob))) for tid, lp_obj in step0.items()]
        probs.sort(key=lambda kv: kv[1], reverse=True)
        return probs[:top_k]
