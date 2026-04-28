"""Diagnostic inspector for `day1_rollouts.jsonl`.

Reads the Day-1 smoke-test rollout artifact and prints the concrete stuff
that the aggregate summary hides: what prompts vLLM actually saw, what
the model emitted, which extraction path the verifier used, and where
things went wrong.

Use when the group-variance gate reports a suspiciously low pass@1 and
you want to see *why* without wiring up wandb's UI. Prints to stdout; no
side effects, no GPU.

Usage
-----
    python scripts/inspect_day1_rollouts.py \\
        --rollouts experiments/oracle/rho1b_sft_gsm8k/day1_rollouts.jsonl \\
        --n-samples 10

    # Also check tokenizer BOS handling (requires grpocredit env):
    python scripts/inspect_day1_rollouts.py \\
        --rollouts experiments/oracle/rho1b_sft_gsm8k/day1_rollouts.jsonl \\
        --tokenizer realtreetune/rho-1b-sft-GSM8K \\
        --n-samples 5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grpocredit.rollout.verifier import MathVerifier, extract_final_answer  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _truncate(s: str, n: int = 200) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"…[+{len(s) - n} chars]"


@app.command()
def main(
    rollouts: str = typer.Option(..., help="Path to day1_rollouts.jsonl"),
    n_samples: int = typer.Option(10, help="Number of trajectories to print"),
    tokenizer: str | None = typer.Option(
        None,
        help=(
            "HF tokenizer id to verify BOS handling. If set, prints "
            "decoded prompt_token_ids[:5] for each sample and flags "
            "missing BOS."
        ),
    ),
) -> None:
    path = Path(rollouts)
    if not path.exists():
        raise typer.BadParameter(f"rollouts file not found: {path}")

    tok = None
    bos_id: int | None = None
    if tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        bos_id = tok.bos_token_id
        print(f"[tokenizer] {tokenizer}: bos_token={tok.bos_token!r} (id={bos_id})")
        print()

    verifier = MathVerifier()
    lines = path.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    print(f"[rollouts] {path} ({total} trajectories)")
    print()

    # ── Aggregate: finish_reason distribution, method distribution,
    # response-length distribution.
    finish_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    resp_lens: list[int] = []
    n_correct = 0
    n_has_boxed = 0
    n_has_hash = 0
    n_has_eos = 0
    n_missing_bos = 0
    for raw in lines:
        rec = json.loads(raw)
        fin = rec.get("finish_reason", "")
        finish_counts[fin] = finish_counts.get(fin, 0) + 1
        text = rec.get("response_text", "") or ""
        resp_lens.append(len(rec.get("token_ids", []) or []))
        _extract, method = extract_final_answer(text)
        method_counts[method] = method_counts.get(method, 0) + 1
        if rec.get("correct"):
            n_correct += 1
        if "\\boxed{" in text:
            n_has_boxed += 1
        if "####" in text:
            n_has_hash += 1
        if fin == "stop":
            n_has_eos += 1
        if tok is not None and bos_id is not None:
            pids = rec.get("prompt_token_ids") or []
            if not pids or pids[0] != bos_id:
                n_missing_bos += 1

    def pct(n: int) -> str:
        return f"{n}/{total} ({100.0 * n / max(total, 1):.1f}%)"

    print(f"[aggregate] correct:         {pct(n_correct)}")
    print(f"[aggregate] finish=stop(EOS):{pct(n_has_eos)}")
    print(f"[aggregate] contains '####': {pct(n_has_hash)}")
    print(f"[aggregate] contains '\\boxed{{}}': {pct(n_has_boxed)}")
    if tok is not None:
        print(f"[aggregate] prompt_token_ids[0] != BOS: {pct(n_missing_bos)}")
    print(f"[aggregate] finish reasons: {finish_counts}")
    print(f"[aggregate] verifier extract method: {method_counts}")
    if resp_lens:
        resp_lens.sort()
        mid = resp_lens[len(resp_lens) // 2]
        p95 = resp_lens[int(0.95 * len(resp_lens))]
        print(
            f"[aggregate] response tokens: median={mid} p95={p95} "
            f"max={resp_lens[-1]} (cfg max_new_tokens=1024)"
        )
    print()

    # ── Detail: print first n_samples trajectories end-to-end.
    print(f"[detail] first {n_samples} trajectories:")
    print()
    for idx, raw in enumerate(lines[:n_samples]):
        rec = json.loads(raw)
        print(f"── traj {idx} [prompt_id={rec.get('prompt_id')}] " + "─" * 40)
        pids = rec.get("prompt_token_ids") or []
        if tok is not None and pids:
            print(
                f"  prompt_token_ids[:8] = {pids[:8]} "
                f"(first token decoded: {tok.decode([pids[0]])!r})"
            )
            if bos_id is not None and pids[0] != bos_id:
                print(f"  ⚠ first prompt token != BOS (expected {bos_id})")
        print(f"  prompt = {_truncate(rec.get('prompt', ''), 300)}")
        resp = rec.get("response_text", "") or ""
        print(f"  response ({len(rec.get('token_ids') or [])} tokens, "
              f"finish={rec.get('finish_reason')}):")
        print(f"    {_truncate(resp, 500)}")
        gt = rec.get("ground_truth_answer", "") or ""
        extracted, method = extract_final_answer(resp)
        vr = verifier.score(resp, gt) if gt else None
        print(f"  ground_truth = {gt!r}")
        print(f"  extracted    = {extracted!r}  (method={method})")
        if vr is not None:
            print(f"  verifier.correct = {vr.correct}  (reward={rec.get('reward')})")
        print()


if __name__ == "__main__":
    app()
