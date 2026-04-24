"""Typer app roots for the four sprint scripts.

Lets callers use either the underlying `scripts/sprint_*.py` CLIs directly, or
the installed entry points (`grpocredit-d1`, `grpocredit-d2-oracle`, ...).
"""

from __future__ import annotations


def d1_smoke() -> None:
    from scripts.sprint_d1_infra_smoke import app

    app()


def d2_oracle() -> None:
    from scripts.sprint_d2_oracle import app

    app()


def d2_concordance() -> None:
    from scripts.sprint_d2_concordance import app

    app()


def d3_gate() -> None:
    from scripts.sprint_d3_gate_report import app

    app()
