"""``--help`` of every script is unchanged by the shared argparse fragments (plan D2).

``tests/golden/help/<script>.txt`` was captured with ``COLUMNS=80 python scripts/<script>.py --help``
before the scripts were switched to ``cli_utils.add_*_arg`` (2026-11-07). The scripts run
in-process (``runpy``) so torch is imported once for all of them.
"""

from __future__ import annotations

import contextlib
import io
import runpy
import sys
from pathlib import Path

import pytest

SIM_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "golden" / "help"
SCRIPTS = sorted(p.stem for p in GOLDEN.glob("*.txt"))


def test_every_script_has_a_golden_help() -> None:
    assert SCRIPTS == sorted(p.stem for p in (SIM_ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("name", SCRIPTS)
def test_help_output_unchanged(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    script = SIM_ROOT / "scripts" / f"{name}.py"
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    monkeypatch.setattr(sys, "path", list(sys.path))  # the scripts insert the repo root
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name="__main__")
    assert exc.value.code in (0, None)
    assert out.getvalue() == (GOLDEN / f"{name}.txt").read_text(encoding="utf-8")
