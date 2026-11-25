"""Test-wide fixtures: every test starts without the environment knobs that change generation output.

``IMAGE_FILTERS`` (set by the GUI / ``run_pipeline.sh``) changes every rendered image, the event-log
variables would append test telemetry to a real ``events.jsonl``, and the render executor / worker
variables change how (not what) images are rendered. A test that needs one sets it with ``monkeypatch``.
"""

from __future__ import annotations

import pytest

_ISOLATED_ENV = ("IMAGE_FILTERS", "SIMPLE_SIM_EVENT_LOG", "EVENT_LOG", "SIMPLE_SIM_RENDER_EXECUTOR",
                 "SIMPLE_SIM_RENDER_WORKERS")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ISOLATED_ENV:
        monkeypatch.delenv(name, raising=False)
