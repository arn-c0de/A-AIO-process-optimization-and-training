"""Tk-free tests for the GUI foundation: theme tokens, EventBus, TaskRunner, help texts."""

from __future__ import annotations

import re
import threading
import time

from gui.core.context import AppContext
from gui.core.events import EventBus
from gui.core.tasks import TaskCancelled, TaskRunner, UiDispatcher
from gui.help.texts import HELP, SECTION_HELP, help_text
from gui.theme.tokens import DARK, FONT_ROLES, LIGHT, class_color, font_name, palette_for

HEX = re.compile(r"^#[0-9a-f]{6}$")


def test_palettes_are_complete_hex() -> None:
    assert LIGHT.as_dict().keys() == DARK.as_dict().keys()
    for pal in (LIGHT, DARK):
        for key, val in pal.as_dict().items():
            if key == "name":
                continue
            assert HEX.match(val), (pal.name, key, val)


def _luma(hex_color: str) -> float:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_text_contrast_both_themes() -> None:
    for pal in (LIGHT, DARK):
        assert abs(_luma(pal.text) - _luma(pal.bg)) > 0.6
        assert abs(_luma(pal.text) - _luma(pal.surface)) > 0.6
        assert abs(_luma(pal.text_muted) - _luma(pal.bg)) > 0.25


def test_palette_for_and_fonts() -> None:
    assert palette_for("dark") is DARK and palette_for(True) is DARK
    assert palette_for("light") is LIGHT and palette_for(None) is LIGHT
    assert font_name("Heading") == "SimSimHeading"
    assert set(FONT_ROLES) >= {"Title", "Heading", "Body", "Small", "Mono"}


def test_class_colors_cover_valid_classes() -> None:
    from simple_sim.config import VALID_CLASSES

    for cls in VALID_CLASSES:
        assert class_color(cls) != class_color("__unknown__"), cls


def test_event_bus_delivers_and_unsubscribes() -> None:
    bus = EventBus()
    got: list = []
    unsub = bus.subscribe("x", lambda **kw: got.append(kw))
    bus.publish("x", a=1)
    unsub()
    bus.publish("x", a=2)
    assert got == [{"a": 1}]


def test_event_bus_isolates_failing_handler() -> None:
    bus = EventBus()
    got: list = []
    bus.subscribe("x", lambda **_kw: 1 / 0)
    bus.subscribe("x", lambda **kw: got.append(kw["v"]))
    bus.publish("x", v=3)
    assert got == [3]


def test_event_bus_marshals_worker_thread_publish() -> None:
    disp = UiDispatcher()
    bus = EventBus(post=disp.post)
    got: list = []
    bus.subscribe("x", lambda **kw: got.append(threading.get_ident()))
    t = threading.Thread(target=lambda: bus.publish("x"))
    t.start()
    t.join()
    assert got == []  # not delivered on the worker
    disp.drain()
    assert got == [threading.get_ident()]


def _wait_drain(disp: UiDispatcher, cond, timeout: float = 5.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        disp.drain()
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError("timeout")


def test_task_runner_done_progress_and_events() -> None:
    ctx = AppContext(sim_root=None)  # type: ignore[arg-type]
    events: list[str] = []
    for topic in ("task.started", "task.progress", "task.finished"):
        ctx.bus.subscribe(topic, lambda task, _t=topic: events.append(_t))
    results: list = []

    def work(task):
        task.progress(0.5, "half")
        return 42

    h = ctx.tasks.run(work, name="t", on_done=results.append)
    _wait_drain(ctx.dispatcher, lambda: results)
    assert results == [42]
    assert h.state == "done" and h.fraction == 0.5 and h.message == "half"
    assert events[0] == "task.started" and events[-1] == "task.finished" and "task.progress" in events
    assert ctx.tasks.active == []


def test_task_runner_error_and_cancel() -> None:
    disp = UiDispatcher()
    runner = TaskRunner(disp.post)
    errors: list = []
    finals: list = []
    h = runner.run(lambda t: (_ for _ in ()).throw(ValueError("boom")), name="e",
                   on_error=errors.append, on_finally=lambda: finals.append(1))
    _wait_drain(disp, lambda: finals)
    assert h.state == "error" and isinstance(errors[0], ValueError)

    started = threading.Event()

    def long(task):
        started.set()
        while True:
            task.check_cancelled()
            time.sleep(0.01)

    done: list = []
    h2 = runner.run(long, name="c", on_done=done.append, on_finally=lambda: finals.append(2))
    started.wait(2)
    h2.cancel()
    _wait_drain(disp, lambda: 2 in finals)
    assert h2.state == "cancelled" and done == []
    assert issubclass(TaskCancelled, Exception)


def test_help_texts_are_well_formed() -> None:
    for key, text in {**HELP, **SECTION_HELP}.items():
        assert re.match(r"^[a-z_]+\.[a-z0-9_]+$", key), key
        assert text.strip() == text and text, key
    assert help_text("missing.key") == ""


def test_help_keys_used_in_gui_exist() -> None:
    """Every help_key="..." literal in gui/ must have a text."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "gui"
    pat = re.compile(r"""help_key\s*=\s*["']([a-z_]+\.[a-z0-9_]+)["']""")
    missing = []
    for py in root.rglob("*.py"):
        for key in pat.findall(py.read_text(encoding="utf-8")):
            if key not in HELP and key not in SECTION_HELP:
                missing.append(f"{py.relative_to(root)}: {key}")
    assert not missing, missing
