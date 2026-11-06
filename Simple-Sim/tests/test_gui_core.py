"""Tk-free tests for the GUI foundation: theme tokens, EventBus, TaskRunner, help texts."""

from __future__ import annotations

import re
import threading
import time

from gui.core.context import AppContext
from gui.core.events import EventBus
from gui.core.tasks import TaskCancelled, TaskRunner, UiDispatcher, runner_for
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


class _FakeWidget:
    """Records ``after`` calls like a Tk widget (no display needed)."""

    def __init__(self) -> None:
        self.scheduled: list = []
        self.cancelled: list = []
        self._n = 0

    def after(self, ms, fn=None, *args):
        self._n += 1
        aid = f"after#{self._n}"
        self.scheduled.append((aid, fn))
        return aid

    def after_cancel(self, aid) -> None:
        self.cancelled.append(aid)


def test_dispatcher_reschedules_before_draining() -> None:
    """A drained callback that blocks (modal dialog) must find the next tick already scheduled."""
    disp = UiDispatcher()
    w = _FakeWidget()
    seen: list[int] = []
    disp.post(lambda: seen.append(len(w.scheduled)))
    disp.attach(w)
    assert seen == [1]  # next tick was scheduled before the callback ran
    assert w.scheduled[-1][1] == disp._tick


def test_dispatcher_keeps_ticking_when_callback_raises() -> None:
    disp = UiDispatcher()
    w = _FakeWidget()

    def boom() -> None:
        raise RuntimeError("x")

    disp.post(boom)
    disp.attach(w)
    assert len(w.scheduled) == 1


def test_dispatcher_attach_is_idempotent() -> None:
    disp = UiDispatcher()
    w = _FakeWidget()
    disp.attach(w)
    disp.attach(w)
    assert w.cancelled == ["after#1"]  # the first polling chain was cancelled
    assert disp._after_id == "after#2"


def test_cancel_after_normal_return_stays_done() -> None:
    """cancel() landing between the worker's return and the UI drain must not drop the result."""
    disp = UiDispatcher()
    runner = TaskRunner(disp.post)
    returned = threading.Event()
    results: list = []

    def work(task):
        returned.set()
        return "ok"

    h = runner.run(work, name="r", on_done=results.append)
    returned.wait(2)
    h.join(2)
    h.cancel()  # too late: fn already returned normally
    _wait_drain(disp, lambda: not h.running)
    assert h.state == "done" and results == ["ok"]


def test_progress_is_coalesced() -> None:
    ctx = AppContext(sim_root=None)  # type: ignore[arg-type]
    seen: list = []
    ctx.bus.subscribe("task.progress", lambda task: seen.append(task.snapshot()))
    go = threading.Event()

    def work(task):
        for i in range(1000):
            task.progress((i + 1) / 1000, f"{i + 1}")
        go.wait(2)
        return None

    h = ctx.tasks.run(work, name="p")
    # Worker posted 1000 updates without a drain in between: only one is queued.
    end = time.time() + 2
    while h.message != "1000" and time.time() < end:
        time.sleep(0.005)
    ctx.dispatcher.drain()
    assert len(seen) == 1 and seen[0] == (1.0, "1000")
    h.progress(None, "again")  # flag was cleared on delivery -> new notification
    ctx.dispatcher.drain()
    assert seen[-1] == (1.0, "again") and len(seen) == 2
    go.set()
    _wait_drain(ctx.dispatcher, lambda: not h.running)


def test_shutdown_waits_for_workers() -> None:
    disp = UiDispatcher()
    runner = TaskRunner(disp.post)
    wrote: list = []

    def work(task):
        while not task.cancelled():
            time.sleep(0.01)
        time.sleep(0.05)  # finishing a write after the cancel request
        wrote.append(1)

    runner.run(work, name="w")
    left = runner.shutdown(timeout=2.0)
    assert left == [] and wrote == [1]


def test_runner_for_falls_back_to_dispatcher_on_widget() -> None:
    class _State:
        ctx = None

    w = _FakeWidget()
    r1 = runner_for(_State(), w)
    assert runner_for(_State(), w) is r1  # one fallback runner per widget
    assert w.scheduled  # its dispatcher polls on the widget's loop
    got: list = []
    r1.post(lambda: got.append(threading.get_ident()))
    w.scheduled[-1][1]()  # run the next tick, as Tk would
    assert got == [threading.get_ident()]


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
