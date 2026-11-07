"""Import hygiene: GUI logic modules without tkinter, light library modules without cv2/torch.

Each check runs in a fresh interpreter so modules already imported by other tests do not
mask a regression.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SIM_ROOT = Path(__file__).resolve().parents[1]

_BLOCKER = """
import sys
class _Block:
    def __init__(self, names):
        self.names = set(names)
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.names:
            raise ImportError(f"blocked for test: {name}")
sys.meta_path.insert(0, _Block(BLOCKED))
"""


def _run(code: str, *, blocked: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
    src = _BLOCKER.replace("BLOCKED", repr(list(blocked))) + code
    return subprocess.run([sys.executable, "-c", src], cwd=SIM_ROOT, capture_output=True, text=True, timeout=120)


GUI_LOGIC_MODULES = [
    "analysis", "board_detection", "datasets", "merge", "pipeline",
    "predictions", "profile_preview", "validation", "weights",
]


@pytest.mark.parametrize("tab", GUI_LOGIC_MODULES)
def test_gui_logic_imports_without_tkinter(tab: str) -> None:
    proc = _run(f"import gui.tabs.{tab}.logic", blocked=("tkinter", "_tkinter"))
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("tab", ["datasets", "merge", "pipeline", "predictions", "weights"])
def test_gui_logic_imports_without_cv2_and_torch(tab: str) -> None:
    proc = _run(f"import gui.tabs.{tab}.logic", blocked=("tkinter", "_tkinter", "cv2", "torch", "torchvision"))
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("module", [
    "simple_sim.dataset_index",
    "simple_sim.filter_settings",
    "simple_sim.filter_profiles",
    "simple_sim.generators",
    "simple_sim.generators.version",
    "simple_sim.images",
    "simple_sim.manifest",
    "simple_sim.model_bundle",
    "simple_sim.reports",
    "simple_sim.telemetry",
    "gui.components.filter_popup",
    "gui.tabs",
])
def test_light_modules_do_not_load_cv2_or_torch(module: str) -> None:
    proc = _run(f"import {module}", blocked=("cv2", "torch", "torchvision", "tkinter", "_tkinter"))
    assert proc.returncode == 0, proc.stderr


def test_generator_backends_load_independently() -> None:
    proc = _run(
        "import simple_sim.generators.opencv2d\n"
        "assert 'simple_sim.generators.blender3d' not in sys.modules\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_filter_popup_exports_resolve_lazily() -> None:
    proc = _run(
        "import gui.components.filter_popup as fp\n"
        "assert 'gui.components.filter_popup.popup' not in sys.modules\n"
        "try:\n    fp.no_such_name\nexcept AttributeError:\n    pass\nelse:\n    raise SystemExit('expected AttributeError')\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_eval_utils_imports_without_torch() -> None:
    # I5: torch/tqdm are imported inside the functions that need them
    proc = _run("import simple_sim.eval_utils as eu\n"
                "assert eu.format_distribution('x', {'OK': 1}).endswith('OK: 1')\n",
                blocked=("torch", "torchvision", "tqdm", "cv2"))
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("module", [
    "simple_sim.board",
    "simple_sim.board.model",
    "simple_sim.board.geometry",
    "simple_sim.board.rules",
    "simple_sim.board.footprints",
    "simple_sim.board.options",
    "simple_sim.board.placement",
])
def test_board_data_model_does_not_load_cv2(module: str) -> None:
    proc = _run(
        f"import {module}\n"
        "assert 'cv2' not in sys.modules, 'cv2 loaded'\n"
        "assert 'simple_sim.board.pipeline' not in sys.modules\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_board_options_validate_without_cv2() -> None:
    """``config._validate_board`` (render.board, I7 for the board part) stays cv2-free."""
    proc = _run(
        "from simple_sim.board import BOARD_ROUTERS, DRC_MODES, board_options, validate_board_options\n"
        "from simple_sim.config import _validate_board\n"
        "_validate_board({'routing': 'escape', 'drc': 'warn', 'pour_probability': 0.5})\n"
        "assert board_options({'board': {'routing': 'escape'}}).routing == 'escape'\n"
        "assert 'cv2' not in sys.modules, 'cv2 loaded'\n",
        blocked=("cv2", "torch", "torchvision"),
    )
    assert proc.returncode == 0, proc.stderr


def test_board_lazy_exports_load_on_access() -> None:
    proc = _run(
        "import simple_sim.board as b\n"
        "assert 'simple_sim.board.drc' not in sys.modules\n"
        "from simple_sim.board import build_window_board, BoardDrcError\n"
        "assert 'simple_sim.board.pipeline' in sys.modules and 'cv2' in sys.modules\n"
    )
    assert proc.returncode == 0, proc.stderr
