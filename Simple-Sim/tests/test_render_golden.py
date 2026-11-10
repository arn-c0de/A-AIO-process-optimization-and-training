"""Golden hashes of the default 2D render (``style: photoreal``, ``render.board.routing: legacy``).

Catches *unintended* changes of the generated images: a changed constant, an extra random draw or a
reordered drawing step changes every image of a seed, and ``generate --extend`` would then append
images with another look to an existing dataset. When the change is intended:

1. bump ``RENDERER_VERSION`` in ``simple_sim/generators/version.py`` (datasets record it, and
   ``--extend`` refuses to mix versions), and
2. regenerate the hashes: ``.venv/bin/python -m tests.test_render_golden --update``.

Rasterisation (``LINE_AA``, ``resize``, blur) depends on the OpenCV build, so the test is skipped
when OpenCV/NumPy differ from the versions that wrote the golden file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from simple_sim.generators.version import RENDERER_VERSION

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "golden" / "render_2d.json"
# (run config, classes): one chip, one fine-pitch IC, one SOT; seeds come from the run configs.
CASES = {
    "run_0001.yaml": ["OK", "MISALIGNED", "TOMBSTONE"],
    "run_qfn32.yaml": ["OK", "MISSING"],
    "run_sot23.yaml": ["OK", "MISALIGNED"],
}


def _versions() -> dict[str, str]:
    return {"opencv": cv2.__version__, "numpy": np.__version__}


def render_hashes(tmp_dir: Path) -> dict[str, str]:
    """md5 of the decoded pixels of every image of a tiny dataset per case (name -> hash)."""
    from simple_sim.generation.dataset import generate_dataset

    out: dict[str, str] = {}
    for run, classes in CASES.items():
        cfg = yaml.safe_load((ROOT / "configs" / run).read_text(encoding="utf-8"))
        cfg["classes"] = dict.fromkeys(classes, 1)
        cfg_path = tmp_dir / f"cfg_{Path(run).stem}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        data = tmp_dir / f"ds_{Path(run).stem}"
        generate_dataset(cfg_path, data, project_root=ROOT)
        for img in sorted((data / "images").rglob("*.png")):
            pixels = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
            out[f"{Path(run).stem}/{img.relative_to(data / 'images').as_posix()}"] = hashlib.md5(pixels.tobytes()).hexdigest()
    return out


def _load() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_default_2d_render_matches_golden(tmp_path, monkeypatch):
    if not GOLDEN.exists():
        pytest.fail(f"missing {GOLDEN}; create it with: python -m tests.test_render_golden --update")
    golden = _load()
    if golden.get("versions") != _versions():
        pytest.skip(f"golden hashes were written with {golden.get('versions')}, running {_versions()}")
    assert golden["renderer_version"] == RENDERER_VERSION, (
        "RENDERER_VERSION changed: regenerate the golden hashes (python -m tests.test_render_golden --update)")
    monkeypatch.delenv("IMAGE_FILTERS", raising=False)
    got = render_hashes(tmp_path)
    changed = sorted(k for k in golden["hashes"] if got.get(k) != golden["hashes"][k])
    assert set(got) == set(golden["hashes"]), "the set of rendered images changed"
    assert not changed, (
        f"2D render output changed for {changed}. If intended: bump RENDERER_VERSION "
        f"(simple_sim/generators/version.py) and run: python -m tests.test_render_golden --update")


def main(argv: list[str]) -> int:
    if "--update" not in argv:
        print(__doc__)
        return 2
    import os
    import tempfile

    os.environ.pop("IMAGE_FILTERS", None)
    with tempfile.TemporaryDirectory() as tmp:
        hashes = render_hashes(Path(tmp))
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps({"renderer_version": RENDERER_VERSION, "versions": _versions(), "hashes": hashes},
                                 indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(hashes)} hashes to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
