"""Golden hashes of complete tiny mixed-profile datasets (``generate_profile_dataset``).

Pins every file of the dataset: decoded image pixels, ``meta.jsonl`` / ``labels.jsonl`` /
``config.yaml`` / split files byte for byte, the manifest without its timestamps and git commit.
The hashes were first written by the pre-N5 implementation (its own sampling loop, plan section 6),
so the ``resolve_run`` + ``plan_records`` rewrite is proven to produce the same datasets; the cases
cover what that generator does differently from ``generate``: samples always in ``domain_A`` (a
``domain_A`` camera and a second domain must stay unused), no v1 config overrides of the
profiles, the image continuing the sampling rng, explicit image filters, render details
(``board_drc``) in ``meta.jsonl``.

When the render output changes on purpose (``RENDERER_VERSION`` bumped), regenerate:
``.venv/bin/python -m tests.test_profile_dataset_golden --update``. Skipped when OpenCV/NumPy
differ from the versions that wrote the file (rasterisation depends on the build).
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "golden" / "profile_dataset.json"
_VOLATILE = {"created_at", "timestamp", "git_commit"}


def _base() -> dict[str, Any]:
    cfg = yaml.safe_load((ROOT / "configs" / "run_profile_cls_quick.yaml").read_text(encoding="utf-8"))
    cfg["classes"] = {"OK": 2, "MISSING": 1, "MISALIGNED": 2, "TOMBSTONE": 1}
    cfg["roi"].update(width_px=48, height_px=48)
    return cfg


def cases() -> dict[str, tuple[dict[str, Any], dict[str, Any] | None]]:
    """name -> (run config, image_filters or None for the default)."""
    out: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {"plain": (_base(), None)}
    cfg = _base()
    cfg["domains"]["domain_A"]["camera"] = {"preset": "phone"}  # must stay unused (no per-domain cameras)
    cfg["domains"]["domain_B"] = copy.deepcopy(_base()["domains"]["domain_A"])  # must stay unused
    out["domain_camera"] = (cfg, None)
    out["filters"] = (_base(), {"enable_blur": False, "enable_grain": False, "cardinal_rotation_90": False})
    cfg = _base()
    cfg["render"]["look"] = "quins"
    cfg["render"]["board"] = {"routing": "escape", "drc": "warn"}  # board_drc details in meta.jsonl
    out["quins_board"] = (cfg, None)
    cfg = _base()
    cfg["run"]["schema_version"] = 1  # v1 tolerance / geometry overrides must not apply to the profiles
    cfg["tolerances"] = {"0603": {"ok_shift_px": 9.0, "ok_rotation_deg": 9.0, "tombstone_tilt_deg": 60.0}}
    cfg["geometry_ranges"] = {"pad_width": [1, 2]}
    cfg["render"]["component_color"] = [30, 30, 30]
    out["v1"] = (cfg, None)
    return out


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _VOLATILE}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def dataset_hashes(ds: Path) -> dict[str, str]:
    """md5 per file of dataset *ds* (images: decoded pixels; manifest: without volatile keys)."""
    out: dict[str, str] = {}
    for p in sorted(x for x in ds.rglob("*") if x.is_file()):
        rel = p.relative_to(ds).as_posix()
        if p.suffix == ".png":
            data = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).tobytes()
        elif p.name == "dataset_manifest.json":
            data = json.dumps(_strip(json.loads(p.read_text(encoding="utf-8"))), sort_keys=True).encode()
        else:
            data = p.read_bytes()
        out[rel] = hashlib.md5(data).hexdigest()
    return out


def generate_cases(tmp_dir: Path) -> dict[str, dict[str, str]]:
    from simple_sim.generation.profile_dataset import generate_profile_dataset

    out: dict[str, dict[str, str]] = {}
    for name, (cfg, filters) in cases().items():
        cfg_path = tmp_dir / f"{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        kw = {} if filters is None else {"image_filters": filters}
        generate_profile_dataset(cfg_path, tmp_dir / name, project_root=ROOT, **kw)
        out[name] = dataset_hashes(tmp_dir / name)
    return out


def _versions() -> dict[str, str]:
    return {"opencv": cv2.__version__, "numpy": np.__version__}


def test_profile_datasets_match_golden(tmp_path, monkeypatch):
    from simple_sim.generators.version import RENDERER_VERSION

    if not GOLDEN.exists():
        pytest.fail(f"missing {GOLDEN}; create it with: python -m tests.test_profile_dataset_golden --update")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if golden.get("versions") != _versions():
        pytest.skip(f"golden hashes were written with {golden.get('versions')}, running {_versions()}")
    assert golden["renderer_version"] == RENDERER_VERSION, (
        "RENDERER_VERSION changed: regenerate (python -m tests.test_profile_dataset_golden --update)")
    monkeypatch.delenv("IMAGE_FILTERS", raising=False)
    monkeypatch.setenv("SIMPLE_SIM_RENDER_EXECUTOR", "thread")
    got = generate_cases(tmp_path)
    assert set(got) == set(golden["datasets"])
    for name, files in golden["datasets"].items():
        assert set(got[name]) == set(files), f"{name}: the set of dataset files changed"
        changed = sorted(k for k in files if got[name][k] != files[k])
        assert not changed, f"{name}: generate_profile_dataset output changed in {changed}"


def test_config_mapping_and_profiles_dir(tmp_path, monkeypatch):
    """A config mapping (not modified) and an explicit profiles dir give the same samples as the file
    and the default dir (the manifest then records the profiles' absolute paths)."""
    import shutil

    from simple_sim.generation.profile_dataset import generate_profile_dataset

    monkeypatch.delenv("IMAGE_FILTERS", raising=False)
    cfg, _ = cases()["plain"]
    cfg["classes"] = {"OK": 1, "MISSING": 1}
    before = json.dumps(cfg, sort_keys=True)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    for pid in cfg["run"]["component_profiles"]:
        shutil.copy(ROOT / "configs" / "profiles" / f"{pid}.yaml", profiles)
    generate_profile_dataset(path, tmp_path / "a", project_root=ROOT)
    generate_profile_dataset(cfg, tmp_path / "b", project_root=ROOT, profiles_dir=profiles)
    assert json.dumps(cfg, sort_keys=True) == before
    a, b = dataset_hashes(tmp_path / "a"), dataset_hashes(tmp_path / "b")
    assert {k: v for k, v in a.items() if k != "dataset_manifest.json"} == \
        {k: v for k, v in b.items() if k != "dataset_manifest.json"}
    manifest = json.loads((tmp_path / "b" / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert [Path(p["profile_path"]).parent for p in manifest["component_profiles"]] == [profiles.absolute()] * 3


def main(argv: list[str]) -> int:
    if "--update" not in argv:
        print(__doc__)
        return 2
    import os
    import tempfile

    from simple_sim.generators.version import RENDERER_VERSION

    os.environ.pop("IMAGE_FILTERS", None)
    os.environ["SIMPLE_SIM_RENDER_EXECUTOR"] = "thread"
    with tempfile.TemporaryDirectory() as tmp:
        datasets = generate_cases(Path(tmp))
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps({"renderer_version": RENDERER_VERSION, "versions": _versions(), "datasets": datasets},
                                 indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {sum(map(len, datasets.values()))} hashes to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
