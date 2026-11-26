"""The Blender part geometry is pinned (D35): ``render_batch.py`` builds every family exactly as before.

``render_batch.py`` reads its per-family decisions (height ratio, standoff, end caps, lead width,
marking vocabulary and layout) from the baked scene's ``part`` entry (``board.footprints.scene_part``)
instead of its own tables. Without ``bpy`` (not installed) the script is loaded against a recording
stub: :func:`trace` records every body, end cap, lead segment, pin-1 mark and marking text that
``_build_part`` creates, the contacts, the solder height and the rng state, for every family and
several bodies / footprints. ``tests/golden/blender_part_params.json`` holds those traces, first captured on
the script before D35 (HEAD ab592ad, which ignored ``part``), regenerated for renderer v5 (real-size body meshes,
bent lead strips, SOT-23-6 leads, chip inductor / LED end caps: ``simple_sim/generators/version.py``).

Regenerate only for an intended Blender geometry change:
``.venv/bin/python -m tests.test_blender_part_params --update [--script PATH]``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from tests.blender_stub import ROOT, SCRIPT, load_script, trace

GOLDEN = ROOT / "tests" / "golden" / "blender_part_params.json"


_PADS = {
    "chip_2pad": [[-30.0, 0.0, 20.0, 28.0], [30.0, 0.0, 20.0, 28.0]],
    "sot23": [[-25.0, -19.0, 14.0, 18.0], [-25.0, 19.0, 14.0, 18.0], [25.0, 0.0, 14.0, 22.0]],
    "soic_16": [[s * 52.0, -35.0 + 10.0 * i, 16.0, 6.0] for i in range(8) for s in (-1.0, 1.0)],
    "qfn_16": ([[s * 40.0, -12.0 + 8.0 * i, 10.0, 4.0] for i in range(4) for s in (-1.0, 1.0)]
               + [[-12.0 + 8.0 * i, s * 40.0, 4.0, 10.0] for i in range(4) for s in (-1.0, 1.0)]
               + [[0.0, 0.0, 36.0, 36.0]]),
}

# (footprint, body, mm_per_px, component_height_mm, seed): normal, rotated/shifted, tombstone, small.
_SCENARIOS = [
    ("chip_2pad", {"center": [3.0, -2.0], "length": 60.0, "width": 32.0, "rotation_deg": 7.5, "lead_frame_deg": 0.0,
                   "tilt_deg": 0.0, "is_tombstone": False}, 0.01, 0.45, 3),
    ("sot23", {"center": [-4.0, 5.0], "length": 58.0, "width": 26.0, "rotation_deg": 93.0, "lead_frame_deg": 90.0,
               "tilt_deg": 2.0, "is_tombstone": False}, 0.0137, 1.2, 11),
    ("soic_16", {"center": [1.5, 0.5], "length": 90.0, "width": 70.0, "rotation_deg": -4.0, "lead_frame_deg": 0.0,
                 "tilt_deg": 35.0, "is_tombstone": True, "raised_sign": -1.0}, 0.02, 1.75, 5),
    ("qfn_16", {"center": [0.0, 0.0], "length": 20.0, "width": 10.0, "rotation_deg": 0.0, "lead_frame_deg": 0.0,
                "tilt_deg": 0.0, "is_tombstone": False}, 0.01, 0.45, 7),
    ("soic_16", {"center": [6.0, -3.0], "length": 90.0, "width": 70.0, "rotation_deg": 12.0, "lead_frame_deg": 0.0,
                 "tilt_deg": 0.0, "is_tombstone": False}, 0.015, 1.75, 13),
]


def cases() -> dict[str, dict[str, Any]]:
    """Every family (plus an unknown name) on every scenario, and MISSING."""
    from simple_sim.board.footprints import FAMILIES, scene_part

    out: dict[str, dict[str, Any]] = {}
    for family in [*FAMILIES, "unknown_family"]:
        for i, (fp, body, mmpp, comp_h, seed) in enumerate(_SCENARIOS):
            board = {"family": family, "pads": _PADS[fp], "body": dict(body), "part": scene_part(family)}
            out[f"{family}/{i}_{fp}"] = {"board": board, "mm_per_px": mmpp, "component_height_mm": comp_h, "seed": seed}
        out[f"{family}/missing"] = {"board": {"family": family, "pads": _PADS["chip_2pad"], "body": None,
                                              "part": scene_part(family)},
                                    "mm_per_px": 0.01, "component_height_mm": 0.45, "seed": 1}
    return out


def all_traces(path: Path = SCRIPT) -> dict[str, Any]:
    mod = load_script(path)
    return json.loads(json.dumps({name: trace(mod, case) for name, case in cases().items()}))


def test_blender_part_geometry_matches_pre_d35_script():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    current = all_traces()
    assert set(current) == set(golden)
    changed = [name for name in golden if current[name] != golden[name]]
    assert not changed, f"Blender part geometry changed for {changed[:5]} ({len(changed)} cases)"


def test_part_params_reads_the_scene_not_the_family_name():
    """The family name is only a label: the scene's ``part`` decides the geometry."""
    from simple_sim.board.footprints import scene_part

    mod = load_script()
    case = cases()["resistor/0_chip_2pad"]
    board = dict(case["board"], family="soic")  # wrong name, resistor part: still a resistor
    p = mod.part_params([(-0.3, 0.0, 0.2, 0.28), (0.3, 0.0, 0.2, 0.28)], board, 0.01, 0.45, random.Random(3))
    assert p["end_caps"] and p["cap"] == 0.22 * 0.6 and p["marking"][0] in scene_part("resistor")["marking"]["codes"]


def test_bake_scene_carries_the_part_table():
    import numpy as np

    from simple_sim.board.footprints import scene_part
    from simple_sim.defects import sample_defect_params
    from simple_sim.generators.opencv2d import sample_nominal_geometry
    from simple_sim.generators.opencv2d.photoreal import render_board

    tol = {"ok_shift_px": 3.0, "ok_rotation_deg": 5.0, "tombstone_tilt_deg": 75.0}
    for footprint, component_type, family in (("chip_2pad", "chip_capacitor", "capacitor"), ("sot23", "", "sot")):
        rng = np.random.default_rng(3)
        nominal = sample_nominal_geometry({}, rng, {"pad_spacing_y": [36.0, 50.0]})
        defect = sample_defect_params("OK", rng, tol)
        _, scene = render_board(nominal, defect, {"component_type": component_type}, tol, np.random.default_rng(3),
                                footprint, (160, 160))
        assert scene["family"] == family and scene["part"] == scene_part(family)
        assert json.loads(json.dumps(scene["part"])) == scene["part"]  # survives the JSONL job file


def main(argv: list[str]) -> int:
    if "--update" not in argv:
        print(__doc__)
        return 2
    path = Path(argv[argv.index("--script") + 1]) if "--script" in argv else SCRIPT
    GOLDEN.write_text(json.dumps(all_traces(path), indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN} from {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
