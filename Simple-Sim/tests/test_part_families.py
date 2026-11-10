"""N1: the PartFamily table (``board.footprints.FAMILIES``) and the constants derived from it."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from simple_sim.board import footprints as fp
from simple_sim.board.footprints import (BODY_KINDS, FAMILIES, GENERIC_FAMILY, PartFamily, family_of, part_family,
                                         profile_footprint)

ROOT = Path(__file__).resolve().parent.parent


def _profile_families() -> set[str]:
    out = set()
    for path in sorted((ROOT / "configs" / "profiles").glob("*.yaml")):
        prof = yaml.safe_load(path.read_text())
        out.add(family_of(profile_footprint(prof), str((prof.get("component") or {}).get("type", ""))))
    return out


def test_every_rendered_family_has_an_entry():
    """Every family family_of yields (all shipped profiles, every footprint, neighbours) is in the table."""
    seen = _profile_families()
    for footprint in ("chip_2pad", "sot23", "qfn_32", "soic_16", "tssop_14", "tssop_20", "sot_6", "chip_array_8"):
        for ct in ("", "chip_resistor", "chip_capacitor", "ferrite_bead", "inductor", "power_inductor",
                   "molded_inductor", "led"):
            seen.add(family_of(footprint, ct))
    seen.update(("resistor", "capacitor"))  # neighbour parts (placement.neighbour_layout)
    assert seen <= set(FAMILIES)
    assert set(FAMILIES) == seen  # no dead rows either


def test_table_rows_are_consistent():
    for name, fam in FAMILIES.items():
        assert isinstance(fam, PartFamily) and fam.name == name
        assert fam.kind in BODY_KINDS
        assert fam.refdes
        assert (fam.cap_len is not None) == (fam.kind == fp.KIND_CHIP)
        assert fam.marking in (None, *fp.PRINTED_MARKINGS, fp.MARK_LASER)
    with pytest.raises(AttributeError):
        FAMILIES["resistor"].refdes = "X"  # type: ignore[misc]
    with pytest.raises(TypeError):
        FAMILIES["x"] = GENERIC_FAMILY  # type: ignore[index]
    assert part_family("no_such_family") is GENERIC_FAMILY
    assert tuple(FAMILIES) == fp.FAMILY_NAMES


def test_legacy_constants_equal_the_derived_ones():
    """The old per-module names keep their values (and order) and agree with the table."""
    from simple_sim.board import placement
    from simple_sim.generators.opencv2d.photoreal import copper, palette

    assert fp.FAMILY_NAMES == ("resistor", "capacitor", "sot", "soic", "qfn", "sot6", "array_r", "array_c", "inductor",
                               "power_inductor", "led", "tht_pin")
    assert fp.CHIP_FAMILIES == ("resistor", "capacitor", "inductor", "power_inductor", "led")
    assert fp.FINE_PITCH_FAMILIES == ("soic", "qfn", "sot6", "array_r", "array_c")
    assert palette.CHIPS == fp.CHIP_FAMILIES == tuple(n for n, f in FAMILIES.items() if f.chip)
    assert palette.LEADED == ("sot", "soic", "sot6")
    assert palette.ARRAY_FAMILIES == ("array_r", "array_c")
    assert palette.UNMARKED == ("capacitor", "array_c", "inductor", "power_inductor", "led", "tht_pin")
    assert palette.CAP_LEN == {"capacitor": 0.26, "resistor": 0.22, "inductor": 0.25, "led": 0.16}
    legacy_refdes = {"resistor": "R", "capacitor": "C", "sot": "Q", "soic": "U", "qfn": "U", "sot6": "U",
                     "array_r": "RN", "array_c": "CN", "inductor": "L", "power_inductor": "L", "led": "D",
                     "tht_pin": "J"}  # tht_pin: the Blender-only through-hole row added after the legacy set
    assert legacy_refdes == placement.REFDES_PREFIX
    assert list(placement.REFDES_PREFIX) == list(legacy_refdes)
    assert palette.REFDES_PREFIX is placement.REFDES_PREFIX
    assert (*fp.CHIP_FAMILIES, "array_r", "array_c") == copper.RULE_CHIP_FAMILIES
    # The former inline dicts of parts.draw_body.
    insets = {"qfn": 0.012, "soic": 0.035, "sot": 0.05, "sot6": 0.05, "array_r": 0.05, "array_c": 0.05, "tht_pin": 0.0}
    grains = {"resistor": 0.025, "capacitor": 0.025, "led": 0.015, "inductor": 0.05, "power_inductor": 0.07}
    for name, fam in FAMILIES.items():
        assert fam.edge_inset == insets.get(name, 0.07)
        assert fam.grain_sd == grains.get(name, 0.06)
        assert (fam.kind == fp.KIND_IC) == (name in ("qfn", "soic", "sot", "sot6", "tht_pin"))  # gloss, pin-1, silk dot
        assert fam.printed == (name in ("resistor", "array_r"))


def test_every_kind_has_a_body_drawer():
    from simple_sim.generators.opencv2d.photoreal import parts

    assert set(parts._BODY_DRAWERS) == set(BODY_KINDS)


def test_blender_part_table():
    """The Blender build (D35) is a row of the table; render_batch.py no longer types its own."""
    from simple_sim.board.footprints import BLENDER_CHIP, BLENDER_GENERIC, scene_part

    src = (ROOT / "simple_sim" / "blender" / "render_batch.py").read_text()
    for gone in ("_HEIGHT_RATIO", "_RESISTOR_CODES", "_MARK_CHARS", '"capacitor"', '"sot"', '"soic"'):
        assert gone not in src
    ratios = {n: f.blender.height_ratio for n, f in FAMILIES.items() if f.blender is not BLENDER_GENERIC}
    assert ratios == {"resistor": 0.56, "capacitor": 0.56, "sot": 0.75, "soic": 0.38, "qfn": 0.18, "tht_pin": 0.0}
    assert {n for n, f in FAMILIES.items() if f.blender.body == "tht"} == {"tht_pin"}
    # X12: the Blender chip height (0.56) differs from the 2D tombstone thickness ratio (0.55) on purpose.
    from simple_sim.generators.opencv2d.photoreal.parts import THICKNESS_RATIO
    assert THICKNESS_RATIO == 0.55 and FAMILIES["resistor"].blender.height_ratio == 0.56
    assert {n for n, f in FAMILIES.items() if f.blender.body == BLENDER_CHIP} == {"resistor", "capacitor"}
    assert {n for n, f in FAMILIES.items() if f.blender.leads} == {"sot", "soic"}
    assert scene_part("unknown") == scene_part("sot6")  # generic moulded IC