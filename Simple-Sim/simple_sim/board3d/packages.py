"""Package name -> body size estimate, 3D family and default body height.

Body sizes normally come from the DeepVision app (the body polygon in ``parts.json``, resolved there by
``modules/package-library``: part override > project > alias > library > IPC-7351 name). This module only fills
the gaps:

* :func:`ipc_dimensions` ports ``ipcDimensions`` of ``package-library/src/library.js`` (same patterns, same
  results; pinned by ``tests/test_board3d_import.py``) for placements the app lists without a body (test points,
  fiducials);
* :func:`package_family` maps a package name to the family that builds it in 3D. Names in
  :data:`simple_sim.board.footprints.FAMILIES` have a Blender builder today; the others
  (:data:`NEW_FAMILIES`) are planned in T2 and render as a generic box until then;
* :func:`default_height_mm` gives a typical body height per package (datasheet-typical values, **not
  measured** on the board; ``heightSource: "default"`` in the scene). T3 replaces them from the photo series.
"""

from __future__ import annotations

import re

from simple_sim.board.footprints import FAMILIES

#: Families the board3d needs that have no Blender builder yet (T2). Value: what the builder will make.
NEW_FAMILIES: dict[str, str] = {
    "electrolytic": "aluminium can on a plastic base, vent cross, polarity stripe",
    "connector": "plastic housing with pins / contacts (geometry and texture from the mounted photos)",
    "crystal": "ceramic package with metal lid",
    "bga": "moulded body on a substrate, balls hidden except at the edge",
    "diode": "moulded two-terminal body with cathode band",
    "filter": "four-terminal chip (EMI filter)",
    "testpoint": "bare round pad, no body",
    "fiducial": "bare copper dot in a mask opening, no body",
    "mounting_hole": "plated hole with annular ring (screw / bolt in the mounted stage)",
    "pads_only": "footprint without a part (programming pads such as Tag-Connect)",
}

# Package name patterns -> family, first match wins (upper-case names).
_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    (r"^RESCA", "array_r"),
    (r"^RES", "resistor"),
    (r"^CAPCA", "array_c"),
    (r"^CAPAE", "electrolytic"),
    (r"^CAP", "capacitor"),
    (r"^INDM", "power_inductor"),
    (r"^IND", "inductor"),
    (r"^LED", "led"),
    (r"^(DIO|SOD)", "diode"),
    (r"^FIL", "filter"),
    (r"^SOT\d+P\d+-3\b", "sot"),
    (r"^SOT\d+P\d+-[456]\b", "sot6"),
    (r"^SOT", "sot"),
    (r"^(SOP|SOIC|TSOP|SSOP)", "soic"),
    (r"^(QFN|SON|DFN)", "qfn"),
    (r"^BGA", "bga"),
    (r"^OSC", "crystal"),
    (r"^(CON|KEY|HDR)", "connector"),
    (r"^TC20\d0", "pads_only"),
    (r"^TP", "testpoint"),
    (r"^FID", "fiducial"),
    (r"^MTG", "mounting_hole"),
)


def package_family(package: str) -> str:
    """3D family of a package name: a :data:`FAMILIES` name, a :data:`NEW_FAMILIES` name or ``generic``.

    (Not :func:`simple_sim.board.footprints.family_of`, which maps a *footprint* plus component type.)
    """
    name = str(package or "").strip().upper()
    for pattern, family in _FAMILY_RULES:
        if re.match(pattern, name):
            return family
    return "generic"


#: Former name, kept for existing importers (tests, other board3d modules).
family_of = package_family


def family_status(family: str) -> str:
    """``builder`` (Blender builder exists), ``planned`` (T2) or ``generic`` (box)."""
    if family in FAMILIES:
        return "builder"
    return "planned" if family in NEW_FAMILIES else "generic"


def _dims(comp_w: float, comp_h: float, body_w: float | None = None, body_h: float | None = None) -> dict[str, float]:
    r = lambda v: round(v * 1000) / 1000  # noqa: E731
    return {"bodyW": r(body_w if body_w is not None else comp_w), "bodyH": r(body_h if body_h is not None else comp_h),
            "compW": r(comp_w), "compH": r(comp_h)}


def ipc_dimensions(package: str) -> dict[str, float | bool] | None:
    """Body / outer size in mm from an IPC-7351 name, or None. Port of ``ipcDimensions`` (package-library):
    width along the part X at 0 deg; ``estimated`` marks rule-of-thumb gull-wing body widths."""
    n = str(package or "").strip().upper()
    if m := re.match(r"^(?:(?:RES|CAP|IND|LED|DIO|FUS|FIL)S?C|(?:LED|DIO)M)(\d{2})(\d{2})(?![\dX])", n):
        return {**_dims(int(m[1]) / 10, int(m[2]) / 10), "estimated": False}
    if (m := re.match(r"^BGA.*_(\d{3,4})X(\d{3,4})", n)) or (m := re.match(r"^(?:QFN|SON|DFN|RESCA|CAPCA|OSC|FIL).*?(\d{3,4})X(\d{3,4})", n)):
        return {**_dims(int(m[1]) / 100, int(m[2]) / 100), "estimated": False}
    if m := re.match(r"^(SOT|SOP|SOIC|TSOP)(\d+)P(\d+)-(\d+)", n):
        kind, p, w, pins = m[1], int(m[2]) / 100, int(m[3]) / 100, int(m[4])
        length = 3 * p + 0.05 if kind == "SOT" else (-(-pins // 2) - 1) * p + 1
        return {**_dims(w, length, w * (0.5 if kind == "SOT" else 0.65), length), "estimated": True}
    if (m := re.match(r"^MTGP(\d{3})", n)) or (m := re.match(r"^CAPAE(\d{3,4})", n)) or (m := re.match(r"^(?:TP|FID)(\d{2,3})", n)):
        d = int(m[1]) / 100
        return {**_dims(d, d), "estimated": False}
    return None


# Typical body heights in mm (datasheet-typical, not measured). Chips: by size code; others: by family.
_CHIP_HEIGHT_MM = {"1005": {"resistor": 0.35, "capacitor": 0.5, "inductor": 0.5, "led": 0.4},
                   "1608": {"resistor": 0.45, "capacitor": 0.8, "inductor": 0.8, "led": 0.6, "filter": 0.6},
                   "2012": {"resistor": 0.5, "capacitor": 1.25, "inductor": 0.9, "filter": 0.85},
                   "2520": {"inductor": 1.0}, "3216": {"resistor": 0.55, "capacitor": 1.6},
                   "3225": {"capacitor": 1.6}}
_FAMILY_HEIGHT_MM = {"sot": 1.1, "sot6": 0.95, "soic": 1.5, "qfn": 0.9, "bga": 1.2, "crystal": 0.9, "array_r": 0.45,
                     "array_c": 0.6, "diode": 1.0, "filter": 0.6, "power_inductor": 3.0, "electrolytic": 5.4, "connector": None,
                     "testpoint": 0.0, "fiducial": 0.0, "mounting_hole": 0.0, "pads_only": 0.0, "generic": None}
# TSSOP / thin SOP: SOP with 0.65 mm pitch or less.
_THIN_SOP = re.compile(r"^(SOP|TSOP)(50|65)P")


def default_height_mm(package: str, family: str, body_w: float, body_h: float) -> float | None:
    """Typical body height (mm) or None when unknown (connectors: from the photos, T3)."""
    n = str(package or "").strip().upper()
    if m := re.search(r"C(\d{4})", n):
        table = _CHIP_HEIGHT_MM.get(m[1], {})
        if family in table:
            return table[family]
    if family == "soic" and _THIN_SOP.match(n):
        return 1.1
    if family == "electrolytic" and (m := re.match(r"^CAPAE(\d{3,4})", n)):
        return round(0.82 * int(m[1]) / 100, 2)      # cans are about as tall as 0.8x their diameter (6.3 x 5.4 mm)
    h = _FAMILY_HEIGHT_MM.get(family)
    if h is None and family in FAMILIES:                # a chip size the table does not list: as high as wide
        return round(min(body_w, body_h), 3)
    return h
