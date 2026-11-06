"""Class names, splits and backends: the closed vocabularies of Simple-Sim (stdlib only).

The enums are ``StrEnum``, so members compare and hash like their string values and
JSON/YAML files keep plain strings. The derived constants below hold plain ``str`` values
(safe for ``yaml.safe_dump``); membership tests work with either form.
"""

from __future__ import annotations

from enum import StrEnum


class DefectClass(StrEnum):
    OK = "OK"
    MISSING = "MISSING"
    MISALIGNED = "MISALIGNED"
    TOMBSTONE = "TOMBSTONE"
    SOLDER_BRIDGE = "SOLDER_BRIDGE"
    CORNER_LIFT = "CORNER_LIFT"
    SOLDER_BALL = "SOLDER_BALL"
    EXCESS_SOLDER = "EXCESS_SOLDER"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Backend(StrEnum):
    OPENCV_2D = "opencv_2d"
    BLENDER_3D = "blender_3d"


def _values(*members: StrEnum) -> tuple[str, ...]:
    return tuple(m.value for m in members)


#: Every class a dataset row may carry (``MetaRow.defect.type``, ``LabelRow.class_name``).
VALID_CLASSES: frozenset[str] = frozenset(_values(*DefectClass))

#: The four original classes, in their canonical order (default class lists, previews, gallery).
STANDARD_CLASSES: tuple[str, ...] = _values(DefectClass.OK, DefectClass.MISSING, DefectClass.MISALIGNED, DefectClass.TOMBSTONE)

#: Classes whose false-negative rate is reported by default (``metrics.compute_metrics``).
CRITICAL_CLASSES: tuple[str, ...] = _values(DefectClass.MISSING, DefectClass.MISALIGNED, DefectClass.TOMBSTONE)

#: Classes labelled by the sampled defect type itself, not by the placement tolerances.
TYPE_LABELLED_CLASSES: frozenset[str] = frozenset(_values(
    DefectClass.SOLDER_BRIDGE, DefectClass.CORNER_LIFT, DefectClass.MISSING,
    DefectClass.SOLDER_BALL, DefectClass.EXCESS_SOLDER,
))

#: Placement classes derived from shift/rotation/tilt via the profile tolerances.
PLACEMENT_CLASSES: frozenset[str] = frozenset(_values(DefectClass.OK, DefectClass.MISALIGNED, DefectClass.TOMBSTONE))

#: Defects drawn as extra solder (balls, excess) on an otherwise OK placement.
SOLDER_ONLY_CLASSES: frozenset[str] = frozenset(_values(DefectClass.SOLDER_BALL, DefectClass.EXCESS_SOLDER))

#: Defects that carry a tilt but must never be drawn as a tombstone.
NEVER_TOMBSTONE: frozenset[str] = frozenset(_values(DefectClass.SOLDER_BRIDGE, DefectClass.CORNER_LIFT))

#: Split names in their canonical order.
SPLIT_NAMES: tuple[str, ...] = _values(*Split)

BACKENDS: tuple[str, ...] = _values(*Backend)

assert VALID_CLASSES == TYPE_LABELLED_CLASSES | PLACEMENT_CLASSES
