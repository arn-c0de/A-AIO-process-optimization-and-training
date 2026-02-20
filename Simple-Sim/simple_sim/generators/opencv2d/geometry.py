"""Nominal geometry sampling for OpenCV 2D renderer."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def sample_nominal_geometry(
    roi_config: Dict[str, Any],
    rng: np.random.Generator,
    geometry_ranges: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Sample nominal component geometry."""
    _ = roi_config  # reserved for future use
    defaults = {
        "pad_width": [25.0, 35.0],
        "pad_height": [30.0, 40.0],
        "pad_spacing": [50.0, 65.0],
        "component_length": [55.0, 65.0],
        "component_width": [25.0, 35.0],
    }
    gr = geometry_ranges or {}

    def _range(name: str) -> tuple[float, float]:
        r = gr.get(name, defaults[name])
        return float(r[0]), float(r[1])

    result = {
        "pad_width": float(rng.uniform(*_range("pad_width"))),
        "pad_height": float(rng.uniform(*_range("pad_height"))),
        "pad_spacing": float(rng.uniform(*_range("pad_spacing"))),
        "component_length": float(rng.uniform(*_range("component_length"))),
        "component_width": float(rng.uniform(*_range("component_width"))),
    }

    if "pad_spacing_y" in gr:
        lo, hi = float(gr["pad_spacing_y"][0]), float(gr["pad_spacing_y"][1])
        result["pad_spacing_y"] = float(rng.uniform(lo, hi))

    return result
