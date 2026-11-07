"""Colours, light direction, part families and marking vocabularies of the photoreal look (BGR, float32).

Visual targets were calibrated against photos of real assembled boards: dark olive solder mask with
copper traces shimmering through, vias, white silkscreen, bright concave solder fillets, black
resistors with white value codes, tan MLCCs, black molded SOT/SOIC/QFN bodies with laser markings
and metallic gull-wing leads.
"""

from __future__ import annotations

import cv2
import numpy as np

from simple_sim.board.footprints import CHIP_FAMILIES


def jittered(colour: np.ndarray, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    """*colour* scaled by one uniform draw in [*lo*, *hi*): the per-image colour variation of bodies,
    terminals, solder, mask and flux (``colour * float(rng.uniform(lo, hi))``)."""
    return colour * float(rng.uniform(lo, hi))


# Unit light direction (image x, image y, up). Light from the top-left, fairly high.
LIGHT = np.array([-0.40, -0.40, 0.82], dtype=np.float32)
LIGHT /= np.linalg.norm(LIGHT)
# Half vector between the light and the view direction (camera straight above).
HALF = LIGHT + np.array([0.0, 0.0, 1.0], dtype=np.float32)
HALF /= np.linalg.norm(HALF)


# Solder-mask colours (BGR). Median mask pixels of the real crops of the production boards
# (DeepVision xpa top: (64, 89, 58), bottom: (68, 86, 50); hue ~135-150 deg, HSV saturation ~90-110)
# plus a few darker/lighter and slightly teal variants. The earlier palette (red ~30, saturation ~150)
# rendered a far too saturated, cyan-lime mask.
MASK_PALETTE = ((64, 89, 58), (68, 86, 50), (58, 80, 46), (72, 96, 56), (54, 78, 50), (76, 98, 60), (70, 92, 44),
                 (62, 84, 40))


# Mask over copper (traces, pours) relative to mask over bare laminate, measured on the same photos.
COPPER_UNDER_MASK = np.array((1.29, 1.39, 1.40), dtype=np.float32)


SILKSCREEN = np.array((215, 228, 228), dtype=np.float32)


SOLDER = np.array((192, 196, 198), dtype=np.float32)  # real joints: bright grey ~(187, 196, 196), p90 V 228


FINISHES = {"enig": np.array((70, 165, 205), dtype=np.float32), "hasl": np.array((165, 170, 175), dtype=np.float32)}


LEAD = np.array((150, 153, 156), dtype=np.float32)


CAP_BODY = np.array((120, 152, 170), dtype=np.float32)  # tan MLCC ceramic (real crops: (100..132, 134..166, 159..177))


END_CAP = np.array((168, 171, 174), dtype=np.float32)


RESISTOR_CODES = ("103", "472", "1002", "000", "0", "221", "4R7", "100", "330", "10K", "1R0", "5602")


REFDES_PREFIX = {"resistor": "R", "capacitor": "C", "inductor": "L", "power_inductor": "L", "led": "D", "sot": "Q",
                  "soic": "U", "qfn": "U", "sot6": "U", "array_r": "RN",
                  "array_c": "CN"}


# 4-element R/C arrays (chip_array_8): body colour and terminal look.
CAP_ARRAY_BODY = np.array((178, 186, 190), dtype=np.float32)  # light grey-beige ceramic (real crops: near white)


ARRAY_FAMILIES = ("array_r", "array_c")


ARRAY_CODES = ("0", "000", "101", "103", "220", "221", "330", "472", "473", "102", "4R7")


LEADED = ("sot", "soic", "sot6")


# Two-terminal chip parts (chip_2pad): the board model's chip families.
CHIPS = CHIP_FAMILIES


UNMARKED = ("capacitor", "array_c", "inductor", "power_inductor", "led")  # no printed/laser code


# Chip terminal (end cap) length relative to the body; moulded power inductors have theirs underneath.
CAP_LEN = {"capacitor": 0.26, "resistor": 0.22, "inductor": 0.25, "led": 0.16}


LED_LENS = np.array((150, 208, 222), dtype=np.float32)   # yellowish translucent epoxy over the die


MARK_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"


FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX)
