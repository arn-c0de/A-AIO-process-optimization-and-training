"""tools/eval_real.py: scoring a model on folders of real crops."""

import cv2
import numpy as np
import torch

from tools.eval_real import class_folders, evaluate, format_report, roi_size

CLASSES = ["OK", "MISSING", "MISALIGNED"]


class _BrightIsOk(torch.nn.Module):
    """Predicts OK for bright images, MISSING for dark ones (on the normalised input)."""

    def forward(self, x):
        bright = (x.mean(dim=(1, 2, 3)) > 0).float()
        return torch.stack([bright, 1 - bright, torch.zeros_like(bright)], dim=1)


def _write(folder, value, n, size=40):
    folder.mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(folder / f"{i}.png"), np.full((size, size, 3), value, np.uint8))


def test_evaluate_scores_groups_and_binary_rates(tmp_path):
    _write(tmp_path / "OK", 230, 4)
    _write(tmp_path / "MISSING", 10, 3)
    _write(tmp_path / "MISSING@empty_pad", 230, 2)  # looks good: escapes
    assert [(g, c, len(f)) for g, c, f in class_folders(tmp_path)] == [
        ("MISSING", "MISSING", 3), ("MISSING@empty_pad", "MISSING", 2), ("OK", "OK", 4)]
    result = evaluate(_BrightIsOk(), CLASSES, tmp_path, (64, 64), limit=10)
    groups = result["groups"]
    assert groups["OK"]["accuracy"] == 1.0 and groups["MISSING"]["accuracy"] == 1.0
    assert groups["MISSING@empty_pad"]["predicted_ok"] == 1.0
    assert result["false_call_rate"] == 0.0 and result["escape_rate"] == 2 / 5
    assert "escape rate" in format_report(result)


def test_roi_size_from_checkpoint_config():
    assert roi_size({"config": {"roi": {"width_px": 128, "height_px": 96}}}, 256) == (128, 96)
    assert roi_size({}, 256) == (256, 256)
