"""scripts/train_profile.py: one-epoch smoke test (guards the build_training / fit_epoch contract it imports)."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("freeze", [None, ["stem"]])
def test_train_profile_runs_one_epoch(tmp_path, freeze):
    from scripts.train_profile import train_profile_model
    from simple_sim.generation.profile_dataset import generate_profile_dataset

    cfg = yaml.safe_load((ROOT / "configs" / "run_profile_cls_quick.yaml").read_text(encoding="utf-8"))
    cfg["run"]["component_profiles"] = cfg["run"]["component_profiles"][:2]
    cfg["roi"].update(width_px=64, height_px=64)
    cfg["classes"] = {"OK": 6, "MISSING": 6}
    cfg["train"] = {**(cfg.get("train") or {}), "epochs": 1, "batch_size": 8, "pretrained": False, "num_workers": 0}
    if freeze:
        cfg["train"]["freeze"] = freeze
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    generate_profile_dataset(tmp_path / "cfg.yaml", tmp_path / "ds", project_root=ROOT)
    summary = train_profile_model(tmp_path / "ds", tmp_path / "profile.pt", device="cpu", amp=False)
    assert (tmp_path / "profile.pt").is_file()
    assert summary
