"""simple_sim.training.train_profile_model: one-epoch smoke test (guards the shared build_training / fit_epoch contract)."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("freeze", [None, ["stem"]])
def test_train_profile_runs_one_epoch(tmp_path, freeze):
    from simple_sim.training import train_profile_model
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


def test_profile_checkpoint_carries_component_profile(tmp_path):
    """X4: profile classifiers get the same ``component_profile`` block as defect checkpoints."""
    import torch

    from simple_sim.generation.profile_dataset import generate_profile_dataset
    from simple_sim.manifest import read_dataset_manifest, resolve_dataset_profile
    from simple_sim.training import PROFILE_CLASSIFIER, train_profile_model

    cfg = yaml.safe_load((ROOT / "configs" / "run_profile_cls_quick.yaml").read_text(encoding="utf-8"))
    cfg["run"]["component_profiles"] = cfg["run"]["component_profiles"][:2]
    cfg["roi"].update(width_px=48, height_px=48)
    cfg["classes"] = {"OK": 4}
    cfg["train"] = {**(cfg.get("train") or {}), "epochs": 1, "batch_size": 8, "pretrained": False, "num_workers": 0}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    generate_profile_dataset(tmp_path / "cfg.yaml", tmp_path / "ds", project_root=ROOT)
    train_profile_model(tmp_path / "ds", tmp_path / "p.pt", device="cpu", amp=False)

    ckpt = torch.load(tmp_path / "p.pt", map_location="cpu", weights_only=False)
    pid, phash = resolve_dataset_profile(read_dataset_manifest(tmp_path / "ds" / "dataset_manifest.json"))
    assert ckpt["component_profile"] == {"profile_id": pid, "profile_hash": phash}
    assert ckpt["checkpoint_type"] == PROFILE_CLASSIFIER
    assert (tmp_path / "p_last.pt").is_file()
