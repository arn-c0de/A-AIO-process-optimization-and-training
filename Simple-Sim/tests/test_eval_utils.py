"""Tests for simple_sim.eval_utils and the importable cores of the train/eval/predict scripts."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.cli_utils import UsageError, parse_result, run_cli
from simple_sim.eval_utils import (
    ThroughputMeter,
    resolve_bundle_checkpoint,
    run_inference,
    top_prediction,
    unpack_batch,
)


def test_unpack_batch_variants():
    x, y = torch.zeros(2, 3), torch.tensor([0, 1])
    assert unpack_batch((x, y))[2:] == (None, None)
    _, _, ids, paths = unpack_batch((x, y, ["a", "b"], ["a.png", "b.png"]))
    assert ids == ["a", "b"] and paths == ["a.png", "b.png"]


def test_throughput_meter_ema():
    m = ThroughputMeter()
    assert m.update(10, 1.0) == pytest.approx(10.0)
    assert m.update(20, 1.0) == pytest.approx(11.0)


def test_run_inference_collects_preds_and_loss():
    # Identity-like linear model: argmax of the input one-hot is the prediction.
    model = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(3))
    x = torch.eye(3).repeat(2, 1)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    res = run_inference(model, loader, torch.device("cpu"), criterion=nn.CrossEntropyLoss())
    assert res.num_samples == 6
    assert np.array_equal(res.y_true, y.numpy())
    assert np.array_equal(res.y_pred, y.numpy())
    assert res.avg_loss is not None and res.avg_loss > 0
    assert run_inference(model, loader, torch.device("cpu")).avg_loss is None


def test_top_prediction():
    name, conf, probs = top_prediction(np.array([0.1, 0.7, 0.2]), ["A", "B", "C"])
    assert (name, conf) == ("B", pytest.approx(0.7))
    assert set(probs) == {"A", "B", "C"}


def test_resolve_bundle_checkpoint(tmp_path):
    plain = tmp_path / "m.pt"
    assert resolve_bundle_checkpoint(plain, "p@1") == plain
    bundle = tmp_path / "m.bundle"
    bundle.mkdir()
    with pytest.raises(FileNotFoundError, match="no checkpoint for profile"):
        resolve_bundle_checkpoint(bundle, "p@1")
    (bundle / "p@1.pt").write_bytes(b"x")
    assert resolve_bundle_checkpoint(bundle, "p@1") == bundle / "p@1.pt"


def test_run_cli_json_success_and_error(capsys):
    assert run_cli("x", lambda: {"a": 1}, json_mode=True) == {"a": 1}
    assert parse_result(capsys.readouterr().out) == {"ok": True, "command": "x", "a": 1}

    def boom():
        raise FileNotFoundError("nope")

    with pytest.raises(SystemExit) as e:
        run_cli("x", boom, json_mode=True)
    assert e.value.code == 1
    res = parse_result(capsys.readouterr().out)
    assert res["ok"] is False and "nope" in res["error"]


def test_run_cli_human_mode():
    with pytest.raises(SystemExit, match="bad args"):
        run_cli("x", lambda: (_ for _ in ()).throw(UsageError("bad args")), json_mode=False)
    with pytest.raises(FileNotFoundError):
        run_cli("x", lambda: (_ for _ in ()).throw(FileNotFoundError("x")), json_mode=False)


# ---------------------------------------------------------------------------
# Tiny end-to-end run through the script cores (generate -> train -> eval -> predict -> sample_check)
# ---------------------------------------------------------------------------

TINY_CONFIG = {
    'run': {'run_id': 'tiny', 'seed': 1, 'schema_version': 1},
    'roi': {'width_px': 64, 'height_px': 64, 'mm_per_px': 0.02},
    'classes': {'OK': 4, 'MISSING': 4, 'MISALIGNED': 4, 'TOMBSTONE': 4},
    'tolerances': {'0603': {'ok_shift_px': 3.0, 'ok_rotation_deg': 5.0, 'tombstone_tilt_deg': 75.0}},
    'domains': {'domain_A': {'lighting_brightness': [0.9, 1.1], 'blur_sigma': [0.0, 1.0], 'noise_stddev': [0.0, 5.0]}},
    'splits': {'train_domains': ['domain_A'], 'val_domains': ['domain_A'], 'test_domains': ['domain_A'],
               'train_frac': 0.5, 'val_frac': 0.25, 'test_frac': 0.25},
    'render': {'backend': 'opencv_2d', 'substrate_color': [40, 90, 40], 'copper_color': [60, 120, 180],
               'component_color': [20, 20, 20], 'solder_mask_alpha': 0.3},
    'augment': {'rotation_deg_range': [-5, 5], 'brightness_factor_range': [0.9, 1.1], 'contrast_factor_range': [0.9, 1.1]},
    'train': {'model': 'resnet18', 'pretrained': False, 'epochs': 1, 'batch_size': 8, 'num_workers': 0,
              'lr': 0.001, 'optimizer': 'adam', 'weight_decay': 0.0},
    'eval': {'batch_size': 8, 'num_workers': 0},
}


def test_script_cores_end_to_end(tmp_path):
    from scripts.eval import run_eval
    from scripts.generate import generate_dataset
    from scripts.predict import predict_image
    from scripts.sample_check import sample_check
    from scripts.train import train_model

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump(TINY_CONFIG), encoding="utf-8")
    data = tmp_path / "ds"
    generate_dataset(cfg, data)

    model = tmp_path / "models" / "tiny.pt"
    tr = train_model(data, model, device="cpu")
    assert tr["epochs_run"] == 1 and tr["best_epoch"] == 1
    assert Path(tr["best_checkpoint"]).exists() and Path(tr["last_checkpoint"]).exists()

    ev = run_eval(data, model, "cpu")
    assert Path(ev["report_path"]).exists()
    assert 0.0 <= ev["accuracy"] <= 1.0 and set(ev["per_class"]) == set(TINY_CONFIG["classes"])

    sid = (data / "splits" / "test.txt").read_text().split()[0]
    pr = predict_image(model, data_dir=data, sample_id=sid, device="cpu", topk=2)
    assert pr["pred"] in pr["probs"] and len(pr["topk"]) == 2
    assert pr["gt"] in TINY_CONFIG["classes"] and pr["match"] == (pr["pred"] == pr["gt"])
    with pytest.raises(UsageError):
        predict_image(model, device="cpu")

    sc = sample_check(model, data, per_class=1, device="cpu")
    assert sc["seen"] == len(sc["rows"]) > 0


def test_ensemble_and_bundle_eval_predict(tmp_path):
    """Ensembles (no top-level config/epoch) evaluate and predict; bundle reports sit next to the bundle dir."""
    import shutil

    from scripts.eval import run_eval
    from scripts.generate import generate_dataset
    from scripts.predict import predict_image
    from scripts.train import train_model
    from simple_sim.eval_utils import dataset_profile
    from simple_sim.model_bundle import bundle_checkpoint_path
    from simple_sim.models import ENSEMBLE_FORMAT, load_checkpoint

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump(TINY_CONFIG), encoding="utf-8")
    data = tmp_path / "ds"
    generate_dataset(cfg, data)
    model = tmp_path / "models" / "tiny.pt"
    train_model(data, model, device="cpu")

    ckpt = load_checkpoint(model)
    ens = tmp_path / "models" / "ens.pt"
    torch.save({
        "format": ENSEMBLE_FORMAT, "class_names": ckpt["class_names"],
        "models": [{"profile_id": "a", "checkpoint": ckpt}, {"profile_id": "b", "checkpoint": ckpt}],
        "component_profile": {"profile_id": "multi", "profiles": ["a", "b"]},
    }, ens)
    ev = run_eval(data, ens, "cpu", allow_profile_mismatch=True)
    assert Path(ev["report_path"]) == ens.parent / "report_ens.json"
    sid = (data / "splits" / "test.txt").read_text().split()[0]
    pr = predict_image(ens, data_dir=data, sample_id=sid, device="cpu")  # "multi" skips the profile guard
    assert pr["pred"] in TINY_CONFIG["classes"]

    bundle = tmp_path / "models" / "my.bundle"
    bundle.mkdir()
    shutil.copy(model, bundle_checkpoint_path(bundle, dataset_profile(data)[0]))
    ev = run_eval(data, bundle, "cpu")
    assert Path(ev["report_path"]) == bundle.parent / "report_my.json"
    assert ev["model_path"] == str(bundle) and ev["resolved_model_path"] != str(bundle)
