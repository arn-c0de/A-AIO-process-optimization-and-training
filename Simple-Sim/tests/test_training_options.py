"""Training-quality options: continuous placement sampling, defect ranges in mm, multi-domain datasets
with a held-out domain, ``render.post_filters: camera``, train-time augmentation, cosine LR schedule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from simple_sim.config import defect_units, domain_camera, placement_sampling, validate_config
from simple_sim.data_loader import IMAGENET_TRANSFORM, ROIDataset, build_train_transform
from simple_sim.defects import classify_defect, fixed_shift_px, sample_defect_params
from simple_sim.generation.common import sample_domain, sample_record
from simple_sim.generators.opencv2d import render_roi, sample_nominal_geometry
from simple_sim.generators.opencv2d.photoreal import apply_view, camera_model, view_canvas
from simple_sim.schema import LabelRow, MetaRow
from simple_sim.splits import generate_splits

ROOT = Path(__file__).resolve().parent.parent
NOMINAL = {"pad_width": 30, "pad_height": 35, "pad_spacing": 55, "component_length": 60, "component_width": 30}
DEFECT = {"type": "OK", "shift_x": 0, "shift_y": 0, "rotation_deg": 0, "tilt_deg": 0}
AUGMENT = {"blur_sigma": 0, "noise_stddev": 0, "brightness_factor": 1.0, "contrast_factor": 1.0, "rotation_deg": 0}
TOL = {"ok_shift_px": 3.0, "ok_rotation_deg": 5.0, "tombstone_tilt_deg": 30.0}
CFG = {"component_color": [20, 20, 20], "substrate_color": [40, 90, 40], "copper_color": [60, 120, 180]}


def _base_config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "run_0001.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- defects.placement_sampling

def test_continuous_sampling_closes_the_gap_at_the_boundary():
    rng = np.random.default_rng(0)
    ok_shift, mis_shift, mis_rot = [], [], []
    for _ in range(3000):
        d = sample_defect_params("OK", rng, TOL, placement="continuous")
        assert classify_defect({}, d, TOL) == "OK"
        ok_shift.append(np.hypot(d["shift_x"], d["shift_y"]))
        d = sample_defect_params("MISALIGNED", rng, TOL, placement="continuous")
        assert classify_defect({}, d, TOL) == "MISALIGNED"
        mis_shift.append(np.hypot(d["shift_x"], d["shift_y"]))
        mis_rot.append(abs(d["rotation_deg"]))
    # OK fills the tolerance (legacy stops at 0.7 x), MISALIGNED starts right behind it (legacy: +1 px).
    assert max(ok_shift) > 0.9 * TOL["ok_shift_px"]
    beyond = [m for m in mis_shift if m > TOL["ok_shift_px"]]
    assert min(beyond) < TOL["ok_shift_px"] + 0.3
    hi = 5 * (TOL["ok_shift_px"] + 1)  # legacy upper bound; u**2 density: median at a quarter of the range
    assert np.median(beyond) < TOL["ok_shift_px"] + 0.3 * (hi - TOL["ok_shift_px"])
    assert max(beyond) > 10.0  # the tail still produces gross misplacements
    assert min(r for r in mis_rot if r > TOL["ok_rotation_deg"]) < TOL["ok_rotation_deg"] + 0.5


def test_continuous_sampling_direction_is_uniform():
    rng = np.random.default_rng(1)
    angles = []
    for _ in range(4000):
        d = sample_defect_params("MISALIGNED", rng, TOL, placement="continuous")
        if np.hypot(d["shift_x"], d["shift_y"]) > TOL["ok_shift_px"]:
            angles.append(np.arctan2(d["shift_y"], d["shift_x"]))
    hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
    assert hist.min() > 0.6 * hist.mean()


def test_legacy_sampling_is_unchanged_and_default():
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    a = [sample_defect_params(c, r1, TOL) for c in ("OK", "MISALIGNED", "TOMBSTONE", "SOLDER_BALL")]
    b = [sample_defect_params(c, r2, TOL, placement="legacy") for c in ("OK", "MISALIGNED", "TOMBSTONE", "SOLDER_BALL")]
    assert a == b
    assert placement_sampling({}) == "legacy"
    with pytest.raises(ValueError):
        sample_defect_params("OK", r1, TOL, placement="random")


def test_placement_sampling_is_validated():
    cfg = _base_config()
    cfg["defects"] = {"placement_sampling": "continuous"}
    validate_config(cfg)
    assert placement_sampling(cfg) == "continuous"
    cfg["defects"] = {"placement_sampling": "nope"}
    with pytest.raises(ValueError, match="placement_sampling"):
        validate_config(cfg)
    cfg["defects"] = {"other": 1}
    with pytest.raises(ValueError, match="defects"):
        validate_config(cfg)


def test_sample_record_uses_the_config_placement():
    cfg = _base_config()
    kw = {"config": cfg, "run_seed": 1, "geometry_ranges": None, "tolerances": TOL, "image_filters": {},
              "enable_cardinal_rotation_90": True, "footprint": "chip_2pad"}
    legacy = [sample_record(i, "MISALIGNED", **kw)[0]["defect"] for i in range(40)]
    cfg["defects"] = {"placement_sampling": "continuous"}
    cont = [sample_record(i, "MISALIGNED", **kw)[0]["defect"] for i in range(40)]
    assert legacy != cont
    assert all(np.hypot(d["shift_x"], d["shift_y"]) >= TOL["ok_shift_px"] + 1 or abs(d["rotation_deg"]) >= TOL["ok_rotation_deg"] + 1
               for d in legacy)


# --------------------------------------------------------------------------- defects.units

FIXED = ("TOMBSTONE", "SOLDER_BRIDGE", "CORNER_LIFT")


def test_units_px_is_default_and_unchanged():
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    a = [sample_defect_params(c, r1, TOL) for c in FIXED]
    b = [sample_defect_params(c, r2, TOL, units="px", mm_per_px=0.01) for c in FIXED]
    assert a == b and r1.bit_generator.state == r2.bit_generator.state
    assert fixed_shift_px("px", None, 3.0) == {"TOMBSTONE": 5.0, "SOLDER_BRIDGE": 1.0, "CORNER_LIFT": 2.0}
    assert defect_units({}) == "px"
    with pytest.raises(ValueError, match="defects.units"):
        sample_defect_params("TOMBSTONE", r1, TOL, units="inch")
    with pytest.raises(ValueError, match="mm_per_px"):
        sample_defect_params("TOMBSTONE", r1, TOL, units="mm")


def test_units_mm_mapping():
    # 0.10 / 0.025 / 0.05 mm at 100 px/mm: 10 / 2.5 / 5 px; bridge and lift capped at 0.7 x ok_shift_px.
    assert fixed_shift_px("mm", 0.01, 10.0) == pytest.approx({"TOMBSTONE": 10.0, "SOLDER_BRIDGE": 2.5, "CORNER_LIFT": 5.0})
    assert fixed_shift_px("mm", 0.01, TOL["ok_shift_px"]) == pytest.approx(
        {"TOMBSTONE": 10.0, "SOLDER_BRIDGE": 2.1, "CORNER_LIFT": 2.1})
    # 45 px/mm (the array profiles): 4.5 / 1.125 / 2.25 px.
    assert fixed_shift_px("mm", 1 / 45, 10.0) == pytest.approx({"TOMBSTONE": 4.5, "SOLDER_BRIDGE": 1.125, "CORNER_LIFT": 2.25})
    rng = np.random.default_rng(3)
    for cls, half in fixed_shift_px("mm", 0.01, 10.0).items():
        for _ in range(200):
            d = sample_defect_params(cls, rng, {**TOL, "ok_shift_px": 10.0}, units="mm", mm_per_px=0.01)
            assert abs(d["shift_x"]) <= half and abs(d["shift_y"]) <= half
            assert classify_defect({}, d, {**TOL, "ok_shift_px": 10.0}) == cls
    # Same number of draws in both modes: the stream continues identically afterwards.
    r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
    for c in FIXED:
        sample_defect_params(c, r1, TOL)
        sample_defect_params(c, r2, TOL, units="mm", mm_per_px=0.01)
    assert r1.bit_generator.state == r2.bit_generator.state


def test_units_are_validated_and_reach_sample_record():
    cfg = _base_config()
    cfg["defects"] = {"placement_sampling": "legacy", "units": "mm"}
    validate_config(cfg)
    assert defect_units(cfg) == "mm"
    cfg["defects"] = {"units": "inch"}
    with pytest.raises(ValueError, match="defects.units"):
        validate_config(cfg)
    cfg = _base_config()  # roi.mm_per_px 0.01
    kw = {"config": cfg, "run_seed": 1, "geometry_ranges": None, "tolerances": TOL, "image_filters": {},
              "enable_cardinal_rotation_90": True, "footprint": "chip_2pad"}
    px = [sample_record(i, "TOMBSTONE", **kw)[0] for i in range(20)]
    cfg["defects"] = {"units": "mm"}
    mm = [sample_record(i, "TOMBSTONE", **kw)[0] for i in range(20)]
    assert all(a["defect"]["shift_x"] * 2 == pytest.approx(b["defect"]["shift_x"]) for a, b in zip(px, mm))
    for a, b in zip(px, mm):  # only the shift differs: same geometry, augmentation, tilt and seed
        assert {k: v for k, v in a.items() if k != "defect"} == {k: v for k, v in b.items() if k != "defect"}
        assert a["defect"]["tilt_deg"] == b["defect"]["tilt_deg"] and a["defect"]["rotation_deg"] == b["defect"]["rotation_deg"]
    ok = [sample_record(i, "OK", **kw)[0] for i in range(5)]
    cfg["defects"] = {}
    assert [sample_record(i, "OK", **kw)[0] for i in range(5)] == ok


# --------------------------------------------------------------------------- domains

def _two_domain_config() -> dict:
    cfg = _base_config()
    cfg["domains"]["domain_B"] = {**cfg["domains"]["domain_A"], "camera": {"preset": "phone"}}
    cfg["splits"]["val_domains"] = ["domain_B"]
    cfg["splits"]["test_domains"] = ["domain_B"]
    return cfg


def test_samples_alternate_between_domains():
    cfg = _two_domain_config()
    assert [sample_domain(cfg, i) for i in range(5)] == ["domain_A", "domain_B", "domain_A", "domain_B", "domain_A"]
    assert sample_domain(_base_config(), 7) == "domain_A"
    rec, _ = sample_record(1, "OK", config=cfg, run_seed=1, geometry_ranges=None, tolerances=TOL, image_filters={},
                           enable_cardinal_rotation_90=True, footprint="chip_2pad", domain="domain_B")
    assert rec["domain"] == "domain_B"
    rec_a, _ = sample_record(1, "OK", config=cfg, run_seed=1, geometry_ranges=None, tolerances=TOL, image_filters={},
                             enable_cardinal_rotation_90=True, footprint="chip_2pad")
    assert rec_a["domain"] == "domain_A" and rec_a["seed"] != rec["seed"]


def test_domain_camera_override_and_validation():
    cfg = _two_domain_config()
    validate_config(cfg)
    assert domain_camera(cfg["render"], cfg["domains"]["domain_B"]) == {"preset": "phone"}
    assert domain_camera(cfg["render"], cfg["domains"]["domain_A"]) is None
    assert domain_camera({"camera": {"zoom": [1.0, 1.2]}}, {"camera": {"preset": "phone"}}) == {"zoom": [1.0, 1.2], "preset": "phone"}
    cfg["domains"]["domain_B"]["camera"] = {"preset": "dslr"}
    with pytest.raises(ValueError, match="domains.domain_B.camera"):
        validate_config(cfg)
    cfg = _two_domain_config()
    cfg["render"] = {"backend": "blender_3d", "blender": {}}
    with pytest.raises(ValueError, match="opencv_2d"):
        validate_config(cfg)


def _rows(domains: list[str], n: int):
    meta, labels = [], []
    for i in range(n):
        d = domains[i % len(domains)]
        sid = f"r/{d}/x/{i:06d}"
        meta.append(MetaRow(schema_version=2, id=sid, run_id="r", domain=d, split="train", seed=i, image_path="images/x.png",
                            render_backend="opencv_2d", footprint="chip_2pad", nominal=NOMINAL, defect=DEFECT, augment=AUGMENT))
        labels.append(LabelRow(schema_version=1, id=sid, class_name="OK"))
    return meta, labels


def test_held_out_domain_is_split_between_val_and_test_only():
    cfg = {"splits": {"train_domains": ["domain_A"], "val_domains": ["domain_B"], "test_domains": ["domain_B"],
                      "train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15}}
    meta, labels = _rows(["domain_A", "domain_B"], 40)
    splits = generate_splits(meta, labels, cfg, 0)
    assert all("domain_A" in s for s in splits["train"]) and len(splits["train"]) == 20
    assert len(splits["val"]) == 10 and len(splits["test"]) == 10
    assert all("domain_B" in s for s in splits["val"] + splits["test"])


def test_shared_domain_splits_as_before():
    cfg = {"splits": {"train_domains": ["domain_A"], "val_domains": ["domain_A"], "test_domains": ["domain_A"],
                      "train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15}}
    meta, labels = _rows(["domain_A"], 40)
    a = generate_splits(meta, labels, cfg, 0)
    b = generate_splits(meta, labels, {"splits": {"train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15}}, 0)
    assert a == b and len(a["train"]) == 28


def test_xdomain_example_config_is_valid():
    cfg = yaml.safe_load((ROOT / "configs" / "run_0001_xdomain.yaml").read_text(encoding="utf-8"))
    validate_config(cfg)
    assert placement_sampling(cfg) == "continuous" and cfg["render"]["post_filters"] == "camera"


# --------------------------------------------------------------------------- render.post_filters

def test_roll_quarter_turns_are_exact_and_jitter_matches_the_legacy_rotation():
    import cv2

    img = np.random.default_rng(0).integers(0, 255, (120, 100, 3)).astype(np.uint8)
    for roll in (90.0, 180.0, 270.0, -90.0):
        ref = np.rot90(img, int(round(roll / 90)) % 4)
        pose = {"tilt": 0.0, "azimuth": 0.0, "zoom": 1.0, "roll": roll}
        assert np.array_equal(apply_view(img, pose, (ref.shape[1], ref.shape[0])), ref)
    sq = img[:100, :100]
    out = apply_view(sq, {"tilt": 0.0, "azimuth": 0.0, "zoom": 1.0, "roll": 7.0}, (100, 100))
    ref = cv2.warpAffine(sq, cv2.getRotationMatrix2D((50, 50), 7.0, 1.0), (100, 100), borderMode=cv2.BORDER_REFLECT101)
    assert np.abs(out[20:80, 20:80].astype(int) - ref[20:80, 20:80]).mean() < 1.0
    assert view_canvas((256, 256), {"tilt": 0.0, "zoom": 1.0, "roll": 95.0}) > (256, 256)
    assert view_canvas((256, 200), {"tilt": 0.0, "zoom": 1.0, "roll": 90.0}) == (200, 256)


def test_camera_model_extra_noise_and_blur():
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    img[:, 32:] = 60
    plain = camera_model(img, np.random.default_rng(0)).astype(float)
    noisy = camera_model(img, np.random.default_rng(0), extra_noise_var=64.0).astype(float)
    soft = camera_model(img, np.random.default_rng(0), extra_blur=3.0).astype(float)
    assert noisy[:, :20].std() > 2 * plain[:, :20].std()
    assert np.abs(np.diff(soft[32, 20:44, 0])).max() < np.abs(np.diff(plain[32, 20:44, 0])).max()


def _render(mode: str, seed: int = 3, rotation: float = 94.0):
    rng = np.random.default_rng(seed)
    nominal = sample_nominal_geometry({}, rng, None)
    defect = sample_defect_params("OK", rng, TOL)
    aug = {"rotation_deg": rotation, "blur_sigma": 0.8, "noise_stddev": 8.0}
    return render_roi(nominal, defect, aug, (96, 96), {**CFG, "post_filters": mode}, TOL, np.random.default_rng(seed), "chip_2pad")


def test_post_filters_camera_renders_once_noise_and_is_deterministic():
    legacy, cam = _render("legacy"), _render("camera")
    assert legacy.shape == cam.shape == (96, 96, 3)
    assert not np.array_equal(legacy, cam)
    assert np.array_equal(cam, _render("camera"))
    # Same scene brightness; the board around the part differs (the rolled view renders a larger canvas).
    assert abs(float(legacy.mean()) - float(cam.mean())) < 15.0
    # A quarter turn is exact: the 90 deg camera-mode render is the 0 deg one turned, up to the camera noise.
    turned, straight = _render("camera", rotation=90.0).astype(float), _render("camera", rotation=0.0).astype(float)
    assert np.abs(turned - np.rot90(straight)).mean() < 2.0 * np.abs(turned[1:] - turned[:-1]).mean()
    with pytest.raises(ValueError, match="post_filters"):
        _render("sometimes")


def test_post_filters_is_validated():
    cfg = _base_config()
    cfg["render"]["post_filters"] = "camera"
    validate_config(cfg)
    cfg["render"]["style"] = "flat"
    with pytest.raises(ValueError, match="post_filters"):
        validate_config(cfg)
    cfg = _base_config()
    cfg["render"]["post_filters"] = "both"
    with pytest.raises(ValueError, match="post_filters"):
        validate_config(cfg)


# --------------------------------------------------------------------------- training

def test_train_transform_defaults_and_augmentations():
    assert build_train_transform(None).transforms[0].__class__ is IMAGENET_TRANSFORM.transforms[0].__class__
    assert len(build_train_transform({}).transforms) == 2
    t = build_train_transform({"hflip": True, "vflip": True, "rot90": True, "brightness": 0.2, "hue": 0.05})
    assert len(t.transforms) == 6
    img = np.random.default_rng(0).integers(0, 255, (32, 32, 3)).astype(np.uint8)
    torch.manual_seed(0)
    a = t(img)
    torch.manual_seed(0)
    assert torch.equal(a, t(img)) and a.shape == (3, 32, 32)
    # Non-square images keep their shape (only half turns) so batches collate.
    rect = np.zeros((32, 48, 3), dtype=np.uint8)
    for _ in range(8):
        assert t(rect).shape == (3, 32, 48)


def test_train_options_are_validated():
    cfg = _base_config()
    cfg["train"].update({"scheduler": "cosine", "warmup_epochs": 1, "label_smoothing": 0.1,
                         "augment": {"hflip": True, "rot90": True, "brightness": 0.2}})
    validate_config(cfg)
    cfg["train"].update({"freeze": ["stem", "layer1"], "cache_images": True})
    validate_config(cfg)
    for bad in ({"scheduler": "step"}, {"warmup_epochs": 10}, {"warmup_epochs": -1}, {"label_smoothing": 1.0},
                {"augment": {"flip": True}}, {"augment": {"hue": 0.9}}, {"augment": {"hflip": "yes"}},
                {"freeze": "stem"}, {"freeze": ["fc"]}, {"cache_images": "yes"}):
        cfg = _base_config()
        cfg["train"].update(bad)
        with pytest.raises(ValueError, match="train."):
            validate_config(cfg)


def test_cosine_lr_factor():
    from simple_sim.training import LR_FLOOR, build_scheduler, lr_factor

    f = [lr_factor(e, 10, scheduler="cosine", warmup_epochs=2) for e in range(10)]
    assert f[0] == pytest.approx(0.5) and f[1] == pytest.approx(1.0) and f[2] == pytest.approx(1.0)
    assert all(x > y for x, y in zip(f[2:], f[3:])) and f[-1] == pytest.approx(LR_FLOOR)
    assert [lr_factor(e, 4) for e in range(4)] == [1.0] * 4
    assert lr_factor(0, 1, scheduler="cosine") == pytest.approx(1.0)
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    assert build_scheduler(opt, {"scheduler": "none"}, 5) is None
    sched = build_scheduler(opt, {"scheduler": "cosine", "warmup_epochs": 1}, 5)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)  # warm-up over one epoch: full lr from the start
    opt.step()  # as in training: the optimizer steps before the scheduler (torch warns otherwise)
    for _ in range(4):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1 * LR_FLOOR)


def test_pipeline_with_all_options(tmp_path):
    """generate -> train -> eval with the xdomain options on a tiny dataset (CPU, 1 epoch)."""
    from simple_sim.evaluation import run_eval
    from simple_sim.generation.dataset import generate_dataset
    from simple_sim.training import train_model

    cfg = yaml.safe_load((ROOT / "configs" / "run_0001_xdomain.yaml").read_text(encoding="utf-8"))
    cfg["roi"] = {"width_px": 64, "height_px": 64, "mm_per_px": 0.04}
    cfg["classes"] = {"OK": 6, "MISSING": 6, "MISALIGNED": 6}
    cfg["train"].update({"epochs": 2, "batch_size": 4, "pretrained": False, "warmup_epochs": 1,
                         "freeze": ["stem", "layer1"], "cache_images": True})
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    data = tmp_path / "ds"
    res = generate_dataset(cfg_path, data, project_root=ROOT)
    assert res["splits"] == {"train": 9, "val": 5, "test": 4} or sum(res["splits"].values()) == 18
    ids = (data / "splits" / "train.txt").read_text().split()
    assert ids and all("/domain_A/" in s for s in ids)
    held = (data / "splits" / "val.txt").read_text().split() + (data / "splits" / "test.txt").read_text().split()
    assert held and all("/domain_B/" in s for s in held)
    model = tmp_path / "m.pt"
    threads_before = torch.get_num_threads()
    tr = train_model(data, model, device="cpu", threads=2)
    assert torch.get_num_threads() == 2
    torch.set_num_threads(threads_before)
    assert tr["epochs_run"] == 2 and model.exists()
    ev = run_eval(data, model, device="cpu")
    assert 0.0 <= float(ev["accuracy"]) <= 1.0
    # train.cache_images: the second read of a sample comes from RAM and is the same tensor.
    ds = ROIDataset(data, "train", sorted(cfg["classes"]), cache_images=True)
    first = ds[0][0]
    assert len(ds._rgb_cache) == 1
    assert torch.equal(first, ds[0][0]) and len(ds._rgb_cache) == 1
    plain = ROIDataset(data, "train", sorted(cfg["classes"]))
    assert torch.equal(first, plain[0][0]) and not plain._rgb_cache


def test_freeze_stages_stops_gradients_and_keeps_bn_stats():
    from simple_sim.models import RESNET_STAGES, build_resnet18, freeze_stages, trainable_parameters
    from simple_sim.training import train_epoch

    model = build_resnet18(3)
    n_all = len(list(model.parameters()))
    frozen = freeze_stages(model, ["stem", "layer1"])
    assert [type(m).__name__ for m in frozen] == ["Conv2d", "BatchNorm2d", "Sequential"]
    assert all(not p.requires_grad for m in frozen for p in m.parameters())
    assert all(p.requires_grad for p in model.layer2.parameters())
    assert 0 < len(trainable_parameters(model)) < n_all
    assert freeze_stages(model, None) == [] and RESNET_STAGES[0] == "stem"
    with pytest.raises(ValueError, match="train.freeze"):
        freeze_stages(model, ["fc"])

    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam(trainable_parameters(model), lr=0.01)
    x = torch.randn(4, 3, 32, 32)
    loader = [(x, torch.tensor([0, 1, 2, 0]))]
    train_epoch(model, loader, torch.nn.CrossEntropyLoss(), opt, torch.device("cpu"), frozen=frozen)  # type: ignore[arg-type]
    after = model.state_dict()
    assert torch.equal(before["conv1.weight"], after["conv1.weight"])
    assert torch.equal(before["bn1.running_mean"], after["bn1.running_mean"])  # eval mode: stats untouched
    assert torch.equal(before["layer1.0.bn1.running_mean"], after["layer1.0.bn1.running_mean"])
    assert not torch.equal(before["layer2.0.conv1.weight"], after["layer2.0.conv1.weight"])
    assert not torch.equal(before["layer2.0.bn1.running_mean"], after["layer2.0.bn1.running_mean"])
    assert model.training  # the model as a whole is still in train mode


# --------------------------------------------------------------------------- review fixes (training comparability)

def test_val_loss_is_plain_cross_entropy_under_label_smoothing():
    """fit_epoch validates with plain CE even when the training criterion smooths labels."""
    from simple_sim.models import build_resnet18
    from simple_sim.training import EpochRun, fit_epoch, plain_criterion

    torch.manual_seed(0)
    model = build_resnet18(3)
    x, y = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 0])
    smoothed = torch.nn.CrossEntropyLoss(label_smoothing=0.3)
    assert plain_criterion(smoothed).label_smoothing == 0.0
    plain = torch.nn.CrossEntropyLoss()
    assert plain_criterion(plain) is plain

    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    run = EpochRun(model, [], [(x, y)], smoothed, opt, torch.device("cpu"), ["a", "b", "c"],  # type: ignore[arg-type]
                   total_epochs=1, emit_events=False)
    stats = fit_epoch(run, 0)
    model.eval()
    with torch.no_grad():
        expected = float(torch.nn.functional.cross_entropy(model(x), y))
        smoothed_val = float(smoothed(model(x), y))
    assert stats["val_loss"] == pytest.approx(expected, rel=1e-5)
    assert stats["val_loss"] != pytest.approx(smoothed_val, rel=1e-3)


def test_epoch_loss_is_sample_weighted():
    from simple_sim.eval_utils import run_inference
    from simple_sim.models import build_resnet18
    from simple_sim.training import train_epoch

    torch.manual_seed(0)
    model = build_resnet18(2)
    big = (torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 0]))
    small = (torch.randn(2, 3, 32, 32), torch.tensor([1, 1]))  # BatchNorm needs > 1 sample in train mode
    crit = torch.nn.CrossEntropyLoss()
    res = run_inference(model, [big, small], torch.device("cpu"), criterion=crit)  # type: ignore[arg-type]
    model.eval()
    with torch.no_grad():
        per_sample = torch.cat([torch.nn.functional.cross_entropy(model(x), y, reduction="none") for x, y in (big, small)])
    assert res.avg_loss == pytest.approx(float(per_sample.mean()), rel=1e-5)
    assert res.num_samples == 5

    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr 0: the weights stay, losses are comparable
    loss, seen = train_epoch(model, [big, small], crit, opt, torch.device("cpu"))  # type: ignore[arg-type]
    assert seen == 5
    model.train()
    with torch.no_grad():
        # train mode (BatchNorm batch stats), same as the epoch saw; weights unchanged under lr 0
        ps = torch.cat([torch.nn.functional.cross_entropy(model(x), y, reduction="none") for x, y in (big, small)])
    assert loss == pytest.approx(float(ps.mean()), rel=1e-3)


def test_resume_takes_lr_from_config_and_keeps_the_resumed_best(tmp_path):
    """Resume: the dataset config's lr / weight_decay win over the optimizer state in the checkpoint,
    and a worse epoch after resume does not replace the resumed best score (also with --out-mode last)."""
    from simple_sim.generation.dataset import generate_dataset
    from simple_sim.models import load_checkpoint
    from simple_sim.training import train_model

    cfg = _base_config()
    cfg["roi"] = {"width_px": 64, "height_px": 64, "mm_per_px": 0.04}
    cfg["classes"] = {"OK": 4, "MISSING": 4, "MISALIGNED": 4}
    cfg["train"].update({"epochs": 1, "batch_size": 4, "pretrained": False, "num_workers": 0, "lr": 0.01})
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    data = tmp_path / "ds"
    generate_dataset(cfg_path, data, project_root=ROOT)

    first = tmp_path / "first.pt"
    tr = train_model(data, first, device="cpu")
    assert tr["epochs_run"] == 1
    ck1 = load_checkpoint(first)
    assert ck1["best_epoch"] == 1 and ck1["optimizer_state_dict"]["param_groups"][0]["lr"] == pytest.approx(0.01)

    # Change lr and weight_decay in the dataset config, resume into a new --out with one extra epoch.
    ds_cfg_path = data / "config.yaml"
    ds_cfg = yaml.safe_load(ds_cfg_path.read_text(encoding="utf-8"))
    ds_cfg["train"]["lr"] = 0.00123
    ds_cfg["train"]["weight_decay"] = 0.5
    ds_cfg_path.write_text(yaml.safe_dump(ds_cfg), encoding="utf-8")

    second = tmp_path / "second.pt"
    tr2 = train_model(data, second, resume=first, extra_epochs=1, device="cpu", out_mode="last")
    assert tr2["start_epoch"] == 1 and tr2["epochs_run"] == 1
    ck2 = load_checkpoint(second)
    group = ck2["optimizer_state_dict"]["param_groups"][0]
    assert group["lr"] == pytest.approx(0.00123) and group["weight_decay"] == pytest.approx(0.5)
    assert group["initial_lr"] == pytest.approx(0.00123)
    # --out-mode last writes epoch 2 to --out, but the best bookkeeping stays honest.
    assert ck2["epoch"] == 2
    assert ck2["best_val_f1"] == pytest.approx(max(ck1["best_val_f1"], ck2["val_f1"]))
    assert ck2["best_epoch"] == (2 if ck2["val_f1"] > ck1["best_val_f1"] else 1)
    assert tr2["best_val_f1"] == pytest.approx(ck2["best_val_f1"])


def test_pretrained_download_failure_is_an_error(monkeypatch):
    from torchvision import models as tv_models

    from simple_sim.models import build_resnet18

    def boom(*_a, **kw):
        if kw.get("weights") is not None:
            raise OSError("no network")
        return original(*_a, **kw)

    original = tv_models.resnet18
    monkeypatch.setattr(tv_models, "resnet18", boom)
    with pytest.raises(RuntimeError, match="train.pretrained: false"):
        build_resnet18(3, pretrained=True)
    assert build_resnet18(3, pretrained=False).fc.out_features == 3
