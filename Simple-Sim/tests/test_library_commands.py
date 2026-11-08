"""Phase 4/5 library cores: inference (L7/X1), batch-predict options (L3), OOM retry, model store (X6)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from simple_sim.data_loader import preprocess_bgr
from simple_sim.eval_utils import is_cuda_oom, oom_retry, predict_probs
from simple_sim.model_store import SNAPSHOT_RUN_RE, snapshot_checkpoint, snapshot_path
from simple_sim.models import build_resnet18

CLASSES = ["MISSING", "OK", "TOMBSTONE"]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    torch.manual_seed(0)
    model = build_resnet18(len(CLASSES))
    path = tmp_path_factory.mktemp("ckpt") / "m.pt"
    torch.save({"model_state_dict": model.state_dict(), "class_names": CLASSES, "epoch": 1}, path)
    return path


@pytest.fixture(scope="module")
def image(tmp_path_factory) -> Path:
    rng = np.random.default_rng(0)
    path = tmp_path_factory.mktemp("img") / "roi.png"
    cv2.imwrite(str(path), rng.integers(0, 256, (48, 80, 3), dtype=np.uint8))  # not 256 px
    return path


def test_classifier_uses_the_training_transform(checkpoint, image):
    from simple_sim.inference import Classifier

    clf = Classifier(checkpoint)
    pred = clf.predict_path(image)
    ref = predict_probs(clf.model, preprocess_bgr(cv2.imread(str(image))).unsqueeze(0))[0]
    np.testing.assert_array_equal(pred.probs_array, ref)
    assert pred.label == CLASSES[int(np.argmax(ref))]
    assert pred.confidence == pytest.approx(float(ref.max()))
    assert list(pred.probs) == CLASSES
    assert [n for n, _ in pred.topk(2)] == [CLASSES[i] for i in np.argsort(-ref)[:2]]


def test_gui_model_wrapper_matches_cli_prediction(checkpoint, image):
    """X1: the GUI no longer resizes to 256 px; it predicts exactly what ``predict`` predicts."""
    from gui.utils.model_inference import ModelWrapper, predict_image as gui_predict
    from simple_sim.inference import predict_image

    cli = predict_image(checkpoint, image_path=image)
    wrapper = ModelWrapper(checkpoint)
    label, conf, probs = wrapper.predict(image)
    assert (label, conf) == (cli["pred"], cli["confidence"])
    assert dict(zip(wrapper.class_names, map(float, probs))) == cli["probs"]
    assert gui_predict(wrapper, image)["predicted_class"] == cli["pred"]
    with pytest.raises(ValueError):
        wrapper.predict(image.parent / "missing.png")


def test_classifier_missing_model(tmp_path):
    from simple_sim.inference import Classifier

    with pytest.raises(FileNotFoundError):
        Classifier(tmp_path / "nope.pt")


# --------------------------------------------------------------------------- OOM retry

def test_oom_retry_halves_on_cuda_oom_only():
    seen: list[int] = []

    def fn(bs: int) -> int:
        seen.append(bs)
        if bs > 8:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return bs

    assert oom_retry(fn, 64, SimpleNamespace(type="cuda")) == 8
    assert seen == [64, 32, 16, 8]
    with pytest.raises(RuntimeError, match="out of memory"):  # CPU: never retried
        oom_retry(fn, 64, torch.device("cpu"))
    with pytest.raises(ValueError):
        oom_retry(lambda bs: (_ for _ in ()).throw(ValueError("x")), 4, SimpleNamespace(type="cuda"))
    with pytest.raises(RuntimeError, match="batch_size=1"):
        oom_retry(lambda bs: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")), 2, SimpleNamespace(type="cuda"))
    assert is_cuda_oom(RuntimeError("cuDNN error: out of memory")) and not is_cuda_oom(ValueError("out of memory cuda"))


# --------------------------------------------------------------------------- batch-predict options

def test_batch_predict_options_defaults_match_the_command():
    from simple_sim.batch_predict import BatchPredictOptions
    from simple_sim.commands.batch_predict import build_parser, options_from_args

    args = build_parser().parse_args(["--model", "m.pt", "--data", "d", "--device", "cpu"])
    assert options_from_args(args) == BatchPredictOptions(model=Path("m.pt"), data=Path("d"))
    args = build_parser().parse_args(["--model", "m.pt", "--data", "d", "--history-dirs", "a, b,", "--out-dir", "o"])
    opts = options_from_args(args)
    assert opts.history_dirs == (Path("a"), Path("b")) and opts.out_dir == Path("o")


@pytest.mark.parametrize("bad", [{"split": "dev"}, {"history_scope": "x"}, {"history_metric": "f2"}, {"history_split": "x"}])
def test_batch_predict_options_validate(bad):
    from simple_sim.batch_predict import BatchPredictOptions

    with pytest.raises(ValueError):
        BatchPredictOptions(model=Path("m"), data=Path("d"), **bad)


def test_eval_report_paths():
    from simple_sim.evaluation import report_path_for

    assert report_path_for(Path("/m/run.pt")) == Path("/m/report_run.json")
    assert report_path_for(Path("/m/b.bundle")) == Path("/m/report_b.json")
    assert report_path_for(Path("/m/b.bundle"), profile_id="chip_0603_resistor@1") == Path("/m/report_b_chip_0603_resistor@1.json")
    assert report_path_for(Path("/m/p.pt"), profile_classifier=True) == Path("/m/report_profile_classifier_p.json")


# --------------------------------------------------------------------------- model store

def test_snapshot_naming_is_shared(tmp_path):
    """X6: manual (Weights tab) and pipeline snapshots use one scheme in versions/<stem>/."""
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    root = tmp_path / "models"
    assert snapshot_path(root, model, tag="20260101_000000") == root / "versions" / "m" / "m_20260101_000000.pt"
    run = snapshot_path(root, model, run=7, tag="20260101_000000")
    assert run.name == "m_20260101_000000_run007.pt" and SNAPSHOT_RUN_RE.search(run.name)["run"] == "007"
    a = snapshot_checkpoint(root, model)
    b = snapshot_checkpoint(root, model)
    assert a != b and a.read_bytes() == b.read_bytes() == b"x"


def test_gui_snapshots_go_through_model_store(tmp_path):
    import queue

    from gui.tabs.weights.logic import WeightsLogic

    model = tmp_path / "outputs" / "models" / "m.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x")
    snap = WeightsLogic(tmp_path, queue.Queue()).snapshot_active_model(model)
    assert snap.parent == tmp_path / "outputs" / "models" / "versions" / "m"
    assert snap.name.startswith("m_") and snap.read_bytes() == b"x"


def test_weights_predict_command_uses_json_result_line(tmp_path):
    """P4/D22: ``batch-predict --json`` via the GUI's interpreter; v2 datasets resolve to ``multi`` (X7)."""
    import queue
    import sys

    from gui.tabs.weights.logic import WeightsLogic
    from simple_sim.cli_utils import RESULT_JSON_PREFIX, parse_result_line

    cmd, out_dir = WeightsLogic(tmp_path, queue.Queue())._predict_cmd(
        tmp_path / "m.pt", tmp_path / "ds", "val", "auto", "5", True)
    assert cmd[:4] == [sys.executable, "-m", "simple_sim", "batch-predict"]
    assert cmd[-1] == "--json" and "--device" not in cmd and cmd[cmd.index("--max-samples") + 1] == "5"
    assert out_dir == tmp_path / "ds" / "predictions" / "weights_tab"
    assert parse_result_line(RESULT_JSON_PREFIX + '{"ok": true, "report_path": "r.json"}\n') == {"ok": True, "report_path": "r.json"}
    assert parse_result_line("Report saved to: r.json") is None
    assert parse_result_line(RESULT_JSON_PREFIX + "{broken") is None
