"""Texts of the pipeline tab's start confirmation (gui.tabs.pipeline.start_summary, Tk-free)."""

from __future__ import annotations

from gui.tabs.pipeline.start_summary import start_summary

_PRECISE = {"total": 40, "per_class": True, "class_names": ["OK", "MISSING"], "classes": {"OK": 30, "MISSING": 10},
            "multi_profiles": [], "multi_totals": {}, "multi_classes": {}}


def _summary(**overrides):
    args = {"task": "full", "out_dir": "outputs/sim_data/runs/r1", "run_mode": "single", "run_count": 1, "dataset_mode": "new",
                "profile_ids": ["0603@1"], "multi_enabled": False, "multi_mode": "separate", "image_filters": {},
                "precise_settings": _PRECISE, "model_path": "outputs/models/r1.pt"}
    args.update(overrides)
    return start_summary(**args)


def test_header_names_task_run_and_model() -> None:
    s = _summary()
    assert "Task: Full Pipeline" in s.header
    assert "Run mode: single (1)" in s.header
    assert "Profiles: 0603@1" in s.header
    assert "Model target: outputs/models/r1.pt" in s.header
    assert "Precise: off" in s.header
    assert s.precise_details == "Precise mode is not active."


def test_generate_only_has_no_model_target_and_continuous_run_label() -> None:
    s = _summary(task="generate_only", run_count=-1, run_mode="continuous")
    assert "Task: Generate Only" in s.header
    assert "(continuous)" in s.header
    assert "Model target: (n/a for generate only)" in s.header


def test_long_profile_list_is_shortened() -> None:
    s = _summary(profile_ids=[f"p{i}" for i in range(8)], multi_enabled=True)
    assert "Profiles: p0, p1, p2, p3, p4 (+3)" in s.header
    assert "Multi-profile: on (separate)" in s.header


def test_filter_details_list_enabled_keys_and_normalise_the_mode() -> None:
    s = _summary(image_filters={"filter_mode": "REALISM", "realism_enabled": True, "enable": True,
                                "enable_blur": True, "enable_noise": False})
    assert "filter_mode=realism" in s.filter_details
    assert "enabled_filter_keys=enable_blur" in s.filter_details
    assert "Filter mode: realism (realism_enabled=True)" in s.header
    assert "filter_mode=custom" in _summary(image_filters={"filter_mode": "bogus"}).filter_details


def test_precise_mode_details() -> None:
    s = _summary(run_mode="precise")
    assert "Precise: on | single profile | per_class=yes" in s.header
    assert "total=40" in s.precise_details
    assert 'classes={"OK": 30, "MISSING": 10}' in s.precise_details
    multi = _summary(run_mode="precise", precise_settings={**_PRECISE, "multi_profiles": ["a", "b"], "per_class": False})
    assert "Precise: on | multi=2 | per_class=no" in multi.header
