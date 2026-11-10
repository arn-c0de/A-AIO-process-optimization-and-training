"""Texts of the "Start pipeline" confirmation dialog (Tk-free)."""

from __future__ import annotations

import json
from typing import Any, NamedTuple


class StartSummary(NamedTuple):
    header: str
    filter_details: str
    precise_details: str


def start_summary(
    *,
    task: str,
    out_dir: str,
    run_mode: str,
    run_count: int,
    dataset_mode: str,
    profile_ids: list[str],
    multi_enabled: bool,
    multi_mode: str,
    image_filters: dict[str, Any],
    precise_settings: dict[str, Any],
    model_path: str,
) -> StartSummary:
    """Header and collapsible detail texts describing the run that is about to start."""
    task_label = {
        "full": "Full Pipeline",
        "generate_only": "Generate Only",
        "generate_mixed": "Generate Mixed",
    }.get(task, task)
    run_label = "continuous" if run_count == -1 else str(run_count)
    profile_label = ", ".join(profile_ids[:5]) if profile_ids else "-"
    if len(profile_ids) > 5:
        profile_label += f" (+{len(profile_ids) - 5})"
    mode_raw = str(image_filters.get("filter_mode", "custom") or "custom").strip().lower()
    filter_mode = mode_raw if mode_raw in {"custom", "realism"} else "custom"
    realism_enabled = bool(image_filters.get("realism_enabled", False))
    realism_constraints = bool(image_filters.get("realism_constraints_enabled", True))
    realism_profile = str(image_filters.get("realism_profile_id", "profile_industrial_cam"))

    filter_toggles = [k for k, v in image_filters.items() if isinstance(k, str) and k.startswith("enable_") and k != "enable" and bool(v)]
    filter_details = (
        f"filter_mode={filter_mode}\n"
        f"enable={bool(image_filters.get('enable', True))}\n"
        f"realism_enabled={realism_enabled}\n"
        f"realism_constraints_enabled={realism_constraints}\n"
        f"realism_profile_id={realism_profile}\n"
        f"cardinal_rotation_90={bool(image_filters.get('cardinal_rotation_90', True))}\n"
        f"enabled_filter_keys={', '.join(sorted(filter_toggles)) if filter_toggles else '-'}"
    )

    precise_summary = "off"
    precise_details = "Precise mode is not active."
    if run_mode == "precise":
        s = precise_settings
        per_class = bool(s.get("per_class", False))
        multi_profiles = list(s.get("multi_profiles") or [])
        if multi_profiles:
            precise_summary = f"on | multi={len(multi_profiles)} | per_class={'yes' if per_class else 'no'}"
        else:
            precise_summary = f"on | single profile | per_class={'yes' if per_class else 'no'}"
        precise_details = (
            f"total={s.get('total', 100)}\n"
            f"per_class={per_class}\n"
            f"class_names={', '.join(s.get('class_names') or []) or '-'}\n"
            f"classes={json.dumps(s.get('classes') or {}, ensure_ascii=True)}\n"
            f"multi_profiles={json.dumps(multi_profiles, ensure_ascii=True)}\n"
            f"multi_totals={json.dumps(s.get('multi_totals') or {}, ensure_ascii=True)}\n"
            f"multi_classes={json.dumps(s.get('multi_classes') or {}, ensure_ascii=True)}"
        )

    model_target = model_path if task != "generate_only" else "(n/a for generate only)"

    header = (
        f"Task: {task_label}\n"
        f"Run mode: {run_mode} ({run_label})\n"
        f"Dataset mode: {dataset_mode}\n"
        f"Output: {out_dir}\n"
        f"Profiles: {profile_label}\n"
        f"Multi-profile: {'on' if multi_enabled else 'off'} ({multi_mode})\n"
        f"Model target: {model_target}\n"
        f"Filter mode: {filter_mode} (realism_enabled={realism_enabled})\n"
        f"Precise: {precise_summary}"
    )
    return StartSummary(header, filter_details, precise_details)
