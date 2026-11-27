"""Model groups, favorites and Arena tracking of the weights tab.

Mixin of :class:`gui.tabs.weights.tab.WeightsTab`; it relies on the attributes that class sets up
(``ui``, ``logic``, ``frame``, ``_favorites``, ``_arena``, ``_group_assignments``, ``_custom_groups`` ...).
"""

from __future__ import annotations

from pathlib import Path

from gui import dialogs
from gui.utils.formatting import fmt_list
from gui.utils.model_groups import available_groups, group_for_model, save_group_state
from simple_sim.io_utils import now_iso


class ModelGroupsMixin:
    """Group / favorite / Arena membership of the checkpoints in the models tree."""

    def _fmt_paths(self, paths: list[Path]) -> str:
        """*paths* relative to the model root, as a truncated list for dialogs."""
        return fmt_list([self.logic.rel_path(p) for p in paths])

    def _is_favorited(self, p: Path) -> bool:
        return self.logic.rel_path(p) in self._favorites

    def _is_arena_tracked(self, p: Path) -> bool:
        return self.logic.rel_path(p) in self._arena

    def _add_selected_to_favorites(self) -> None:
        paths = self._selected_model_paths()
        if not paths: return
        self._assign_selected_to_group(paths, "Favorites")

    def _remove_selected_from_favorites(self) -> None:
        paths = [p for p in self._selected_model_paths() if self._is_favorited(p)]
        if not paths: return
        if not dialogs.confirm(self.frame, "Remove from Favorites", f"Remove {len(paths)} checkpoint(s) from favorites?\n\n{self._fmt_paths(paths)}"): return
        self._assign_selected_to_group(paths, "Uncategorized")

    def _add_selected_to_arena(self) -> None:
        paths = self._selected_model_paths()
        if not paths: return
        self.logic.add_to_arena(paths, self._arena)
        self.logic.save_arena(self._arena)
        self.ui.append_log(f"[arena] added {len(paths)} model(s)\n")
        self._status(f"Added {len(paths)} model(s) to the Arena")
        self._refresh_models()

    def _remove_selected_from_arena(self) -> None:
        paths = [p for p in self._selected_model_paths() if self._is_arena_tracked(p)]
        if not paths: return
        if not dialogs.confirm(self.frame, "Remove from Arena", f"Remove {len(paths)} checkpoint(s) from Arena tracking?\n\n{self._fmt_paths(paths)}"): return
        self.logic.remove_from_arena(paths, self._arena)
        self.logic.save_arena(self._arena)
        self.ui.append_log(f"[arena] removed {len(paths)} model(s)\n")
        self._refresh_models()

    def _available_groups(self) -> list[str]:
        return available_groups(self._custom_groups, self._group_assignments)

    def _group_for_path(self, p: Path) -> str:
        return group_for_model(p, self.logic.rel_path(p), model_root=self.logic._model_root(),
                               assignments=self._group_assignments, groups=self._available_groups(),
                               favorite=self._is_favorited(p))

    def _is_group_header(self, iid: str) -> bool: return bool(iid and iid.startswith("group::"))

    def _group_from_iid(self, iid: str) -> str | None:
        if not iid or "::" not in iid: return None
        return iid.split("::", 1)[1]

    def _refresh_groups_menu(self) -> None:
        if not self.ui._groups_menu: return
        self.ui._groups_menu.delete(0, "end")
        for group in self._available_groups():
            self.ui._groups_menu.add_command(label=group, command=lambda g=group: self._assign_selected_to_group(self._selected_model_paths(), g))

    def _assign_selected_to_group(self, paths: list[Path], group: str | None) -> None:
        if not paths or not group: return
        fav_changed = False
        for p in paths:
            rel = self.logic.rel_path(p)
            if group == "Favorites":
                if rel not in self._favorites:
                    self._favorites[rel] = {"path": rel, "added": now_iso(timespec="seconds")}
                    fav_changed = True
            elif rel in self._favorites:
                self._favorites.pop(rel, None)
                fav_changed = True
            self._group_assignments[rel] = group
        if fav_changed: self.logic.save_favorites(self._favorites)
        self._save_group_state()
        self.ui.append_log(f"[group] assigned {len(paths)} model(s) -> {group}\n")
        self._refresh_models()

    def _save_group_state(self) -> None:
        st = self._store()
        if st is None: return
        save_group_state(st, self._custom_groups, self._group_assignments)
        st.schedule_save(self.frame)

    def _prompt_new_group(self) -> None:
        name = dialogs.ask_string(self.frame, "New Group", "Group name:")
        if name is None: return  # cancelled
        if not (name := name.strip()) or "::" in name:
            dialogs.show_error(self.frame, "Invalid Group Name", "Group name must be non-empty and not contain '::'.")
            return
        if name in self._available_groups():
            dialogs.show_info(self.frame, "Group Exists", f"Group already exists: {name}")
            return
        self._custom_groups.append(name)
        self._save_group_state()
        self._refresh_models()
        self._status(f"Group created: {name}")
