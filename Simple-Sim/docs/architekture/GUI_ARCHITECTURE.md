# GUI Architecture

The monitor GUI (`gui/`, started with `./run.sh`) is a Tk/ttk application built from
four layers. Each layer may only use the layers below it.

```
 tabs/<name>/        tab.py (controller) · ui.py (widgets only) · logic.py (tk-free)
 ─────────────────────────────────────────────────────────────────────────────
 widgets/ dialogs.py  reusable building blocks, no knowledge of datasets/models
 ─────────────────────────────────────────────────────────────────────────────
 theme/  help/        design tokens + ttk styles · all help texts
 ─────────────────────────────────────────────────────────────────────────────
 core/   utils/       AppContext, EventBus, TaskRunner · tk-free helpers
```

`gui/monitor.py` is the shell: window, menu (File / View / Help), keyboard shortcuts,
status bar, theme switching, and the `AppContext`.

## Services (`gui/core`)

| Service | Access in a tab | Use it for |
|---|---|---|
| `AppContext` | `self.state.ctx` | entry point to everything below |
| `EventBus` | `ctx.bus.subscribe(Topics.DATASET_CHANGED, fn)` / `ctx.bus.publish(...)` | cross-tab notifications; tabs never call each other |
| `TaskRunner` | `ctx.tasks.run(fn, name=, on_done=, on_error=)` | every background job; callbacks run on the Tk thread; progress + cancel show in the status bar |
| Status bar | `ctx.status("Saved", "success")` | non-blocking feedback (levels: info, success, warning, error) |
| Settings | `self.state.settings_store` | persistence (`bind_persisted_vars` in `gui/utils/settings_store.py`) |

`TaskRunner` replaces `threading.Thread` + `queue` + `after()` polling:

```python
def work(task):
    for i, item in enumerate(items):
        task.check_cancelled()
        process(item)
        task.progress((i + 1) / len(items), item.name)
    return summary

ctx.tasks.run(work, name="Validate dataset", on_done=self._show_summary,
              on_error=lambda e: dialogs.show_error(self.frame, "Validation failed", e))
```

## Design system (`gui/theme`)

- `tokens.py`: `LIGHT` / `DARK` palettes, `SPACE` (4 px grid), font roles, `class_color()`.
- `style.py`: `apply_theme(root, mode)`, `palette(widget)`, `font(role)`, `on_theme_changed(widget, cb)`.
- Never hardcode colours or fonts. Use styles: `Accent.TButton` (primary action),
  `Danger.TButton` (destructive), `Icon.TButton`, `Muted/Small/Heading/Title/Success/Warning/Danger.TLabel`,
  `Card*` variants inside cards, `Badge.TLabel`, `Invalid.TEntry`.
- Custom canvases subscribe with `on_theme_changed` and read `palette(widget)`.
- The profile editor (`tools/profile_editor`) uses the same `apply_theme`.

## Building blocks (`gui/widgets`, `gui/dialogs.py`)

| Need | Use |
|---|---|
| Group of controls with a title (+ "?" help) | `Section(parent, "Title", help_key="tab.section")` |
| Expert options | `CollapsibleSection(..., store=settings, state_key="tab.ui.x_open")` |
| Label/field rows that line up | `FormGrid` → `form.row("Label", widget, help_key=...)` |
| Path entry with Browse | `PathPicker(parent, var, mode="file"/"dir", relative_to=sim_root)` |
| Row of buttons | `Toolbar` / `ButtonBar`, `menu_button` for secondary actions |
| Icon button | `icon_button(parent, "refresh", cmd, help_key=...)` |
| Table | `ScrolledTree(parent, [("id", "Heading", width, anchor), ...])` (sortable, striped) |
| Log | `LogView` (level colours, line limit, context menu) |
| Long settings column | `ScrollableFrame` |
| Empty view | `EmptyState(parent, "No models yet", "…", action="Train", command=...)` |
| Messages | `dialogs.require`, `confirm(danger=True)`, `show_error(parent, title, exc)`, `show_text`, `pick_from_list` |

## Help (`gui/help/texts.py`)

Every control gets `ToolTip(widget, help_key="<tab>.<control>")` (or `help_key=` on the
widget helpers). Texts live in `HELP`; section texts for the "?" button in `SECTION_HELP`;
the Help menu uses `QUICK_START` and `SHORTCUTS`. Hover hints also appear in the status bar.
`tests/test_gui_core.py` fails when a `help_key` literal has no text.

## Add a tab in 3 steps

1. Create `gui/tabs/<name>/` with `tab.py` (subclass `gui.tabs.core.base.BaseTab`, implement
   `build_ui()`), `ui.py` (widgets only, built from `gui/widgets`) and `logic.py` (no tkinter).
2. Register it in `gui/tabs/core/registry.py`: `TabSpec("<name>", "Title", NameTab)`.
3. Add help texts under `# ---- <name>` in `gui/help/texts.py`.

`tests/test_gui_smoke.py` builds every registered tab, so a broken tab fails CI.

## Keyboard shortcuts

F1 help · F5 start pipeline · Shift+F5 stop · Ctrl+R refresh tab · Ctrl+1…9 switch page ·
Ctrl+Shift+D dark mode · Ctrl+Q quit · Esc closes dialogs.

## Checking changes visually

```bash
python -m simple_sim gui-screenshots --out /tmp/gui_shots --demo   # light + dark PNG per tab
python -m pytest -q tests/test_gui_smoke.py                        # needs a display or Xvfb
```

UI scale: View → UI Scale, or `SIMPLE_SIM_UI_SCALE=1.5 ./run.sh`.
