# GUI Modernization Plan

Goal: a modern, modular, scalable and user-friendly Tk monitor GUI (`gui/`) with one
shared design system (also used by `tools/profile_editor/`).

Status tracking: each phase lists its exit criteria; tick them off in the PR that lands it.

## Status (2026-11-06)

| Phase | State |
|---|---|
| 0 Safety net | done: `tests/test_gui_smoke.py` (headless Xvfb, every tab, tooltip on every button, theme toggle), `tests/test_gui_core.py`, `python -m simple_sim gui-screenshots` |
| 1 Foundation | done: `gui/theme`, `gui/core` (EventBus, TaskRunner, AppContext), dark mode, UI scale, profile editor on shared theme |
| 2 Widgets | done: `gui/widgets`, `gui/dialogs.py`, theme-aware tooltips, `gui/help/texts.py` |
| 3 Tabs | done for pipeline, datasets, validation, board detection, analysis, predictions, weights, merge. **Open:** `profile_preview` (new tab); move remaining threads in `weights/logic.py`, `board_detection/logic.py` to TaskRunner; split `pipeline/tab.py` (run-spec logic, remaining Toplevels for start confirmation and sample/target selectors) |
| 4 Navigation | done: grouped sidebar (Workflow / Data / Evaluate / Models / Inference), page header. **Open:** single prediction path (Weights "Evaluate" vs Predictions), shared sample browser Analysis/Predictions |
| 5 Help & polish | done: tooltips + section "?" on all migrated tabs, Help menu, quick start on first launch, shortcuts, status bar hints, danger confirmations |
| 6 Docs | done: `docs/architekture/GUI_ARCHITECTURE.md`. **Open:** visual review with screenshots (light/dark, 1280x720, scale 1.5) |

## Findings (baseline, 2026-11)

- **No theme:** no `ttk.Style` in `gui/`, no dark mode, 19 inline `font=(...)` tuples,
  hardcoded colors (`gray20`, `orange`, `#ffffe0`), 9 different `padx` variants. Only the
  profile editor has a palette (`tools/profile_editor/app.py:_apply_theme`).
- **Layout:** the Pipeline tab is 6 loose `pack` rows with ~110 widgets and no grouping;
  labels do not align; Config/Out have no Browse button. `pipeline/tab.py` is 2125 lines
  (7 dialogs + run-spec logic).
- **Duplication:** Treeview+Scrollbar ~12x, thread + `after()` polling ~10x, dataset
  combo + refresh 4x, list-picker dialog 4x, text-viewer dialog 4x, image canvas 4x,
  "Select X first" / "X failed: {e}" message boxes in every tab.
- **Functional overlap:** Predictions vs Weights "Evaluate" (two prediction paths),
  Analysis vs Predictions (two sample browsers), Weights vs Merge (two checkpoint trees).
- **Help:** 12 tooltips, half of which only echo the current value.
- **Tests:** no test creates a widget.

## Decision

Stay on Tk/ttk; modernize the architecture and the visual layer. Logic becomes tk-free,
which keeps a future framework switch possible.

## Target architecture

```
gui/
├── monitor.py             # shell: window, navigation, status bar, menu, shortcuts
├── core/
│   ├── context.py         # AppContext: services handed to every tab
│   ├── events.py          # EventBus (dataset.changed, model.changed, task.*)
│   └── tasks.py           # TaskRunner: background jobs -> UI thread, progress, cancel
├── theme/
│   ├── tokens.py          # colors (light/dark), spacing, fonts
│   └── style.py           # apply_theme(root, mode) - shared with the profile editor
├── widgets/               # reusable presentational building blocks
├── dialogs/               # list picker, text viewer, confirm, error/info
├── help/texts.py          # all help texts, keyed (i18n-ready)
└── tabs/<name>/{tab.py,ui.py,logic.py}
```

Rules:

1. `logic.py` never imports tkinter.
2. Tabs talk via the EventBus / AppContext services, never to each other directly.
3. Background work goes through the TaskRunner.
4. Colors, spacing and fonts come only from `gui/theme/tokens.py`.
5. Every control's explanation lives in `gui/help/texts.py`.

## Design system

| Token | Light | Dark | Use |
|---|---|---|---|
| `bg` | `#f5f6f8` | `#16191d` | window |
| `surface` | `#ffffff` | `#1f2329` | cards / sections |
| `surface_alt` | `#eef0f3` | `#262b32` | toolbars, table headings |
| `border` | `#d9dde3` | `#323840` | subtle separators |
| `text` / `text_muted` | `#1b1f24` / `#6b7280` | `#e8ebef` / `#9aa3ad` | |
| `accent` | `#2f6fed` | `#4c8dff` | primary action, selection, focus |
| `success` / `warning` / `danger` | green / amber / red | lighter variants | status, badges, delete |

Fonts (compact): `Title` 11 bold, `Heading` 9 bold, `Body` 9, `Small` 8, `Mono` 9.
Spacing (compact): 1, 3, 5, 8, 10, 16 px. Buttons are plain; `Accent`/`Danger` only tint text and border (no filled colour blocks).

## UX principles

1. One clear primary action per view (`Accent.TButton`).
2. Progressive disclosure: expert options in collapsible "Advanced" sections (state persisted).
3. Guided flow: Data -> Validate -> Train -> Evaluate.
4. Immediate feedback: inline validation, disabled buttons explain why in their tooltip.
5. No blocking pop-ups for routine success: toast / status bar instead; message boxes only
   for errors and destructive confirmations.
6. Safe deletes: confirmation names the count and the target.
7. Uniform progress: every task shows up in the status bar.
8. Empty states with an action.
9. Readable errors with expandable details.
10. The GUI remembers window size, panes, filters, selection, open sections.

## Phases

### Phase 0 - Safety net
- `tests/test_gui_smoke.py`: build every tab under Xvfb (skip without display/tk), activate,
  close.
- `tools/gui_screenshots.py`: capture every tab (light/dark) for before/after reviews.
- Exit: smoke test green, baseline screenshots captured.

### Phase 1 - Foundation
- `gui/theme/` (tokens, `apply_theme`, named fonts, HiDPI scaling, dark mode toggle).
- Profile editor uses the shared palette.
- `gui/core/` EventBus + TaskRunner; dataset change events go through the bus.

### Phase 2 - Widget library
- `gui/widgets/`: `Section`, `CollapsibleSection`, `FormGrid`, `PathPicker`,
  `ScrolledTree`, `LogView`, `ButtonBar`, `StatusBar`, `Toast`, `HelpTooltip`,
  `EmptyState`.
- `gui/dialogs/`: `ListPickerDialog`, `TextViewerDialog`, `confirm`, `show_error`.
- Theme-aware `ToolTip`.

### Phase 3 - Migrate tabs (one at a time)
Order: Board Detection, Validation, Datasets, Merge, Predictions, Analysis, Weights,
Pipeline. Pipeline layout: left column with cards Source / Output / Run / Model /
Multi-Profile (expert parts collapsible), action bar on top, logs + image grid on the right.
Target: no file above ~600 lines.

### Phase 4 - Navigation & functional dedup
Grouped navigation: Data (Datasets, Validation), Generate & Train (Pipeline),
Evaluate (Analysis, Predictions), Models (Weights, Merge), Inference (Board Detection).
One prediction path, shared `MetricsPanel`, shared sample browser.

### Phase 5 - Help & polish
Tooltips on every control, `?` per section, Help menu (quick start, shortcuts),
shortcuts (F5 start, Esc stop, Ctrl+R refresh, Ctrl+1..8 tabs, F1 help), consistent
Title Case labels, Danger style for destructive actions.

### Phase 6 - Future-proofing
`docs/architekture/GUI_ARCHITECTURE.md` ("add a tab in 3 steps"), unit tests for every
tk-free logic module, smoke tests for every view.

## Per-tab review checklist

- [ ] Labels aligned (form grid), no hard pixel widths for layout.
- [ ] Only tokens: no hardcoded color or font.
- [ ] Checked in light and dark (screenshots in the PR).
- [ ] Works at 1280x720 and with HiDPI scaling 1.5.
- [ ] Every button has a tooltip; icon buttons always.
- [ ] Logical focus order, visible focus, Enter/Esc work.
- [ ] Empty, loading and error states are designed.
- [ ] Consistent labels: Title Case, "..." only for actions that open a dialog.
