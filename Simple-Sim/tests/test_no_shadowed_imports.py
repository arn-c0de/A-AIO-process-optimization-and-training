"""Guard: a function must not rebind a name the module imports (``report_path = ...`` inside
``run_batch_predict`` shadowed ``reports.report_path`` and crashed the history summary)."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODULES = sorted((ROOT / "simple_sim").rglob("*.py"))


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            names |= {a.asname or a.name for a in node.names}
    return names


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(ROOT)))
def test_functions_do_not_rebind_imported_functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = _imported_names(tree)
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned = {t.id for node in ast.walk(fn) if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                    for t in (node.targets if isinstance(node, ast.Assign) else [node.target]) if isinstance(t, ast.Name)}
        called = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        bad += [f"{fn.name}: {name}" for name in sorted(assigned & imported & called)]
    assert not bad, f"{path.name}: local variable shadows an imported function that is also called: {bad}"
