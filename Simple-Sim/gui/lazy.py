"""Lazy package exports (PEP 562), so tk-free submodules import without tkinter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any


def lazy_exports(package: str, exports: Mapping[str, str], namespace: dict[str, Any]) -> Callable[[str], Any]:
    """Module ``__getattr__`` resolving ``exports`` (``name -> relative module``) on first access.

    The value is cached in ``namespace`` (the package's ``globals()``), so ``__getattr__`` runs once per name.
    """

    def __getattr__(name: str) -> Any:
        module_name = exports.get(name)
        if module_name is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(import_module(module_name, package), name)
        namespace[name] = value
        return value

    return __getattr__
