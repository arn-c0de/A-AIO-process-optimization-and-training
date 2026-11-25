#!/usr/bin/env python3
"""Block words that must never reach GitHub (company, customer, product names) in commits and pushes.

The word list itself stays local: one word / regex per line in ``<repo>/.forbidden-words`` (git-ignored) or the file
named by ``$FORBIDDEN_WORDS_FILE``; ``#`` starts a comment. Words match case-insensitively as whole words.

  python tools/forbidden_words.py --install-hooks      # pre-commit, commit-msg and pre-push hooks of this clone
  python tools/forbidden_words.py --tree               # all tracked files (text) and file names
  python tools/forbidden_words.py --staged             # what the next commit adds (pre-commit hook)
  python tools/forbidden_words.py --message FILE       # a commit message (commit-msg hook)
  python tools/forbidden_words.py --range A..B         # contents, file names and messages of commits (pre-push hook)

Exit code 1 lists every hit (file:line, never the full line); 2 when the word list is missing.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

LIST_ENV = "FORBIDDEN_WORDS_FILE"
LIST_NAME = ".forbidden-words"
ZERO = "0" * 40


def git(*args: str, root: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True, errors="replace").stdout


def repo_root() -> Path:
    return Path(git("rev-parse", "--show-toplevel").strip())


def load_pattern(root: Path) -> re.Pattern[str] | None:
    path = Path(os.environ.get(LIST_ENV) or root / LIST_NAME).expanduser()
    if not path.is_file():
        return None
    words = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not words:
        return None
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(f"(?:{w})" for w in words) + r")(?![A-Za-z0-9])", re.IGNORECASE)


def scan_lines(pat: re.Pattern[str], label: str, text: str) -> list[str]:
    return [f"{label}:{i}" for i, line in enumerate(text.splitlines(), 1) if pat.search(line)]


def scan_diff(pat: re.Pattern[str], diff: str) -> list[str]:
    """Added lines (``+``) and file names of a unified diff (``git diff`` / ``git log -p``, binaries skipped)."""
    hits, cur, commit, n = [], "?", "", 0
    for line in diff.splitlines():
        if line.startswith("commit "):
            commit = line.split()[1][:9] + " "
        elif line.startswith("diff --git "):
            names = line[len("diff --git "):]
            if pat.search(names):
                hits.append(f"{commit}file name: {names}")
        elif line.startswith("+++ "):
            cur = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            n = int(m[1]) - 1 if m else 0
        elif line.startswith("+"):
            n += 1
            if pat.search(line[1:]):
                hits.append(f"{commit}{cur}:{n}")
        elif not line.startswith("-"):
            n += 1
    return hits


def check_tree(pat: re.Pattern[str], root: Path) -> list[str]:
    hits = []
    for name in git("ls-files", "-z", root=root).split("\0"):
        if not name:
            continue
        if pat.search(name):
            hits.append(f"file name: {name}")
        p = root / name
        if p.is_symlink() or not p.is_file():
            continue
        data = p.read_bytes()
        if b"\0" in data[:8192]:
            continue                                  # binary
        hits += scan_lines(pat, name, data.decode("utf-8", errors="replace"))
    return hits


def check_range(pat: re.Pattern[str], root: Path, rev_range: str) -> list[str]:
    hits = scan_diff(pat, git("log", "-p", "--no-color", "--no-ext-diff", "--format=commit %H", rev_range, root=root))
    for line in git("log", "--format=%H%x00%B%x01", rev_range, root=root).split("\x01"):
        if "\0" in line:
            sha, msg = line.strip("\n").split("\0", 1)
            hits += [f"{sha[:9]} message:{h.split(':')[-1]}" for h in scan_lines(pat, "msg", msg)]
            for who in git("show", "-s", "--format=%an%n%ae%n%cn%n%ce", sha, root=root).splitlines():
                if pat.search(who):
                    hits.append(f"{sha[:9]} author / committer")
                    break
    return hits


def pre_push_ranges(stdin: str, root: Path) -> list[str]:
    """``A..B`` per pushed ref (new branch: commits not on any remote branch yet)."""
    out = []
    for line in stdin.splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[1] == ZERO:
            continue                                   # deleting a ref
        local, remote = parts[1], parts[3]
        out.append(f"{remote}..{local}" if remote != ZERO else f"{local} --not --remotes")
    return out


HOOKS = {
    "pre-commit": 'exec python3 "{script}" --staged\n',
    "commit-msg": 'exec python3 "{script}" --message "$1"\n',
    "pre-push": 'exec python3 "{script}" --pre-push\n',
}


def install_hooks(root: Path) -> None:
    hooks = Path(git("rev-parse", "--git-path", "hooks", root=root).strip())
    hooks = hooks if hooks.is_absolute() else root / hooks
    hooks.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    for name, body in HOOKS.items():
        p = hooks / name
        if p.exists() and "forbidden_words.py" not in p.read_text(errors="replace"):
            print(f"{p} exists and is not ours: left alone", file=sys.stderr)
            continue
        p.write_text("#!/bin/sh\n# installed by tools/forbidden_words.py --install-hooks\n" + body.format(script=script))
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"installed {p}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--install-hooks", action="store_true")
    g.add_argument("--tree", action="store_true")
    g.add_argument("--staged", action="store_true")
    g.add_argument("--message", type=Path)
    g.add_argument("--range")
    g.add_argument("--pre-push", action="store_true")
    a = ap.parse_args(argv)
    root = repo_root()
    if a.install_hooks:
        install_hooks(root)
        if load_pattern(root) is None:
            print(f"note: no word list yet - create {root / LIST_NAME} (one word per line)", file=sys.stderr)
        return 0
    pat = load_pattern(root)
    if pat is None:
        print(f"forbidden words: no list at {os.environ.get(LIST_ENV) or root / LIST_NAME} - refusing (create it)", file=sys.stderr)
        return 2
    if a.tree:
        hits = check_tree(pat, root)
    elif a.staged:
        hits = scan_diff(pat, git("diff", "--cached", "--no-color", "--no-ext-diff", root=root))
    elif a.message:
        text = "\n".join(ln for ln in a.message.read_text(errors="replace").splitlines() if not ln.startswith("#"))
        hits = scan_lines(pat, "commit message", text)
    elif a.range:
        hits = check_range(pat, root, a.range)
    else:
        hits = []
        for r in pre_push_ranges(sys.stdin.read(), root):
            hits += check_range(pat, root, r) if " " not in r else check_range_not_remotes(pat, root, r.split()[0])
    if hits:
        print("forbidden words found (word list: .forbidden-words) - nothing may contain them:", file=sys.stderr)
        for h in hits:
            print(f"  {h}", file=sys.stderr)
        return 1
    return 0


def check_range_not_remotes(pat: re.Pattern[str], root: Path, local: str) -> list[str]:
    shas = git("rev-list", local, "--not", "--remotes", root=root).split()
    return [h for s in shas for h in check_range(pat, root, f"{s}^!")]


if __name__ == "__main__":
    sys.exit(main())
