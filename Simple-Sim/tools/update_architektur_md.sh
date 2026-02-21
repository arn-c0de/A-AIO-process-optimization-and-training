#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_FILE="$ROOT_DIR/architektur.md"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S %Z (%z)')"

collect_paths() {
  (
    cd "$ROOT_DIR"

    # Top-level snapshot (exclude transient env/cache dirs).
    find . -mindepth 1 -maxdepth 1 \
      ! -name ".venv" \
      ! -name ".venv-wheelhouse" \
      ! -name ".pytest_cache" \
      -printf '%P\t%y\n'

    # Detailed subtrees with practical depth limits.
    find configs -maxdepth 2 -name '__pycache__' -prune -o -printf '%p\t%y\n'
    find simple_sim -maxdepth 3 -name '__pycache__' -prune -o -printf '%p\t%y\n'
    find scripts -maxdepth 1 -name '__pycache__' -prune -o -printf '%p\t%y\n'
    find tools -maxdepth 2 -name '__pycache__' -prune -o -printf '%p\t%y\n'
    find gui -maxdepth 3 -name '__pycache__' -prune -o -printf '%p\t%y\n'
    find tests -maxdepth 1 -name '__pycache__' -prune -o -printf '%p\t%y\n'
    find docs -maxdepth 3 -name '__pycache__' -prune -o -printf '%p\t%y\n'
    find images -maxdepth 3 -name '__pycache__' -prune -o -printf '%p\t%y\n'
    find third_party -maxdepth 2 -name '__pycache__' -prune -o -printf '%p\t%y\n'
    # Keep outputs concise: directories only (+ top-level files), avoid huge runtime file lists.
    find outputs -maxdepth 1 -name '__pycache__' -prune -o -printf '%p\t%y\n'
    find outputs -mindepth 2 -maxdepth 2 -type d -name '__pycache__' -prune -o -type d -printf '%p\t%y\n'
  ) | sed 's#^\./##' | sort -u
}

render_structure() {
  collect_paths | awk -F '\t' '
    {
      path = $1
      typ = $2
      n = split(path, parts, "/")
      depth = n - 1
      indent = ""
      for (i = 0; i < depth; i++) {
        indent = indent "  "
      }
      label = parts[n]
      if (typ == "d") {
        label = label "/"
      }
      print indent "- " label
    }
  '
}

{
  echo "# Architecture"
  echo
  echo "Last update: $TIMESTAMP"
  echo
  echo "Note: Auto-generated structure snapshot. Temporary artifacts such as \`__pycache__/\`, \`.venv/\`, \`.venv-wheelhouse/\`, and \`.pytest_cache/\` are hidden."
  echo
  echo "## Project Structure (Auto)"
  echo
  echo '```text'
  echo "Simple-Sim/"
  render_structure
  echo '```'
  echo
  echo "Generated with: \`tools/update_architektur_md.sh\`"
} > "$OUT_FILE"

echo "Updated: $OUT_FILE"
