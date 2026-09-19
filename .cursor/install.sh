#!/usr/bin/env bash
# Idempotent bootstrap for the AI Engineering Cohort assignments.
# Creates a repo-local virtualenv and installs each assignment's dependencies.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="$REPO_ROOT/.venv"

# The default image ships Python 3 but not the venv module; install it once.
# Guarded so re-runs (and snapshot boots where it is already present) are no-ops.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq "python3-venv" "python3-pip"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip

# Install dependencies for every assignment that declares them.
shopt -s nullglob
for req in "$REPO_ROOT"/Assignment*/requirements.txt; do
  echo "Installing dependencies from $req"
  "$VENV_DIR/bin/pip" install -r "$req"
done

echo "Environment bootstrap complete."
