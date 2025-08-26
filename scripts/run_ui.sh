#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PROJ_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

export PYTHONUNBUFFERED=1

cd "$PROJ_DIR"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools | cat
python -m pip install -r requirements.txt | cat

exec python -m streamlit run app/ui.py --server.headless true --server.fileWatcherType none | cat


