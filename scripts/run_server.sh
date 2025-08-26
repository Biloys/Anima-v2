#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PROJ_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

cd "$PROJ_DIR"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools | cat
python -m pip install -r requirements.txt | cat

exec uvicorn app.remote_server:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120 | cat


