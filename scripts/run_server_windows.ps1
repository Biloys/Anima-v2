param(
  [string]$CondaEnv = "anima-cuda"
)

$ErrorActionPreference = "Stop"

Write-Host "Activating conda environment: $CondaEnv"
conda activate $CondaEnv

Write-Host "Installing project requirements..."
pip install --upgrade pip wheel setuptools | Out-Host
pip install -r requirements.txt | Out-Host

Write-Host "Starting FastAPI server on 0.0.0.0:8000"
uvicorn app.remote_server:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120


