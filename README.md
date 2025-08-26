# Cartoon Image → Short Video (Local I2V)

Minimal Streamlit app that animates a single image into a short MP4 using Diffusers I2VGen-XL. Runs locally on Apple Silicon (MPS) or CPU/GPU.

## Requirements
- Python 3.10+
- macOS with Apple Silicon recommended (M1/M2/M3/M4) and enough RAM/VRAM

## Setup & Run
```bash
# From repo root
./scripts/run_ui.sh
```
Then open the URL shown by Streamlit (usually http://localhost:8501).

### Desktop app (no browser)
```bash
./scripts/run_desktop.sh
```

## Remote CUDA Server (Windows 11 + RTX 3070)

### 1) Install system prerequisites
- Update NVIDIA GeForce driver (Studio/Game Ready)
- Install Git for Windows
- Install Miniconda (64-bit). Use "Anaconda Prompt (Miniconda3)" afterwards.

### 2) Create environment and install CUDA PyTorch
Open "Anaconda Prompt (Miniconda3)":
```bash
conda create -n anima-cuda python=3.11 -y
conda activate anima-cuda

pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 3) Install project requirements
```bash
pip install -r requirements.txt
```

### 4) Start the FastAPI server (CUDA)
```bash
scripts\run_server_windows.ps1  # PowerShell
# or on Unix-like (if server is Linux):
# ./scripts/run_server.sh
```
Server listens on `0.0.0.0:8000`.

### 5) Optional: enable SSH on Windows for tunneling
PowerShell (Run as Administrator):
```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
New-NetFirewallRule -Name sshd -DisplayName "OpenSSH" -Protocol TCP -LocalPort 22 -Action Allow -Direction Inbound
```

Open HTTP port:
```powershell
New-NetFirewallRule -Name AnimaAPI -DisplayName "Anima API 8000" -Protocol TCP -LocalPort 8000 -Action Allow -Direction Inbound
```

### 6) Test server locally
```bash
curl -X GET http://127.0.0.1:8000/health
```

## Mac client (Desktop GUI)

1) Run desktop UI on Mac:
```bash
./scripts/run_desktop.sh
```
2) In the app:
- Check "Use Remote (CUDA)"
- Set server URL: `http://192.168.50.38:8000` (replace with your PC IP)
- Choose image, set prompt and parameters, click Generate

3) SSH tunnel (optional, safer even in LAN):
```bash
ssh -N -L 8000:localhost:8000 <user>@192.168.50.38
# Then set server URL in app to http://localhost:8000
```

### Notes
- For stability, the local runner on Mac may use CPU when MPS is not compatible; remote server uses CUDA (fp16) for speed.
- Recommended generation params for RTX 3070: width/height 512–768 (multiples of 64), frames 12–16, steps 20–30, guidance 6–8, fps 8–12.

## Usage
- Upload a PNG/JPG (cartoon-style images work best).
- Enter a prompt describing the motion.
- Click Generate. MP4 preview and download will appear.

## Example prompts
- "The character looks left and right, subtle head movement"
- "Two characters talk to each other, gentle lip and jaw motion"
- "The hero waves a sword with dynamic motion"
- "The hero sails down a river, camera pans slowly"

## Notes
- Dimensions are rounded to multiples of 64 for stability. Leaving width/height empty preserves aspect ratio with a safe cap.
- Generation is compute-heavy; longer videos or larger sizes will take more time and memory.

## License
For model license/commercial terms, please review the upstream model card in Diffusers/Hugging Face for `ali-vilab/i2vgen-xl`. 
