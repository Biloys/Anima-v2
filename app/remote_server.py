import io
import os
import logging
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from PIL import Image

from .i2v_runner import I2VRunner


app = FastAPI(title="Anima CUDA I2V Server", version="0.1.0")
LOGGER = logging.getLogger("anima")


def get_runner_cuda() -> I2VRunner:
    forced = os.getenv("ANIMA_DEVICE", "").strip().lower() or None
    cuda_available = torch.cuda.is_available()
    if forced == "cuda" and not cuda_available:
        LOGGER.warning("ANIMA_DEVICE=cuda requested but CUDA is not available; falling back to CPU")

    if forced in ("cuda", "cpu"):
        device = forced if (forced != "cuda" or cuda_available) else "cpu"
    else:
        device = "cuda" if cuda_available else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32
    LOGGER.info("Server runner device resolved: %s | cuda_available=%s", device, cuda_available)
    runner = I2VRunner(device=device, dtype=dtype)
    return runner


RUNNER: Optional[I2VRunner] = None


@app.on_event("startup")
def _startup() -> None:
    global RUNNER
    RUNNER = get_runner_cuda()


@app.get("/health")
def health() -> JSONResponse:
    device = RUNNER.device if RUNNER else "uninitialized"
    cuda_available = torch.cuda.is_available()
    cuda_device_name: Optional[str] = None
    cuda_device_count: int = 0
    cuda_capability: Optional[str] = None
    if cuda_available:
        try:
            cuda_device_name = torch.cuda.get_device_name(0)
            cuda_device_count = torch.cuda.device_count()
            major, minor = torch.cuda.get_device_capability(0)
            cuda_capability = f"{major}.{minor}"
        except Exception:
            pass
    return JSONResponse({
        "ok": True,
        "device": device,
        "cuda_available": bool(cuda_available),
        "cuda_device_name": cuda_device_name,
        "cuda_device_count": int(cuda_device_count),
        "cuda_capability": cuda_capability,
    })


@app.post("/i2v")
def i2v(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    num_frames: int = Form(16),
    fps: int = Form(8),
    steps: int = Form(30),
    guidance: float = Form(7.5),
    width: int = Form(0),
    height: int = Form(0),
    seed: int = Form(0),
) -> StreamingResponse:
    if RUNNER is None:
        raise HTTPException(status_code=503, detail="Runner not ready")

    data = image.file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    mp4_path = RUNNER.run(
        image=img,
        prompt=prompt,
        num_frames=int(num_frames),
        width=(int(width) if width else None),
        height=(int(height) if height else None),
        guidance_scale=float(guidance),
        num_inference_steps=int(steps),
        seed=(int(seed) if seed else None),
        fps=int(fps),
    )

    def _iterfile() -> bytes:
        with open(mp4_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(_iterfile(), media_type="video/mp4")


