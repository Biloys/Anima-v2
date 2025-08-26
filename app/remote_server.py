import io
import os
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from PIL import Image

from .i2v_runner import I2VRunner


app = FastAPI(title="Anima CUDA I2V Server", version="0.1.0")


def get_runner_cuda() -> I2VRunner:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = I2VRunner(device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    return runner


RUNNER: Optional[I2VRunner] = None


@app.on_event("startup")
def _startup() -> None:
    global RUNNER
    RUNNER = get_runner_cuda()


@app.get("/health")
def health() -> JSONResponse:
    device = RUNNER.device if RUNNER else "uninitialized"
    return JSONResponse({"ok": True, "device": device})


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


