import os
import math
import tempfile
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import imageio

# External dependency; ensure diffusers is installed via requirements.txt
from diffusers import I2VGenXLPipeline


def _setup_logger() -> logging.Logger:
    """Create a rotating file logger under project logs directory."""
    logger = logging.getLogger("anima")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    # Resolve project root from this file location
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "anima.log")

    handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Also emit to stdout during development
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    logger.propagate = False
    logger.info("Logger initialized → %s", log_path)
    return logger


LOGGER = _setup_logger()


def _round_to_multiple_of_64(value: int) -> int:
    """Round dimension to nearest multiple of 64, minimum 256 for stability."""
    if value < 256:
        return 256
    return int(round(value / 64.0) * 64)


class I2VRunner:
    """Thin wrapper around Diffusers I2VGen-XL pipeline for image-to-video on Apple Silicon.

    The runner lazily loads the model on first use to reduce startup time.
    """

    def __init__(
        self,
        model_id: str = "ali-vilab/i2vgen-xl",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.float16,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        # I2V uses 3D UNet ops; MPS does not support Conv3D. Force CPU for reliability.
        if self.device == "mps":
            LOGGER.info("MPS detected but Conv3D is unsupported → forcing CPU for I2V")
            self.device = "cpu"
        self.dtype = dtype if self.device in ("cuda",) else torch.float32
        self._pipe: Optional[I2VGenXLPipeline] = None
        LOGGER.info("I2VRunner init | model_id=%s device=%s dtype=%s", self.model_id, self.device, self.dtype)

    def _load_pipe(self) -> I2VGenXLPipeline:
        if self._pipe is not None:
            return self._pipe

        LOGGER.info("Loading pipeline '%s'…", self.model_id)
        pipe = I2VGenXLPipeline.from_pretrained(self.model_id, torch_dtype=self.dtype)

        # Disable safety checker if present to avoid sporadic slowdowns on MPS
        if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            pipe.safety_checker = None

        pipe.to(self.device)
        LOGGER.info("Pipeline loaded → device=%s", self.device)

        # Small memory/perf tweaks
        try:
            pipe.enable_attention_slicing()
        except Exception:
            LOGGER.debug("enable_attention_slicing not supported")
        try:
            pipe.vae.enable_slicing()
        except Exception:
            LOGGER.debug("vae.enable_slicing not supported")

        self._pipe = pipe
        LOGGER.info("Pipeline ready")
        return pipe

    def _resolve_dims(self, image: Image.Image, width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
        if width and height:
            return _round_to_multiple_of_64(width), _round_to_multiple_of_64(height)

        # Preserve aspect ratio based on input image; cap longest side to 768 for memory
        w, h = image.size
        max_side = max(w, h)
        scale = 768.0 / max_side if max_side > 768 else 1.0
        new_w = _round_to_multiple_of_64(int(w * scale))
        new_h = _round_to_multiple_of_64(int(h * scale))
        return new_w, new_h

    def _letterbox_square(self, image: Image.Image, side: int) -> Image.Image:
        """Pad image to a square canvas of size `side` with white background.
        Keeps aspect ratio; resizes the longer side to `side` and centers.
        """
        side = _round_to_multiple_of_64(side)
        src = image.convert("RGB")
        ratio = min(side / src.width, side / src.height)
        new_w = max(1, int(round(src.width * ratio)))
        new_h = max(1, int(round(src.height * ratio)))
        # Ensure even dims to keep encoders/FFmpeg happy
        new_w -= new_w % 2
        new_h -= new_h % 2
        resized = src.resize((new_w, new_h), Image.BICUBIC)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        offset = ((side - new_w) // 2, (side - new_h) // 2)
        canvas.paste(resized, offset)
        return canvas

    def generate_frames(
        self,
        image: Image.Image,
        prompt: str,
        *,
        num_frames: int = 16,
        width: Optional[int] = None,
        height: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Run I2V and return a list of PIL frames."""
        pipe = self._load_pipe()
        tgt_w, tgt_h = self._resolve_dims(image, width, height)
        # Workaround for PyTorch MPS adaptive pool constraint: enforce square dims
        # This avoids non-divisible pooling shapes in some UNet branches on MPS.
        processed_image = image
        if self.device == "mps":
            # Use a conservative, architecture-friendly size on MPS
            side = 512
            if width and height:
                side = max(256, min(768, _round_to_multiple_of_64(min(width, height))))
                # snap to 256-multiples to reduce pooling edge cases
                side = max(256, min(768, (side // 256) * 256 or 256))
            LOGGER.info("MPS workaround: enforce square letterbox to %dx%d", side, side)
            tgt_w = tgt_h = side
            processed_image = self._letterbox_square(image, side)
        LOGGER.info(
            "Generate frames | size_in=%sx%s size_out=%sx%s frames=%s steps=%s guidance=%.2f seed=%s",
            image.size[0], image.size[1], tgt_w, tgt_h, num_frames, num_inference_steps, guidance_scale, seed,
        )

        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
            except Exception:
                generator = torch.Generator().manual_seed(int(seed))

        try:
            result = pipe(
                image=processed_image.convert("RGB"),
                prompt=prompt,
                num_frames=int(num_frames),
                height=int(tgt_h),
                width=int(tgt_w),
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(num_inference_steps),
                generator=generator,
            )
        except RuntimeError as e:
            # Automatic CPU fallback for MPS adaptive pooling constraints
            if self.device == "mps" and "Adaptive pool MPS" in str(e):
                LOGGER.warning("MPS adaptive pool error. Falling back to CPU for this run…")
                pipe.to("cpu")
                self.device = "cpu"
                generator = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None
                result = pipe(
                    image=processed_image.convert("RGB"),
                    prompt=prompt,
                    num_frames=int(num_frames),
                    height=int(tgt_h),
                    width=int(tgt_w),
                    guidance_scale=float(guidance_scale),
                    num_inference_steps=int(num_inference_steps),
                    generator=generator,
                )
            else:
                LOGGER.exception("Pipeline generation failed: %s", e)
                raise

        frames: List[Image.Image] = result.frames  # type: ignore[attr-defined]
        LOGGER.info("Frames generated: %d", len(frames))
        return frames

    def render_mp4(
        self,
        frames: List[Image.Image],
        *,
        fps: int = 8,
        out_path: Optional[str] = None,
        quality: int = 8,
    ) -> str:
        """Write frames to an MP4 file and return path."""
        if out_path is None:
            tmp_dir = tempfile.mkdtemp(prefix="i2v_")
            out_path = os.path.join(tmp_dir, "output.mp4")

        # H.264 requires even width/height; enforce by cropping a single pixel if needed
        safe_frames: List[Image.Image] = []
        for frame in frames:
            w, h = frame.size
            safe_w = w - (w % 2)
            safe_h = h - (h % 2)
            if safe_w != w or safe_h != h:
                LOGGER.debug("Crop to even size: %sx%s → %sx%s", w, h, safe_w, safe_h)
                frame = frame.crop((0, 0, safe_w, safe_h))
            safe_frames.append(frame)

        writer = imageio.get_writer(
            out_path,
            fps=int(fps),
            codec="libx264",
            quality=int(quality),
            format="FFMPEG",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        LOGGER.info("Encoding MP4 | frames=%d fps=%d out=%s", len(safe_frames), fps, out_path)
        try:
            for frame in safe_frames:
                writer.append_data(np.array(frame.convert("RGB")))
        finally:
            writer.close()
        LOGGER.info("MP4 written: %s", out_path)
        return out_path

    def run(
        self,
        image: Image.Image,
        prompt: str,
        *,
        num_frames: int = 16,
        width: Optional[int] = None,
        height: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        fps: int = 8,
    ) -> str:
        """Convenience method: generate frames then encode to MP4 and return file path."""
        frames = self.generate_frames(
            image=image,
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        return self.render_mp4(frames, fps=fps)


