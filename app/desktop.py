import os
import logging
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import requests
from typing import Optional

from PIL import Image, ImageTk

# Import runner robustly
try:
    from i2v_runner import I2VRunner
except Exception:
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from app.i2v_runner import I2VRunner


class DesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Anima - Image → Video")
        self.geometry("720x640")
        self.resizable(True, True)

        self.runner = I2VRunner()
        self.image: Optional[Image.Image] = None
        self.video_path: Optional[str] = None

        # UI
        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}

        # File chooser
        file_row = tk.Frame(self)
        file_row.pack(fill=tk.X, **pad)
        tk.Button(file_row, text="Choose image…", command=self._choose_image).pack(side=tk.LEFT)
        self.file_label = tk.Label(file_row, text="No file selected", anchor="w")
        self.file_label.pack(side=tk.LEFT, padx=8)

        # Preview canvas
        self.preview = tk.Label(self, text="Preview")
        self.preview.pack(fill=tk.BOTH, expand=True, **pad)

        # Prompt
        tk.Label(self, text="Prompt").pack(anchor="w", **pad)
        self.prompt_entry = tk.Entry(self)
        self.prompt_entry.insert(0, "The hero waves a sword with gentle camera sway")
        self.prompt_entry.pack(fill=tk.X, **pad)

        # Params row
        params = tk.Frame(self)
        params.pack(fill=tk.X, **pad)

        self.frames_var = tk.IntVar(value=16)
        self.fps_var = tk.IntVar(value=8)
        self.steps_var = tk.IntVar(value=30)
        self.guidance_var = tk.DoubleVar(value=7.5)

        self._add_labeled_entry(params, "Frames", self.frames_var)
        self._add_labeled_entry(params, "FPS", self.fps_var)
        self._add_labeled_entry(params, "Steps", self.steps_var)
        self._add_labeled_entry(params, "Guidance", self.guidance_var)

        # Actions
        actions = tk.Frame(self)
        actions.pack(fill=tk.X, **pad)
        tk.Button(actions, text="Generate", command=self._on_generate).pack(side=tk.LEFT)
        tk.Button(actions, text="Open video", command=self._open_video).pack(side=tk.LEFT, padx=8)
        tk.Button(actions, text="Open logs", command=self._open_logs).pack(side=tk.LEFT)

        # Remote server controls
        remote = tk.Frame(self)
        remote.pack(fill=tk.X, **pad)
        tk.Label(remote, text="Remote server URL (optional)").pack(anchor="w")
        self.server_entry = tk.Entry(remote)
        self.server_entry.insert(0, "http://192.168.50.38:8000")
        self.server_entry.pack(fill=tk.X)
        self.use_remote = tk.BooleanVar(value=True)
        tk.Checkbutton(remote, text="Use Remote (CUDA)", variable=self.use_remote).pack(anchor="w")

        # Status
        self.status = tk.Label(self, text="Idle", anchor="w")
        self.status.pack(fill=tk.X, **pad)

    def _add_labeled_entry(self, parent: tk.Widget, label: str, var: tk.Variable) -> None:
        frame = tk.Frame(parent)
        frame.pack(side=tk.LEFT, padx=6)
        tk.Label(frame, text=label).pack(anchor="w")
        tk.Entry(frame, textvariable=var, width=7).pack()

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if not path:
            return
        try:
            self.image = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")
            return
        self.file_label.config(text=os.path.basename(path))
        self._update_preview()
        logging.getLogger("anima").info("Image selected: %s", path)

    def _update_preview(self) -> None:
        if self.image is None:
            self.preview.config(image="", text="Preview")
            return
        img = self.image.copy()
        img.thumbnail((640, 360))
        tk_img = ImageTk.PhotoImage(img)
        self.preview.configure(image=tk_img, text="")
        self.preview.image = tk_img

    def _on_generate(self) -> None:
        if self.image is None:
            messagebox.showwarning("Missing image", "Please choose an image first.")
            return
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            messagebox.showwarning("Missing prompt", "Please enter a prompt.")
            return

        self.status.config(text="Generating…")
        logging.getLogger("anima").info("Start generation | prompt=%s frames=%s fps=%s steps=%s guidance=%.2f",
                                         prompt, self.frames_var.get(), self.fps_var.get(), self.steps_var.get(), self.guidance_var.get())
        threading.Thread(target=self._run_generate, args=(prompt,), daemon=True).start()

    def _run_generate(self, prompt: str) -> None:
        try:
            if self.use_remote.get():
                url = self.server_entry.get().strip()
                files = {"image": ("image.png", self._pil_to_png_bytes(self.image), "image/png")}
                data = {
                    "prompt": prompt,
                    "num_frames": str(int(self.frames_var.get())),
                    "fps": str(int(self.fps_var.get())),
                    "steps": str(int(self.steps_var.get())),
                    "guidance": str(float(self.guidance_var.get())),
                    "width": "0",
                    "height": "0",
                    "seed": "0",
                }
                resp = requests.post(f"{url}/i2v", files=files, data=data, timeout=600)
                resp.raise_for_status()
                # Save MP4 to temp file
                tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                out_path = os.path.join(tmp_dir, "remote_output.mp4")
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                video_path = out_path
            else:
                video_path = self.runner.run(
                    image=self.image,  # type: ignore[arg-type]
                    prompt=prompt,
                    num_frames=int(self.frames_var.get()),
                    guidance_scale=float(self.guidance_var.get()),
                    num_inference_steps=int(self.steps_var.get()),
                    fps=int(self.fps_var.get()),
                )
            self.video_path = video_path
            self.status.config(text=f"Done → {os.path.basename(video_path)}")
            logging.getLogger("anima").info("Generation done: %s", video_path)
        except Exception as e:
            self.status.config(text="Error")
            messagebox.showerror("Generation failed", str(e))
            logging.getLogger("anima").exception("Generation failed: %s", e)

    def _open_video(self) -> None:
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showinfo("Open video", "No video yet. Generate first.")
            return
        path = self.video_path
        if os.name == "posix":
            os.system(f"open '{path}'")
        else:
            os.startfile(path)  # type: ignore[attr-defined]

    def _open_logs(self) -> None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(project_root, "logs", "anima.log")
        if not os.path.exists(log_path):
            messagebox.showinfo("Logs", "Log file not found yet. Try after a generation.")
            return
        if os.name == "posix":
            os.system(f"open '{log_path}'")
        else:
            os.startfile(log_path)  # type: ignore[attr-defined]

    @staticmethod
    def _pil_to_png_bytes(img: Image.Image) -> bytes:
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        return bio.getvalue()


def main() -> None:
    app = DesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()


