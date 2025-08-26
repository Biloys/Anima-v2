import io
import os
from typing import Optional

import streamlit as st
from PIL import Image

# Robust import for Streamlit execution (script dir on sys.path)
try:
    from i2v_runner import I2VRunner  # when working directory is app/
except Exception:
    import os
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from app.i2v_runner import I2VRunner  # when importing as package


st.set_page_config(page_title="Cartoon I2V", page_icon="🎬", layout="centered")
st.title("Cartoon Image → Short Video")
st.caption("Minimal UI for animating a single image with a text prompt. No audio.")


@st.cache_resource(show_spinner=False)
def get_runner() -> I2VRunner:
    return I2VRunner()


def main() -> None:
    with st.sidebar:
        st.markdown("**Generation settings**")
        num_frames = st.slider("Frames", min_value=8, max_value=48, value=16, step=2)
        fps = st.slider("FPS", min_value=4, max_value=16, value=8, step=1)
        guidance = st.slider("Guidance scale", min_value=1.0, max_value=12.0, value=7.5, step=0.5)
        steps = st.slider("Steps", min_value=10, max_value=50, value=30, step=2)
        width = st.number_input("Width (px, optional)", min_value=0, max_value=2048, value=0, step=64)
        height = st.number_input("Height (px, optional)", min_value=0, max_value=2048, value=0, step=64)
        seed = st.number_input("Seed (optional)", min_value=0, max_value=2**31 - 1, value=0, step=1)

    uploaded = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    prompt = st.text_input("Prompt", placeholder="e.g., The hero waves a sword with gentle camera sway")

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input image", use_column_width=True)
    else:
        image = None

    col_run, col_dl = st.columns(2)
    run_clicked = col_run.button("Generate", type="primary", use_container_width=True, disabled=uploaded is None or not prompt)
    download_container = col_dl.container()

    if run_clicked and image is not None and prompt:
        st.toast("Starting generation…")
        runner = get_runner()
        video_path = runner.run(
            image=image,
            prompt=prompt,
            num_frames=int(num_frames),
            width=(int(width) if width else None),
            height=(int(height) if height else None),
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            seed=(int(seed) if seed else None),
            fps=int(fps),
        )
        st.success("Done")
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        download_container.download_button(
            label="Download MP4",
            data=video_bytes,
            file_name="output.mp4",
            mime="video/mp4",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()


