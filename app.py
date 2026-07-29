from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from autoencoder_lab.training import load_checkpoint
from autoencoder_lab.visualize import save_grid

st.set_page_config(page_title="Latent Canvas", page_icon="✦", layout="wide")
st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 10% 10%, #172554 0, #09090b 45%, #020617 100%); color:#eef2ff}
[data-testid=stFileUploader], [data-testid=stMetric] {background:#ffffff0d;border:1px solid #818cf844;border-radius:18px;padding:1rem}
.hero {padding:2rem;border-radius:24px;background:linear-gradient(120deg,#4f46e555,#db277755);border:1px solid #a5b4fc55;animation:glow 4s ease-in-out infinite alternate}
@keyframes glow {from{box-shadow:0 0 18px #4f46e522}to{box-shadow:0 0 40px #db277744}}
@media (prefers-reduced-motion: reduce){.hero{animation:none!important}}
</style>
<div class="hero"><h1>✦ Latent Canvas</h1><p>Explore a trained MNIST autoencoder's generative space.</p></div>
""", unsafe_allow_html=True)

checkpoint = st.file_uploader("Drop a `.pt` checkpoint", type=["pt", "pth"])
left, right = st.columns([1, 2])
with left:
    count = st.slider("Samples", 8, 64, 32, 8)
    temperature = st.slider("Latent temperature", 0.1, 2.0, 0.8, 0.1)
    st.caption("Train a checkpoint with `python -m autoencoder_lab.cli train`.")
if checkpoint and st.button("Generate", type="primary", use_container_width=True):
    temp_dir = Path(os.getenv("AE_OUTPUT_DIR", "artifacts"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    uploaded = checkpoint.getvalue()
    if len(uploaded) > 25 * 1024 * 1024:
        st.error("Checkpoint uploads are limited to 25 MB.")
    else:
        with tempfile.NamedTemporaryFile(suffix=Path(checkpoint.name).suffix or ".pt", delete=False) as handle:
            handle.write(uploaded)
            checkpoint_path = Path(handle.name)
        try:
            model, payload = load_checkpoint(checkpoint_path)
            with tempfile.NamedTemporaryFile(suffix=".png", dir=temp_dir, delete=False) as image_handle:
                output_path = Path(image_handle.name)
            try:
                output = save_grid(model.sample(count, temperature=temperature), output_path)
                with right:
                    st.image(output.read_bytes(), use_container_width=True)
                    st.success(f"Generated with latent dimension {payload['config']['latent_dim']}")
            finally:
                output_path.unlink(missing_ok=True)
        except Exception as exc:
            st.error(f"Could not load this checkpoint: {exc}")
        finally:
            checkpoint_path.unlink(missing_ok=True)
elif not checkpoint:
    with right:
        st.info("Upload a checkpoint to activate the live latent sampler. The training CLI works on CPU, CUDA, and Apple Silicon.")
