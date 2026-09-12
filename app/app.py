"""Streamlit interface for research-only FLAIR MRI segmentation."""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nifti import load_nifti, save_nifti  # noqa: E402
from src.inference.predict import load_model_checkpoint, predict_volume  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.visualization.plotting import normalize_for_display, overlay_mask  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "config.yaml"
DISCLAIMER = (
    "This application is intended for educational and research purposes. "
    "The model has not been clinically validated and must not be used for "
    "patient diagnosis or treatment decisions."
)


@st.cache_resource(show_spinner=False)
def cached_model(checkpoint_path: str, checkpoint_mtime: float):
    """Cache a model while invalidating it when the checkpoint changes."""
    del checkpoint_mtime
    config = load_config(CONFIG_PATH)
    device = get_device()
    model, checkpoint = load_model_checkpoint(config, checkpoint_path, device)
    return model, checkpoint, device


def _suffix(filename: str) -> str:
    return ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"


def _mask_download(mask: np.ndarray, source, filename: str) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / filename
        save_nifti(mask, source.affine, source.header, path)
        return path.read_bytes()


def main() -> None:
    st.set_page_config(page_title="Brain Tumor MRI Analysis", page_icon="🧠", layout="wide")
    st.title("Brain Tumor MRI Analysis")
    st.caption("AI-assisted MRI tumor segmentation using U-Net")
    st.warning(f"Research Use Only — {DISCLAIMER}")
    config = load_config(CONFIG_PATH)
    checkpoint_path = ROOT / config["paths"]["best_checkpoint"]

    with st.sidebar:
        st.subheader("Model information")
        st.write("Model: 2D U-Net")
        st.write("Input modality: FLAIR")
        st.write("Task: Binary whole-tumor segmentation")
        st.write(f"Threshold: {config['inference']['threshold']:.2f}")
        st.write(f"Checkpoint: {checkpoint_path}")
        if not checkpoint_path.is_file():
            st.info("Train the model or place a valid checkpoint at the path above before analysis.")

    uploaded = st.file_uploader("Upload FLAIR MRI (.nii or .nii.gz)", type=["nii", "gz"])
    if uploaded is None:
        return
    if not (uploaded.name.lower().endswith(".nii") or uploaded.name.lower().endswith(".nii.gz")):
        st.error("Unsupported file. Upload a .nii or .nii.gz NIfTI volume.")
        return
    if not checkpoint_path.is_file():
        st.error("The model checkpoint is missing. Complete training before running an analysis.")
        return

    try:
        with tempfile.TemporaryDirectory() as directory:
            upload_path = Path(directory) / f"uploaded{_suffix(uploaded.name)}"
            upload_path.write_bytes(uploaded.getvalue())
            source = load_nifti(upload_path)
        model, checkpoint, device = cached_model(str(checkpoint_path), checkpoint_path.stat().st_mtime)
        with st.spinner("Analyzing the MRI volume..."):
            result = predict_volume(model, source.array[None], source.spacing, config, device)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        st.error(f"The MRI could not be analyzed: {exc}")
        return

    st.subheader("AI Analysis Result")
    status = "Detected" if result["detected"] else "Not detected"
    metric_columns = st.columns(4)
    metric_columns[0].metric("Suspicious tumor-like region", status)
    metric_columns[1].metric("Predicted tumor voxels", f"{result['tumor_voxels']:,}")
    metric_columns[2].metric("Estimated tumor volume", f"{result['volume_cm3']:.3f} cm³")
    probability = result["mean_lesion_probability"]
    metric_columns[3].metric("Mean lesion probability", "N/A" if probability is None else f"{probability:.3f}")

    default_slice = int(result["important_slice"])
    slice_index = st.slider("Slice index", 0, source.array.shape[2] - 1, default_slice)
    original = source.array[:, :, slice_index]
    predicted = result["mask"][:, :, slice_index]
    columns = st.columns(3)
    columns[0].image(normalize_for_display(original), caption="Original FLAIR", clamp=True)
    columns[1].image(predicted * 255, caption="AI-predicted mask", clamp=True)
    columns[2].image(overlay_mask(original, predicted), caption="FLAIR + prediction overlay", clamp=True)
    area = int(predicted.sum()) * source.spacing[0] * source.spacing[1]
    st.caption(f"Slice {slice_index}: {int(predicted.sum()):,} predicted pixels ({area:.2f} mm²). Largest predicted slice: {default_slice}.")

    mask_filename = f"{Path(uploaded.name).name.split('.')[0]}_pred_seg.nii.gz"
    st.download_button(
        "Download predicted NIfTI mask",
        data=_mask_download(result["mask"], source, mask_filename),
        file_name=mask_filename,
        mime="application/gzip",
    )
    st.warning(f"Research Use Only — {DISCLAIMER}")


if __name__ == "__main__":
    main()
