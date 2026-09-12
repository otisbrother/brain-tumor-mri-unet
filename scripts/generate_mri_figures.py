"""Generate publication-quality Brain MRI visualization figures for README."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from src.inference.predict import load_model_checkpoint
from src.data.preprocessing import zscore_normalize_nonzero
from src.visualization.plotting import normalize_for_display

def make_mri_figures():
    output_dir = Path("docs/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load patient BraTS2021_00000
    pdir = Path("data/extracted/BraTS2021_00000")
    flair_vol = nib.load(pdir / "BraTS2021_00000_flair.nii.gz").get_fdata()
    t1_vol = nib.load(pdir / "BraTS2021_00000_t1.nii.gz").get_fdata()
    t1ce_vol = nib.load(pdir / "BraTS2021_00000_t1ce.nii.gz").get_fdata()
    t2_vol = nib.load(pdir / "BraTS2021_00000_t2.nii.gz").get_fdata()
    seg_vol = nib.load(pdir / "BraTS2021_00000_seg.nii.gz").get_fdata()
    
    # Load trained U-Net model
    device = torch.device("cpu")
    model, _ = load_model_checkpoint({"model": {"in_channels": 1, "base_channels": 16}}, "models/best_model.pt", device)
    model.eval()

    def predict_slice(sl):
        norm = zscore_normalize_nonzero(sl)
        inp = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        return (probs >= 0.35).astype(np.float32)

    # -------------------------------------------------------------
    # FIGURE 1: 4 MRI MODALITIES + GROUND TRUTH (Slice 74)
    # -------------------------------------------------------------
    sl_idx = 74
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), facecolor="#0e1117")
    modalities = [
        ("T1 (Cấu trúc não)", t1_vol[:, :, sl_idx], "gray"),
        ("T1ce (Tiêm đối quang)", t1ce_vol[:, :, sl_idx], "gray"),
        ("T2 (Phù nề & dịch)", t2_vol[:, :, sl_idx], "gray"),
        ("FLAIR (Đầu vào mô hình)", flair_vol[:, :, sl_idx], "gray"),
        ("Ground Truth (Khối u)", None, None),
    ]

    for ax, (title, data, cmap) in zip(axes, modalities):
        ax.set_facecolor("#0e1117")
        if data is not None:
            norm_img = normalize_for_display(data)
            ax.imshow(norm_img, cmap=cmap)
        else:
            # Show FLAIR with ground truth overlay in vibrant coral/red
            base = normalize_for_display(flair_vol[:, :, sl_idx])
            rgb = np.repeat(base[..., None], 3, axis=-1)
            gt = seg_vol[:, :, sl_idx] > 0
            rgb[gt] = [1.0, 0.2, 0.2]  # Red overlay
            ax.imshow(rgb)
        
        ax.set_title(title, color="#ffffff", fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")

    plt.suptitle("Bộ 4 chuỗi xung MRI Não (BraTS 2021) & Vùng u phân đoạn (Slice 74)", 
                 color="#00e5ff", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig1_path = output_dir / "mri_modalities.png"
    plt.savefig(fig1_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------
    # FIGURE 2: SEGMENTATION PREDICTION (FLAIR, GT, Pred, Overlay)
    # -------------------------------------------------------------
    slices_to_show = [
        ("Lát cắt u lớn (Slice 74)", 74),
        ("Lát cắt u trung bình (Slice 62)", 62),
        ("Lát cắt não bình thường (Slice 35)", 35),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), facecolor="#0e1117")
    
    col_titles = [
        "1. Ảnh MRI FLAIR gốc",
        "2. Ground Truth (Bác sĩ)",
        "3. U-Net Dự đoán (AI)",
        "4. So sánh Chồng lấp (Overlay)"
    ]

    for row_idx, (slice_label, s_idx) in enumerate(slices_to_show):
        flair_slice = flair_vol[:, :, s_idx]
        gt_slice = (seg_vol[:, :, s_idx] > 0).astype(np.float32)
        pred_slice = predict_slice(flair_slice)
        
        # Calculate dice
        intersection = np.sum(pred_slice * gt_slice)
        denom = np.sum(pred_slice) + np.sum(gt_slice)
        dice = (2.0 * intersection / denom) if denom > 0 else (1.0 if np.sum(gt_slice) == 0 else 0.0)

        # Columns
        base_norm = normalize_for_display(flair_slice)
        
        # 1. Original FLAIR
        ax = axes[row_idx, 0]
        ax.imshow(base_norm, cmap="gray")
        ax.set_ylabel(f"{slice_label}\nDice: {dice:.3f}", color="#00e5ff", fontsize=12, fontweight="bold", labelpad=10)
        
        # 2. Ground Truth Mask
        ax = axes[row_idx, 1]
        ax.imshow(gt_slice, cmap="Greens_r" if gt_slice.max() > 0 else "gray", vmin=0, vmax=1)
        if gt_slice.max() > 0:
            ax.imshow(gt_slice, cmap="winter", vmin=0, vmax=1)
        
        # 3. Model Prediction Mask
        ax = axes[row_idx, 2]
        ax.imshow(pred_slice, cmap="autumn_r" if pred_slice.max() > 0 else "gray", vmin=0, vmax=1)

        # 4. Color Overlay Comparison
        # Green = True Positive (overlap), Red = False Positive, Blue/Cyan = False Negative
        ax = axes[row_idx, 3]
        overlay_rgb = np.repeat(base_norm[..., None], 3, axis=-1)
        
        tp = (pred_slice == 1) & (gt_slice == 1)
        fp = (pred_slice == 1) & (gt_slice == 0)
        fn = (pred_slice == 0) & (gt_slice == 1)
        
        overlay_rgb[tp] = [0.0, 1.0, 0.0]  # Green: True Positive (trùng khớp)
        overlay_rgb[fp] = [1.0, 0.0, 0.0]  # Red: False Positive (dự đoán thừa)
        overlay_rgb[fn] = [0.0, 0.6, 1.0]  # Light Blue: False Negative (bỏ sót)
        
        ax.imshow(overlay_rgb)

        for col in range(4):
            axes[row_idx, col].set_facecolor("#0e1117")
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])
            if row_idx == 0:
                axes[0, col].set_title(col_titles[col], color="#ffffff", fontsize=13, fontweight="bold", pad=10)

    legend_patches = [
        mpatches.Patch(color=[0.0, 1.0, 0.0], label="Trùng khớp (True Positive)"),
        mpatches.Patch(color=[1.0, 0.0, 0.0], label="Dự đoán thừa (False Positive)"),
        mpatches.Patch(color=[0.0, 0.6, 1.0], label="Bỏ sót (False Negative)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, framealpha=0.8, 
               facecolor="#1e222a", edgecolor="#444", labelcolor="#fff", fontsize=11, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Kết quả phân đoạn thực tế của mạng U-Net trên các lát cắt MRI Não", 
                 color="#00e5ff", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig2_path = output_dir / "mri_prediction_comparison.png"
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------
    # FIGURE 3: 3D SLICES SCAN PROGRESSION (Slices 55, 65, 74, 82, 90)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), facecolor="#0e1117")
    progression_slices = [55, 65, 74, 82, 90]
    
    for ax, s_idx in zip(axes, progression_slices):
        ax.set_facecolor("#0e1117")
        base = normalize_for_display(flair_vol[:, :, s_idx])
        pred = predict_slice(flair_vol[:, :, s_idx])
        rgb = np.repeat(base[..., None], 3, axis=-1)
        rgb[pred > 0] = [1.0, 0.15, 0.15]  # Bright red for tumor
        
        ax.imshow(rgb)
        ax.set_title(f"Lát cắt Z={s_idx}", color="#ffffff", fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")

    plt.suptitle("Mặt cắt Axial đa lát cắt qua thể tích não 3D (Đỏ: Khối u do U-Net phát hiện)", 
                 color="#00e5ff", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig3_path = output_dir / "brain_slices_axial_progression.png"
    plt.savefig(fig3_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {fig3_path}")

if __name__ == "__main__":
    make_mri_figures()
