#!/usr/bin/env python3
"""
Generate qualitative, step-by-step method figures from a real Decathlon BraTS case.

Outputs (into publication/paper/figures/):
  - preprocess_steps.png: modalities + mask montage (multiple cases)
  - qualitative_segmentation.png: qualitative segmentation montage (multiple cases)
  - decision_process.png: confidence + uncertainty visualization (multiple cases)
"""

import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
DATA_DIR = os.path.join(ROOT, "data", "Task01_BrainTumour")
WEIGHTS = os.path.join(ROOT, "results", "best_model.pth")
NUM_CASES = 2
PATCH = (64, 64, 64)
RNG_SEED = 7


def _list_cases(images_dir: Path, labels_dir: Path):
    cases = []
    for img_file in sorted(images_dir.glob("*.nii.gz")):
        if img_file.name.startswith("._") or img_file.name.startswith("."):
            continue
        label_file = labels_dir / img_file.name
        if label_file.exists() and not label_file.name.startswith("._"):
            cases.append((img_file, label_file))
    if not cases:
        raise FileNotFoundError(f"No valid cases found in {images_dir}")
    return cases


def _normalize_like_training(image_cdhw: np.ndarray) -> np.ndarray:
    image = image_cdhw.copy().astype(np.float32)
    for c in range(image.shape[0]):
        mask = image[c] > 0
        if mask.sum() > 0:
            mean_val = float(image[c][mask].mean())
            std_val = float(image[c][mask].std())
            if std_val > 1e-8:
                image[c][mask] = (image[c][mask] - mean_val) / std_val
            else:
                image[c][mask] = 0.0
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image


def _center_crop_around_tumor(image: np.ndarray, seg: np.ndarray, patch=(64, 64, 64)):
    C, D, H, W = image.shape
    pd, ph, pw = patch

    coords = np.argwhere(seg > 0)
    if coords.size == 0:
        center = np.array([D // 2, H // 2, W // 2], dtype=np.int64)
    else:
        center = coords.mean(axis=0).round().astype(np.int64)

    d0 = int(np.clip(center[0] - pd // 2, 0, max(0, D - pd)))
    h0 = int(np.clip(center[1] - ph // 2, 0, max(0, H - ph)))
    w0 = int(np.clip(center[2] - pw // 2, 0, max(0, W - pw)))

    img_patch = image[:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]
    seg_patch = seg[d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]

    # Pad if needed (edge cases)
    if img_patch.shape[1:] != patch:
        pad_d = pd - img_patch.shape[1]
        pad_h = ph - img_patch.shape[2]
        pad_w = pw - img_patch.shape[3]
        img_patch = np.pad(img_patch, [(0, 0), (0, pad_d), (0, pad_h), (0, pad_w)])
        seg_patch = np.pad(seg_patch, [(0, pad_d), (0, pad_h), (0, pad_w)])
    return img_patch, seg_patch, (d0, h0, w0)


def _pick_slice(seg_dhw: np.ndarray) -> int:
    # Choose a slice with tumor if possible.
    per_slice = seg_dhw.reshape(seg_dhw.shape[0], -1).sum(axis=1)
    if per_slice.max() > 0:
        return int(per_slice.argmax())
    return int(seg_dhw.shape[0] // 2)

def _clip_for_display(x2d: np.ndarray) -> np.ndarray:
    """Robust windowing for MRI display using percentiles on non-zero voxels."""
    x = x2d.astype(np.float32)
    nz = x[np.isfinite(x) & (x != 0)]
    if nz.size < 50:
        vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))
    else:
        vmin, vmax = np.percentile(nz, [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1.0
    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin + 1e-8)
    return x

def _overlay_alpha(ax, base: np.ndarray, mask: np.ndarray, color: str, alpha: float):
    """Overlay a boolean mask on base grayscale."""
    ax.imshow(base, cmap="gray", vmin=0, vmax=1)
    if mask.any():
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        # hex -> rgb
        color = color.lstrip("#")
        r, g, b = int(color[0:2], 16) / 255.0, int(color[2:4], 16) / 255.0, int(color[4:6], 16) / 255.0
        rgba[..., 0] = r
        rgba[..., 1] = g
        rgba[..., 2] = b
        rgba[..., 3] = alpha * mask.astype(np.float32)
        ax.imshow(rgba)

def _entropy_from_probs(probs_chw: np.ndarray) -> np.ndarray:
    """Shannon entropy over classes for each pixel."""
    p = np.clip(probs_chw.astype(np.float32), 1e-8, 1.0)
    ent = -(p * np.log(p)).sum(axis=0)
    return ent

def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x[np.isfinite(x)], [1, 99]) if np.isfinite(x).any() else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-8)

def _overlay_mask(ax, mask: np.ndarray, color, label: str):
    # mask: HxW bool
    if mask.sum() == 0:
        return
    ax.contour(mask.astype(np.uint8), levels=[0.5], colors=[color], linewidths=1.2)
    ax.plot([], [], color=color, label=label)  # legend handle


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    images_dir = Path(DATA_DIR) / "imagesTr"
    labels_dir = Path(DATA_DIR) / "labelsTr"
    if not images_dir.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_DIR}. Expected imagesTr/labelsTr.")

    import nibabel as nib

    cases = _list_cases(images_dir, labels_dir)
    rng = np.random.default_rng(RNG_SEED)
    rng.shuffle(cases)

    # Filter to tumor-containing cases
    picked = []
    for img_file, label_file in cases:
        seg = nib.load(str(label_file)).get_fdata().astype(np.int64)
        seg = np.clip(seg, 0, 3)
        if (seg > 0).sum() > 500:  # ensure visible tumor
            picked.append((img_file, label_file))
        if len(picked) >= NUM_CASES:
            break
    if len(picked) < NUM_CASES:
        picked = cases[:NUM_CASES]

    modality_cols = [3, 0, 1, 2]  # display order: FLAIR, T1, T1ce, T2
    modality_titles = ["FLAIR", "T1", "T1ce", "T2"]

    # Load and preprocess picked cases
    loaded = []
    for img_file, label_file in picked:
        case_id = img_file.name.replace(".nii.gz", "")
        img = nib.load(str(img_file)).get_fdata().astype(np.float32)
        if img.ndim == 3:
            img = np.stack([img] * 4, axis=-1)
        img_cdhw = np.transpose(img, (3, 0, 1, 2))
        seg = nib.load(str(label_file)).get_fdata().astype(np.int64)
        seg = np.clip(seg, 0, 3)
        norm_cdhw = _normalize_like_training(img_cdhw)
        patch_img, patch_seg, origin = _center_crop_around_tumor(norm_cdhw, seg, patch=PATCH)
        s_patch = _pick_slice(patch_seg)
        # keep normalized full volume for nicer visualization context
        loaded.append((case_id, norm_cdhw, seg, patch_img, patch_seg, s_patch, origin))

    # Figure 1: modalities + mask montage (like paper-style Fig.5)
    # Keep fewer cases with larger panels for print clarity.
    fig, axes = plt.subplots(nrows=len(loaded), ncols=5, figsize=(14.5, 5.0 * len(loaded)))
    if len(loaded) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (case_id, _norm_full, _seg_full, patch_img, patch_seg, s, _origin) in enumerate(loaded):
        for c, mod_idx in enumerate(modality_cols):
            img2d = _clip_for_display(patch_img[mod_idx, s])
            axes[r, c].imshow(img2d, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(modality_titles[c], fontsize=12, fontweight="bold")
            # Panel labels (a)-(e) per row
            axes[r, c].text(
                0.5,
                -0.08,
                f"({chr(ord('a') + c)})",
                transform=axes[r, c].transAxes,
                ha="center",
                va="top",
                fontsize=12,
            )

        # Mask column (binary tumor for clarity)
        tumor = (patch_seg[s] > 0).astype(np.float32)
        axes[r, 4].imshow(tumor, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r, 4].axis("off")
        if r == 0:
            axes[r, 4].set_title("Mask", fontsize=12, fontweight="bold")
        axes[r, 4].text(
            0.5,
            -0.08,
            "(e)",
            transform=axes[r, 4].transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )
        # left label = case id
        axes[r, 0].text(-0.02, 0.5, f"Case {r+1}", transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=11, fontweight="bold")

    fig.suptitle("Visualization of preprocessing inputs (tumor-centered patches)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig_path = os.path.join(FIG_DIR, "preprocess_steps.png")
    fig.savefig(fig_path, dpi=450, bbox_inches="tight")
    plt.close(fig)

    # Inference on patch (CPU)
    from cnn import LightweightUNet3D

    device = torch.device("cpu")
    model = LightweightUNet3D(
        in_ch=4,
        out_ch=4,
        base_filters=24,
        num_stages=4,
        use_separable=True,
        use_ssfb=True,
        use_uncertainty=False,
        use_scanner_norm=True,
        max_channels=384,
        ssfb_rank=8,
        bottleneck_channels=432,
    ).to(device)

    ckpt = torch.load(WEIGHTS, map_location="cpu")
    # Support both plain state_dict and checkpoint dicts
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    # Figure 2: qualitative segmentation montage (like paper-style multi-panel grids)
    colors = {"NCR": "#E24A33", "ED": "#348ABD", "ET": "#988ED5"}
    class_ids = {"NCR": 1, "ED": 2, "ET": 3}

    fig, axes = plt.subplots(nrows=len(loaded), ncols=6, figsize=(17.0, 5.1 * len(loaded)))
    if len(loaded) == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Original (FLAIR)", "Ground truth", "All classes", "NCR pred", "ED pred", "ET pred"]
    for c in range(6):
        axes[0, c].set_title(col_titles[c], fontsize=12, fontweight="bold")

    for r, (case_id, _norm_full, _seg_full, patch_img, patch_seg, s, _origin) in enumerate(loaded):
        # model inference
        x = torch.from_numpy(patch_img.copy()).unsqueeze(0)
        scanner_id = torch.tensor([hash(case_id) % 8], dtype=torch.long)
        with torch.no_grad():
            out = model(x.to(device), scanner_id.to(device))
            pred = out["probs"].argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

        flair = _clip_for_display(patch_img[3, s])
        gt = patch_seg[s]
        pr = pred[s]

        # 0) original
        axes[r, 0].imshow(flair, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r, 0].axis("off")

        # 1) ground truth (binary tumor for legibility)
        axes[r, 1].imshow((gt > 0).astype(np.float32), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r, 1].axis("off")

        # 2) all classes overlay (GT)
        _overlay_alpha(axes[r, 2], flair, gt == 1, colors["NCR"], alpha=0.45)
        _overlay_alpha(axes[r, 2], flair, gt == 2, colors["ED"], alpha=0.35)
        _overlay_alpha(axes[r, 2], flair, gt == 3, colors["ET"], alpha=0.45)
        axes[r, 2].axis("off")

        # 3-5) per-class predictions overlay
        _overlay_alpha(axes[r, 3], flair, pr == 1, colors["NCR"], alpha=0.55)
        axes[r, 3].axis("off")
        _overlay_alpha(axes[r, 4], flair, pr == 2, colors["ED"], alpha=0.55)
        axes[r, 4].axis("off")
        _overlay_alpha(axes[r, 5], flair, pr == 3, colors["ET"], alpha=0.55)
        axes[r, 5].axis("off")

        # Panel labels (a)-(f) under each column
        for c in range(6):
            axes[r, c].text(
                0.5,
                -0.08,
                f"({chr(ord('a') + c)})",
                transform=axes[r, c].transAxes,
                ha="center",
                va="top",
                fontsize=12,
            )

        axes[r, 0].text(-0.02, 0.5, f"Case {r+1}", transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=11, fontweight="bold")

    fig.suptitle("Qualitative results on random tumor-containing cases (tumor-centered patches)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig_path = os.path.join(FIG_DIR, "qualitative_segmentation.png")
    fig.savefig(fig_path, dpi=450, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: decision process (confidence + uncertainty)
    # Columns: Original (FLAIR), Prediction (all classes), Confidence, Uncertainty
    fig, axes = plt.subplots(nrows=len(loaded), ncols=4, figsize=(14.5, 5.0 * len(loaded)))
    if len(loaded) == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Original (FLAIR)", "Prediction (overlay)", "Confidence (max prob)", "Uncertainty (entropy)"]
    for c in range(4):
        axes[0, c].set_title(col_titles[c], fontsize=12, fontweight="bold")

    for r, (case_id, norm_full, seg_full, patch_img, patch_seg, s, origin) in enumerate(loaded):
        x = torch.from_numpy(patch_img.copy()).unsqueeze(0)
        scanner_id = torch.tensor([hash(case_id) % 8], dtype=torch.long)
        with torch.no_grad():
            out = model(x.to(device), scanner_id.to(device))
            probs = out["probs"].squeeze(0).cpu().numpy().astype(np.float32)  # (C,D,H,W)
            pred = probs.argmax(axis=0).astype(np.int64)  # (D,H,W)

        # Build full-slice canvases for visualization (so the brain looks normal-sized)
        d0, h0, w0 = origin
        pd, ph, pw = PATCH
        full_flair = _clip_for_display(norm_full[3, d0 + s])  # (H,W) full slice at corresponding depth
        H, W = full_flair.shape

        pr2d_patch = pred[s]              # (ph,pw)
        probs2d = probs[:, s]             # (C,ph,pw)
        conf2d = probs2d.max(axis=0)
        ent2d = _entropy_from_probs(probs2d)

        # Paste patch maps back into full slice coordinates
        pr2d = np.zeros((H, W), dtype=np.int64)
        pr2d[h0:h0 + ph, w0:w0 + pw] = pr2d_patch
        conf_full = np.zeros((H, W), dtype=np.float32)
        conf_full[h0:h0 + ph, w0:w0 + pw] = conf2d
        ent_full = np.zeros((H, W), dtype=np.float32)
        ent_full[h0:h0 + ph, w0:w0 + pw] = ent2d

        # Original
        axes[r, 0].imshow(full_flair, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r, 0].axis("off")

        # Prediction overlay (all classes)
        _overlay_alpha(axes[r, 1], full_flair, pr2d == 1, colors["NCR"], alpha=0.55)
        _overlay_alpha(axes[r, 1], full_flair, pr2d == 2, colors["ED"], alpha=0.45)
        _overlay_alpha(axes[r, 1], full_flair, pr2d == 3, colors["ET"], alpha=0.55)
        axes[r, 1].axis("off")

        # Confidence heatmap
        axes[r, 2].imshow(full_flair, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r, 2].imshow(_norm01(conf_full), cmap="viridis", alpha=0.55, interpolation="nearest")
        axes[r, 2].axis("off")

        # Uncertainty heatmap
        axes[r, 3].imshow(full_flair, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r, 3].imshow(_norm01(ent_full), cmap="magma", alpha=0.55, interpolation="nearest")
        axes[r, 3].axis("off")

        axes[r, 0].text(-0.02, 0.5, f"Case {r+1}", transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=11, fontweight="bold")

        for c in range(4):
            axes[r, c].text(
                0.5, -0.08, f"({chr(ord('a') + c)})",
                transform=axes[r, c].transAxes, ha="center", va="top", fontsize=12
            )

    fig.suptitle("Model decision process: prediction, confidence, and uncertainty", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig_path = os.path.join(FIG_DIR, "decision_process.png")
    fig.savefig(fig_path, dpi=450, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", os.path.join(FIG_DIR, "preprocess_steps.png"))
    print("Saved:", os.path.join(FIG_DIR, "qualitative_segmentation.png"))
    print("Saved:", os.path.join(FIG_DIR, "decision_process.png"))


if __name__ == "__main__":
    main()

