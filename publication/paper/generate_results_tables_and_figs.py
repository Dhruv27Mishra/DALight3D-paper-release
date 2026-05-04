#!/usr/bin/env python3
"""
Generate publication-friendly results figures from canonical results/.

Outputs into publication/paper/figures/:
  - per_class_metrics_full.png
  - confusion_matrix.png
  - compare_val_dice.png
  - compare_train_loss.png
  - compare_dice_params_bar.png
  - loss_and_dice.png
  - dice_params_tradeoff_scatter.png
"""

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "eval")

RESULTS_DIR = os.path.join(ROOT, "results")
PUB_METRICS = os.path.join(RESULTS_DIR, "publication_metrics.json")
TRAIN_HIST = os.path.join(RESULTS_DIR, "training_history.json")
BASE_HIST = os.path.join(RESULTS_DIR, "baseline_histories.json")

OURS = "DALight-3D"
MODEL_LABELS = {
    "Standard UNet": "Standard 3D U-Net",
    "Attention UNet": "Attention U-Net",
    "Residual UNet": "Residual 3D U-Net",
    "V-Net": "V-Net",
}
MODEL_COLORS = {
    OURS: "#0b3c6f",
    "Standard 3D U-Net": "#4d5563",
    "Attention U-Net": "#5a3e8d",
    "Residual 3D U-Net": "#176d62",
    "V-Net": "#7a4e2d",
}
METRIC_COLORS = {
    "dice": "#1f4e79",
    "iou": "#4c9a8a",
    "precision": "#b55d60",
    "recall": "#8f7ab8",
    "specificity": "#c59a3d",
}


def _paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.weight": "bold",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "legend.fontsize": 8.7,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.18,
        "lines.linewidth": 2.1,
        "figure.dpi": 140,
        "savefig.dpi": 450,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
    })


def _safe_list(values):
    return [v for v in values if v is not None]


def _pretty_name(name):
    return MODEL_LABELS.get(name, name)


def _confusion_consistent_metrics(metrics_dict):
    tp = float(metrics_dict.get("tp", 0.0))
    fp = float(metrics_dict.get("fp", 0.0))
    fn = float(metrics_dict.get("fn", 0.0))
    denom = (2.0 * tp) + fp + fn
    dice = (2.0 * tp / denom) if denom > 0 else 0.0
    return {
        "dice": dice,
        "iou": float(metrics_dict.get("iou", 0.0)),
        "precision": float(metrics_dict.get("precision", 0.0)),
        "recall": float(metrics_dict.get("recall", 0.0)),
        "specificity": float(metrics_dict.get("specificity", 0.0)),
    }


def _style_axes(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, linestyle="--", dashes=(2, 2))
    ax.spines["left"].set_alpha(0.7)
    ax.spines["bottom"].set_alpha(0.7)
    ax.tick_params(length=3, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def _save(fig, filename):
    out = os.path.join(FIG_DIR, filename)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


def _line_kwargs(name):
    if name == OURS:
        return {
            "color": MODEL_COLORS[name],
            "linewidth": 2.8,
            "marker": "o",
            "markersize": 5.2,
            "zorder": 4,
        }
    return {
        "color": MODEL_COLORS[name],
        "linewidth": 1.8,
        "linestyle": "--",
        "marker": "o",
        "markersize": 4.0,
        "alpha": 0.95,
        "zorder": 2,
    }


def _style_legend(ax, **kwargs):
    return ax.legend(prop={"weight": "bold", "size": 8.7}, frameon=True, **kwargs)


def _set_epoch_xlim(ax, epochs_array):
    start = 1
    end = int(epochs_array[-1]) if len(epochs_array) else 1
    if end <= start:
        end = start + 1
    ax.set_xlim(start, end)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    _paper_style()

    with open(PUB_METRICS, "r") as f:
        pm = json.load(f)
    with open(TRAIN_HIST, "r") as f:
        th = json.load(f)
    with open(BASE_HIST, "r") as f:
        bh = json.load(f)

    cm = np.array(pm["confusion_matrix"], dtype=np.int64)
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total > 0 else 0.0

    per_raw = pm["per_class_metrics"]
    per = {cls: _confusion_consistent_metrics(values) for cls, values in per_raw.items()}
    tumor = ["NCR", "ED", "ET"]
    metrics = ["dice", "iou", "precision", "recall", "specificity"]
    metric_labels = {
        "dice": "Dice / F1",
        "iou": "IoU",
        "precision": "Precision",
        "recall": "Sensitivity",
        "specificity": "Specificity",
    }

    print("Overall accuracy:", acc)
    print("Macro tumor metrics:", pm["overall_metrics"])
    print("\nLaTeX rows (NCR/ED/ET):")
    for cls in tumor:
        v = per[cls]
        print(
            f"{cls} & "
            f"{v.get('dice', 0):.3f} & {v.get('iou', 0):.3f} & "
            f"{v.get('precision', 0):.3f} & {v.get('recall', 0):.3f} & {v.get('specificity', 0):.3f} \\\\"
        )

    # ----------------------------
    # 1) Per-class grouped metrics
    # ----------------------------
    x = np.arange(len(tumor))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    all_bars = []
    for i, metric in enumerate(metrics):
        vals = [per[c].get(metric, 0.0) for c in tumor]
        bars = ax.bar(
            x + (i - 2) * width,
            vals,
            width,
            label=metric_labels[metric],
            color=METRIC_COLORS[metric],
            alpha=0.95,
            edgecolor="white",
            linewidth=0.6,
        )
        all_bars.extend(list(bars))

    ax.set_xticks(x)
    ax.set_xticklabels(tumor)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    _style_axes(ax, "y")
    for bar in all_bars:
        height = bar.get_height()
        y = max(height - 0.06, 0.03)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{height:.2f}",
            ha="center",
            va="center",
            fontsize=7.7,
            fontweight="bold",
            color="white",
        )
    _style_legend(ax, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.20), columnspacing=1.1, handlelength=1.6)
    fig.tight_layout()
    _save(fig, "per_class_metrics_full.png")

    # ----------------------------
    # 2) Confusion matrix
    # ----------------------------
    class_names = ["BG", "NCR", "ED", "ET"]
    cm_float = cm.astype(float)
    row_sums = cm_float.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_float, row_sums, out=np.zeros_like(cm_float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion", fontweight="bold")
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    _style_axes(ax, "both")
    ax.grid(False)
    thresh = 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm_norm[i, j]:.2f}\n({int(cm[i, j])})"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "#0b2239",
                fontsize=8.1,
                fontweight="bold",
            )
    fig.tight_layout()
    _save(fig, "confusion_matrix.png")

    # ----------------------------
    # 3) Validation Dice curves
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    epochs = np.array(th["epochs"], dtype=int) + 1
    val_mask = np.array([v is not None for v in th["val_dices"]], dtype=bool)
    val_epochs = epochs[val_mask]
    val_vals = np.array(_safe_list(th["val_dices"]), dtype=float)
    ax.plot(val_epochs, val_vals, label=OURS, **_line_kwargs(OURS))

    for model in bh:
        label = _pretty_name(model["name"])
        ax.plot(
            np.array(model["epochs"], dtype=int) + 1,
            np.array(model["val_dices"], dtype=float),
            label=label,
            **_line_kwargs(label),
        )

    ax.scatter([val_epochs[-1]], [val_vals[-1]], s=40, color=MODEL_COLORS[OURS], zorder=5)
    ax.annotate(
        f"{val_vals[-1]:.3f}",
        (val_epochs[-1], val_vals[-1]),
        xytext=(7, 6),
        textcoords="offset points",
        fontsize=8.5,
        color=MODEL_COLORS[OURS],
        fontweight="bold",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice")
    _set_epoch_xlim(ax, epochs)
    ax.set_ylim(0.25, 0.72)
    _style_axes(ax, "both")
    _style_legend(ax, loc="lower right", ncol=1)
    fig.tight_layout()
    _save(fig, "compare_val_dice.png")

    # ----------------------------
    # 3b) Combined loss and Dice
    # ----------------------------
    fig, ax1 = plt.subplots(figsize=(6.9, 4.0))
    loss_epochs = np.array(th["epochs"], dtype=int) + 1
    train_losses = np.array(th["train_losses"], dtype=float)
    ax1.plot(
        loss_epochs,
        train_losses,
        color="#4c9a8a",
        linewidth=2.5,
        marker="o",
        markersize=4.2,
        label="Training Loss",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color="#2f6f64")
    ax1.tick_params(axis="y", labelcolor="#2f6f64")
    _set_epoch_xlim(ax1, loss_epochs)
    _style_axes(ax1, "both")

    ax2 = ax1.twinx()
    ax2.plot(
        val_epochs,
        val_vals,
        color=MODEL_COLORS[OURS],
        linewidth=2.6,
        marker="o",
        markersize=4.8,
        label="Validation Dice",
    )
    ax2.set_ylabel("Validation Dice", color=MODEL_COLORS[OURS])
    ax2.tick_params(axis="y", labelcolor=MODEL_COLORS[OURS])
    ax2.set_ylim(0.25, 0.72)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_alpha(0.7)
    for label in ax2.get_yticklabels():
        label.set_fontweight("bold")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="center right", prop={"weight": "bold", "size": 8.7}, frameon=True)
    fig.tight_layout()
    _save(fig, "loss_and_dice.png")

    # ----------------------------
    # 4) Training loss curves
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    ax.plot(
        np.array(th["epochs"], dtype=int) + 1,
        np.array(th["train_losses"], dtype=float),
        label=OURS,
        **_line_kwargs(OURS),
    )
    for model in bh:
        label = _pretty_name(model["name"])
        ax.plot(
            np.array(model["epochs"], dtype=int) + 1,
            np.array(model["train_losses"], dtype=float),
            label=label,
            **_line_kwargs(label),
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    _set_epoch_xlim(ax, epochs)
    ymin = min(min(th["train_losses"]), *(min(m["train_losses"]) for m in bh)) * 0.92
    ymax = max(max(th["train_losses"]), *(max(m["train_losses"]) for m in bh)) * 1.04
    ax.set_ylim(ymin, ymax)
    _style_axes(ax, "both")
    _style_legend(ax, loc="upper right", ncol=1)
    fig.tight_layout()
    _save(fig, "compare_train_loss.png")

    # ----------------------------
    # 5) Dice vs Params comparison
    # ----------------------------
    names = [OURS] + [_pretty_name(model["name"]) for model in bh]
    dices = [float(th["best_dice"])] + [float(max(model["val_dices"])) for model in bh]
    params_m = [float(th["model_params"]) / 1e6] + [float(model["params"]) / 1e6 for model in bh]
    x = np.arange(len(names))
    width = 0.34

    fig, ax1 = plt.subplots(figsize=(7.6, 4.1))
    dice_colors = [MODEL_COLORS[name] for name in names]
    bars1 = ax1.bar(
        x - width / 2,
        dices,
        width,
        color=dice_colors,
        edgecolor="white",
        linewidth=0.6,
    )
    bars1[0].set_hatch("//")
    ax1.set_ylabel("Mean Dice")
    ax1.set_ylim(0, 0.76)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=14, ha="right")
    _style_axes(ax1, "y")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        params_m,
        width,
        color="#d9d9d9",
        edgecolor="#7f7f7f",
        linewidth=0.6,
    )
    bars2[0].set_facecolor("#f0d77b")
    bars2[0].set_edgecolor("#9a7b14")
    ax2.set_ylabel("Parameters (M)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_alpha(0.7)
    ax2.tick_params(length=3, width=0.8)

    for bar, val in zip(bars1, dices):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    for bar, val in zip(bars2, params_m):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=8.0)

    fig.tight_layout()
    _save(fig, "compare_dice_params_bar.png")

    # ----------------------------
    # 6) Dice-parameter trade-off
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for name, dice, params in zip(names, dices, params_m):
        size = 120 if name == OURS else 88
        edge = "#0d223a" if name == OURS else "white"
        lw = 1.0 if name == OURS else 0.8
        ax.scatter(
            params,
            dice,
            s=size,
            color=MODEL_COLORS[name],
            edgecolors=edge,
            linewidths=lw,
            zorder=4 if name == OURS else 3,
        )
        ax.annotate(
            name,
            (params, dice),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color=MODEL_COLORS[name],
        )

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Mean Dice")
    ax.set_xlim(2.1, 3.35)
    ax.set_ylim(0.54, 0.69)
    _style_axes(ax, "both")
    fig.tight_layout()
    _save(fig, "dice_params_tradeoff_scatter.png")


if __name__ == "__main__":
    main()
