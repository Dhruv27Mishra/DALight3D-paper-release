#!/usr/bin/env python3
"""
Generate ablation summary table/figure from completed ablation runs.

Expected usage:
  python publication/paper/generate_ablation_tables_and_figs.py \
    --run full=/abs/path/to/run1 \
    --run no_sepconv=/abs/path/to/run2
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "eval")

LABELS = {
    "full": "Full DALight-3D",
    "no_sepconv": "w/o SepConv",
    "no_scannorm": "w/o ScannerAwareNorm",
    "no_csa": "w/o CSA",
    "no_ssfb": "w/o SSFB",
}

COLORS = {
    "full": "#0b3c6f",
    "no_sepconv": "#4d5563",
    "no_scannorm": "#7a4e2d",
    "no_csa": "#5a3e8d",
    "no_ssfb": "#176d62",
}


def _paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.weight": "bold",
        "font.size": 10,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "savefig.dpi": 450,
    })


def _load_history(run_dir):
    history_path = os.path.join(run_dir, "training_history.json")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Missing training history: {history_path}")
    with open(history_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate ablation summary figure/table")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Ablation mapping in the form key=/absolute/or/relative/run_dir",
    )
    args = parser.parse_args()

    if not args.run:
        raise SystemExit("Provide at least one --run key=path entry.")

    os.makedirs(FIG_DIR, exist_ok=True)
    _paper_style()

    entries = []
    for item in args.run:
        key, value = item.split("=", 1)
        run_dir = value if os.path.isabs(value) else os.path.join(ROOT, value)
        hist = _load_history(run_dir)
        entries.append({
            "key": key,
            "label": LABELS.get(key, key),
            "best_dice": float(hist["best_dice"]),
            "params_m": float(hist["model_params"]) / 1e6,
        })

    entries.sort(key=lambda x: (0 if x["key"] == "full" else 1, x["label"]))

    print("\nLaTeX ablation rows:")
    for row in entries:
        print(f"{row['label']} & {row['best_dice']:.3f} & {row['params_m']:.2f} \\\\")

    x = np.arange(len(entries))
    fig, ax1 = plt.subplots(figsize=(7.6, 4.2))
    bars1 = ax1.bar(
        x - 0.18,
        [row["best_dice"] for row in entries],
        0.36,
        color=[COLORS.get(row["key"], "#4d5563") for row in entries],
        edgecolor="white",
        linewidth=0.7,
    )
    ax1.set_ylabel("Mean Dice")
    ax1.set_ylim(0, max(row["best_dice"] for row in entries) * 1.15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([row["label"] for row in entries], rotation=18, ha="right", fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.2)
    for bar, row in zip(bars1, entries):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{row['best_dice']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
        )

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + 0.18,
        [row["params_m"] for row in entries],
        0.36,
        color="#d9d9d9",
        edgecolor="#7f7f7f",
        linewidth=0.6,
    )
    ax2.set_ylabel("Parameters (M)")
    for bar, row in zip(bars2, entries):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{row['params_m']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ablation_dice_params.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
