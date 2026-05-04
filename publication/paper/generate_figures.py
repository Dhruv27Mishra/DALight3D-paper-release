#!/usr/bin/env python3
"""
Generate publication figures from saved training and baseline results.
Run from project root: python publication/paper/generate_figures.py
"""
import json
import os
import sys
import numpy as np

# Project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

# Use Agg for headless plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data paths: canonical results live in results/
RUN_DIR = os.path.join(ROOT, 'results')
PUB_METRICS_PATH = os.path.join(ROOT, 'results', 'publication_metrics.json')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
EVAL_DIR = os.path.join(FIGURES_DIR, 'eval')

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # 1) Load proposed model history
    history_path = os.path.join(RUN_DIR, 'training_history.json')
    with open(history_path, 'r') as f:
        history = json.load(f)

    # 2) Build SOTA results: Proposed + baselines (best Dice per model)
    with open(os.path.join(RUN_DIR, 'baseline_histories.json'), 'r') as f:
        baseline_histories = json.load(f)

    sota_results_list = [
        {
            'name': 'Proposed (DALight-3D)',
            'dice': history['best_dice'],
            'params': history['model_params'],
        }
    ]
    for bl in baseline_histories:
        best_dice = max(bl['val_dices']) if bl['val_dices'] else 0
        sota_results_list.append({
            'name': bl['name'],
            'dice': best_dice,
            'params': bl['params'],
        })

    # 3) Import and run cnn plotting (figures go to publication/paper/figures)
    output_dir = os.path.dirname(__file__)
    from cnn import generate_visualizations, generate_sota_comparison_plots

    print('Generating training curves...')
    generate_visualizations(history, output_dir)

    print('Generating SOTA comparison plots...')
    generate_sota_comparison_plots(sota_results_list, output_dir)

    # 4) Confusion matrix and per-class Dice/IoU from publication_metrics
    if os.path.exists(PUB_METRICS_PATH):
        with open(PUB_METRICS_PATH, 'r') as f:
            pub = json.load(f)
        names = ['BG', 'NCR', 'ED', 'ET']
        num_classes = 4
        conf_mat = np.array(pub['confusion_matrix'])
        per_class = pub['per_class_metrics']
        tumor_classes = [n for n in names[1:] if n in per_class]

        # Confusion matrix
        plt.figure(figsize=(6, 5))
        im = plt.imshow(conf_mat, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix (DALight-3D)', fontsize=14, fontweight='bold')
        plt.colorbar(im)
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, names[:num_classes], rotation=45)
        plt.yticks(tick_marks, names[:num_classes])
        thresh = conf_mat.max() / 2.0 if conf_mat.max() > 0 else 0
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, format(int(conf_mat[i, j]), 'd'), ha='center', va='center',
                         color='white' if conf_mat[i, j] > thresh else 'black', fontsize=8)
        plt.ylabel('True label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print('Saved confusion_matrix.png')

        # Per-class Dice and IoU
        tumor_metrics = [per_class[c] for c in tumor_classes]
        x = np.arange(len(tumor_classes))
        width = 0.35
        plt.figure(figsize=(8, 5))
        dice_vals = [m.get('dice', 0.0) for m in tumor_metrics]
        iou_vals = [m.get('iou', 0.0) for m in tumor_metrics]
        plt.bar(x - width / 2, dice_vals, width, label='Dice', color='#4C72B0', alpha=0.9)
        plt.bar(x + width / 2, iou_vals, width, label='IoU', color='#55A868', alpha=0.9)
        plt.xticks(x, tumor_classes)
        plt.ylim(0, 1.0)
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('Per-Class Dice and IoU (DALight-3D)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, 'per_class_dice_iou.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print('Saved per_class_dice_iou.png')
    else:
        print(f'Publication metrics not found at {PUB_METRICS_PATH}, skipping confusion matrix and per-class figures.')

    # Keep only figures referenced in the paper; remove extras
    keep = {
        'architecture.png', 'methodology_rationale.png',  # methodology (pre-existing)
        'dice_params_tradeoff_scatter.png', 'loss_and_dice.png',
    }
    for f in os.listdir(FIGURES_DIR):
        if f.endswith('.png') and f not in keep:
            p = os.path.join(FIGURES_DIR, f)
            os.remove(p)
            print(f'Removed unused: {f}')

    print(f'All figures written to {FIGURES_DIR}')

if __name__ == '__main__':
    main()
