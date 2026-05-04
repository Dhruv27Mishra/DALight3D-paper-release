# DALight-3D (Paper Release)

Clean repository for the DALight-3D paper: training code, paper source, plotting scripts, and curated 50-epoch result artifacts.

## Included

- `cnn.py`: main model/training/evaluation script
- `cnnv2.py`: experiment orchestrator (staged runs, manifests, logs)
- `publication/paper/`: LaTeX paper source and figure-generation scripts
- `publication/paper/figures/eval/`: generated evaluation figures used in the paper
- `results_50ep/`: curated 50-epoch proposed + baseline result JSON/TXT artifacts

## Not Included

- Full dataset (`Task01_BrainTumour`) is **not** included due to size
- Model checkpoint files (`.pth`) are not included in this clean release

## Setup

```bash
python -m pip install -r requirements.txt
```

## Dataset Instructions

The code expects Medical Segmentation Decathlon `Task01_BrainTumour`.

You have two options:

1. **Automatic download via script** (recommended):
   - Use `cnnv2.py` with `--download_dataset`
   - Dataset will be downloaded/extracted under `--data_dir`

2. **Manual placement**:
   - Place dataset at `./data/Task01_BrainTumour`
   - Structure should contain `imagesTr/`, `labelsTr/` (and optionally `imagesTs/`)

## Example Commands

### Run paper profile

```bash
python cnnv2.py --profile paper_full --data_dir ./data --output_root ./cnnv2_results --download_dataset
```

### Ablations only

```bash
python cnnv2.py --stages ablations --ablation_epochs 25 --data_dir ./data --output_root ./cnnv2_results --download_dataset
```

### Generate publication result figures from canonical `results/`

```bash
python publication/paper/generate_results_tables_and_figs.py
```

## Reproducibility Notes

- If you regenerate figures, ensure your `results/` folder contains a consistent set of:
  - `training_history.json`
  - `baseline_histories.json`
  - `publication_metrics.json`
- Curated 50-epoch artifacts are provided in `results_50ep/` for reference.
