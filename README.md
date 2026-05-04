# DALight-3D (Minimal Runnable Release)

Minimal repository to run DALight-3D training and experiment stages.

## Included

- `cnn.py`: main training/evaluation script
- `cnnv2.py`: orchestrator for staged runs (proposed, baselines, ablations)
- `requirements.txt`: Python dependencies

## Not Included

- Dataset files
- Any result artifacts (`results/`, logs, plots)
- PDFs, images, and model checkpoints

## Setup

```bash
python -m pip install -r requirements.txt
```

## Dataset Instructions

The code uses Medical Segmentation Decathlon `Task01_BrainTumour`.

### Option A: automatic download (recommended)

```bash
python cnnv2.py --profile paper_full --data_dir ./data --output_root ./cnnv2_results --download_dataset
```

### Option B: manual dataset placement

Place dataset at:

`./data/Task01_BrainTumour`

Expected subfolders:

- `imagesTr/`
- `labelsTr/`
- `imagesTs/` (optional)

## Example Runs

### Full profile

```bash
python cnnv2.py --profile paper_full --data_dir ./data --output_root ./cnnv2_results --download_dataset
```

### Ablations only

```bash
python cnnv2.py --stages ablations --ablation_epochs 25 --data_dir ./data --output_root ./cnnv2_results --download_dataset
```
