#!/usr/bin/env python3
"""
Paper-run orchestrator for DALight-3D experiments.

This script wraps `cnn.py` so a collaborator can run the required experiments
and send back a clean results bundle with:
  - proposed-model training history
  - baseline training histories
  - ablation training histories
  - canonical merged results
  - generated paper figures
  - per-stage logs and a run manifest

Recommended default:
  python cnnv2.py --profile paper_full --output_root ./cnnv2_results

Lower-compute fallback:
  python cnnv2.py --profile paper_priority --output_root ./cnnv2_results
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback if tqdm is unavailable
    tqdm = None


ROOT = Path(__file__).resolve().parent
CNN_SCRIPT = ROOT / "cnn.py"
RESULTS_SCRIPT = ROOT / "publication" / "paper" / "generate_results_tables_and_figs.py"
ABLATION_SCRIPT = ROOT / "publication" / "paper" / "generate_ablation_tables_and_figs.py"
QUAL_SCRIPT = ROOT / "publication" / "paper" / "generate_qualitative_figures.py"
DATASET_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
DATASET_FOLDER = "Task01_BrainTumour"

PROPOSED_FILES = [
    "training_history.json",
    "best_model.pth",
    "final_model.pth",
    "training_summary.txt",
]

ABLATION_KEYS = ["no_sepconv", "no_scannorm", "no_csa", "no_ssfb"]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DALight-3D paper experiments and collect results.")
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data"), help="Dataset root or parent data directory.")
    parser.add_argument("--output_root", type=str, default=str(ROOT / "cnnv2_results"), help="Directory where all run outputs will be stored.")
    parser.add_argument("--profile", choices=["paper_full", "paper_priority"], default="paper_full")
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages to run: proposed,baselines,attention,ablations,figures,qualitative or 'all'.",
    )
    parser.add_argument("--proposed_epochs", type=int, default=50)
    parser.add_argument("--baseline_epochs", type=int, default=50)
    parser.add_argument("--ablation_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--base_filters", type=int, default=24)
    parser.add_argument("--bottleneck_channels", type=int, default=432)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--validate_every", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--torch_threads", type=int, default=0)
    parser.add_argument("--torch_interop_threads", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--amp_mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument(
        "--download_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the Task01_BrainTumour dataset automatically if it is missing.",
    )
    parser.add_argument(
        "--include_attention",
        action="store_true",
        help="When using paper_priority, also run Attention U-Net as a separate baseline stage.",
    )
    return parser.parse_args()


def resolve_amp(amp_mode: str) -> bool:
    if amp_mode == "on":
        return True
    if amp_mode == "off":
        return False
    try:
        import torch
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    name = torch.cuda.get_device_name(0).upper()
    return "H100" not in name


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def resolve_dataset_path(data_dir: str, allow_download: bool) -> Path:
    requested = Path(data_dir).resolve()
    if requested.name == DATASET_FOLDER and requested.exists():
        return requested

    candidate = requested / DATASET_FOLDER
    if candidate.exists():
        return candidate

    if not allow_download:
        raise FileNotFoundError(
            f"Dataset not found at {requested} or {candidate}. "
            f"Re-run with --download_dataset or place {DATASET_FOLDER} there."
        )

    requested.mkdir(parents=True, exist_ok=True)
    tar_path = requested / f"{DATASET_FOLDER}.tar"
    print(f"Dataset missing. Downloading {DATASET_FOLDER} from:")
    print(DATASET_URL)
    urllib.request.urlretrieve(DATASET_URL, tar_path)
    print("Extracting dataset...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=requested)
    tar_path.unlink(missing_ok=True)
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset download finished but {candidate} was not found.")
    return candidate


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def merge_baseline_histories(output_path: Path, history_paths: list[Path]) -> None:
    merged = []
    seen = set()
    for path in history_paths:
        rows = load_json(path)
        if not rows:
            continue
        for row in rows:
            name = row.get("name")
            if name in seen:
                continue
            merged.append(row)
            seen.add(name)
    write_json(output_path, merged)


def summarize_proposed(run_dir: Path) -> dict[str, Any]:
    hist = load_json(run_dir / "training_history.json") or {}
    return {
        "best_dice": hist.get("best_dice"),
        "model_params": hist.get("model_params"),
        "total_epochs": hist.get("total_epochs"),
    }


def summarize_baselines(run_dir: Path) -> dict[str, Any]:
    rows = load_json(run_dir / "baseline_histories.json") or []
    summary = []
    for row in rows:
        vals = [v for v in row.get("val_dices", []) if v is not None]
        summary.append(
            {
                "name": row.get("name"),
                "params": row.get("params"),
                "best_val_dice": max(vals) if vals else None,
                "epochs": len(row.get("epochs", [])),
            }
        )
    return {"baselines": summary}


def build_base_command(args: argparse.Namespace, output_dir: Path, epochs: int) -> list[str]:
    cmd = [
        sys.executable,
        str(CNN_SCRIPT),
        "--data_dir",
        str(args.data_dir),
        "--output_dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--validate_every",
        str(args.validate_every),
        "--batch_size",
        str(args.batch_size),
        "--patch_size",
        str(args.patch_size),
        "--base_filters",
        str(args.base_filters),
        "--bottleneck_channels",
        str(args.bottleneck_channels),
        "--lr",
        str(args.lr),
        "--workers",
        str(args.workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
    ]
    if args.torch_threads > 0:
        cmd += ["--torch_threads", str(args.torch_threads)]
    if args.torch_interop_threads > 0:
        cmd += ["--torch_interop_threads", str(args.torch_interop_threads)]
    if args.max_cases is not None:
        cmd += ["--max_cases", str(args.max_cases)]
    if resolve_amp(args.amp_mode):
        cmd.append("--amp")
    return cmd


def run_and_log(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(f"$ {shlex.join(cmd)}\n\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def ensure_figure_inputs(canonical: Path) -> None:
    required = [
        canonical / "training_history.json",
        canonical / "baseline_histories.json",
        canonical / "publication_metrics.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required inputs for evaluation figures:\n  - " + "\n  - ".join(missing)
        )
    rows = load_json(canonical / "baseline_histories.json") or []
    if not rows:
        raise ValueError("baseline_histories.json is empty; run the baseline stage before generating figures.")


def ensure_qualitative_inputs(canonical: Path) -> None:
    required = [
        canonical / "best_model.pth",
        canonical / "training_history.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required inputs for qualitative figures:\n  - " + "\n  - ".join(missing)
        )


def sync_canonical_results(output_root: Path, runs_dir: Path) -> Path:
    canonical = output_root / "canonical_results"
    canonical.mkdir(parents=True, exist_ok=True)

    proposed_dir = runs_dir / "proposed_50ep"
    baseline_dir = runs_dir / "baselines_50ep"
    priority_dir = runs_dir / "baselines_priority_50ep"
    attention_dir = runs_dir / "baseline_attention_50ep"

    for name in PROPOSED_FILES:
        copy_if_exists(proposed_dir / name, canonical / name)
    copy_if_exists(proposed_dir / "publication" / "publication_metrics.json", canonical / "publication_metrics.json")
    copy_if_exists(proposed_dir / "publication" / "publication_summary.json", canonical / "publication_summary.json")

    merge_baseline_histories(
        canonical / "baseline_histories.json",
        [
            baseline_dir / "baseline_histories.json",
            priority_dir / "baseline_histories.json",
            attention_dir / "baseline_histories.json",
        ],
    )
    return canonical


def sync_into_repo_results(canonical: Path) -> Path:
    repo_results = ROOT / "results"
    repo_results.mkdir(parents=True, exist_ok=True)
    for item in canonical.iterdir():
        if item.is_file():
            shutil.copy2(item, repo_results / item.name)
    return repo_results


def run_stage(
    stage_name: str,
    cmd: list[str],
    log_dir: Path,
    manifest: dict[str, Any],
    output_dir: Path | None = None,
    progress_bar: Any | None = None,
    stage_units: int = 1,
    progress_state: dict[str, Any] | None = None,
) -> float:
    stage_started = time.time()
    stage = {
        "name": stage_name,
        "started_at": now_iso(),
        "command": cmd,
        "output_dir": str(output_dir) if output_dir else None,
        "status": "running",
    }
    manifest["stages"].append(stage)
    write_json(Path(manifest["manifest_path"]), manifest)

    try:
        run_and_log(cmd, log_dir / f"{stage_name}.log")
        stage["status"] = "completed"
    except Exception as exc:
        stage["status"] = "failed"
        stage["error"] = str(exc)
        stage["finished_at"] = now_iso()
        write_json(Path(manifest["manifest_path"]), manifest)
        raise

    stage["finished_at"] = now_iso()
    if output_dir is not None:
        if (output_dir / "training_history.json").exists():
            stage["summary"] = summarize_proposed(output_dir)
        elif (output_dir / "baseline_histories.json").exists():
            stage["summary"] = summarize_baselines(output_dir)
    elapsed_sec = max(0.0, time.time() - stage_started)
    stage["elapsed_sec"] = round(elapsed_sec, 2)
    write_json(Path(manifest["manifest_path"]), manifest)
    if progress_bar is not None:
        progress_bar.update(stage_units)
        if progress_state is not None:
            progress_state["completed_units"] += stage_units
            eta = estimate_remaining_time(
                progress_state["start_time"],
                progress_state["completed_units"],
                progress_state["total_units"],
            )
            progress_bar.set_postfix_str(f"eta={eta}")
    return elapsed_sec


def parse_stage_selection(args: argparse.Namespace) -> set[str]:
    if args.stages == "all":
        stages = {"proposed", "baselines", "ablations", "figures", "qualitative"}
        if args.profile == "paper_priority" and args.include_attention:
            stages.add("attention")
        return stages
    return {x.strip() for x in args.stages.split(",") if x.strip()}


def build_execution_plan(args: argparse.Namespace, baseline_names: str, attention_separate: bool, selected_stages: set[str]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    if "proposed" in selected_stages:
        plan.append({"name": "proposed", "units": args.proposed_epochs})
    if "baselines" in selected_stages:
        baseline_count = len([x for x in baseline_names.split(",") if x.strip()]) if baseline_names else 4
        plan.append({"name": "baselines", "units": args.baseline_epochs * baseline_count})
    if "attention" in selected_stages and attention_separate:
        plan.append({"name": "attention", "units": args.baseline_epochs})
    if "ablations" in selected_stages:
        for key in ABLATION_KEYS:
            plan.append({"name": f"ablation_{key}", "units": args.ablation_epochs})
    if "figures" in selected_stages:
        plan.append({"name": "figures_results", "units": 3})
        plan.append({"name": "figures_ablation", "units": 2})
    if "qualitative" in selected_stages:
        plan.append({"name": "qualitative", "units": 3})
    return plan


def estimate_remaining_time(start_time: float, completed_units: int, total_units: int) -> str:
    if completed_units <= 0 or total_units <= completed_units:
        return "estimating..."
    elapsed = max(0.0, time.time() - start_time)
    sec_per_unit = elapsed / completed_units
    remaining = max(0.0, (total_units - completed_units) * sec_per_unit)
    if remaining < 60:
        return f"{remaining:.0f}s"
    if remaining < 3600:
        return f"{remaining / 60:.1f}m"
    return f"{remaining / 3600:.2f}h"


def main() -> None:
    args = parse_args()
    ensure_exists(CNN_SCRIPT, "training script")
    ensure_exists(RESULTS_SCRIPT, "results plotting script")
    ensure_exists(ABLATION_SCRIPT, "ablation plotting script")
    ensure_exists(QUAL_SCRIPT, "qualitative plotting script")

    output_root = Path(args.output_root).resolve()
    runs_dir = output_root / "runs"
    logs_dir = output_root / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = resolve_dataset_path(args.data_dir, args.download_dataset)

    selected_stages = parse_stage_selection(args)
    baseline_names = ""
    attention_separate = False
    if args.profile == "paper_priority":
        baseline_names = "standard,residual,vnet"
        attention_separate = args.include_attention
    execution_plan = build_execution_plan(args, baseline_names, attention_separate, selected_stages)
    total_units = sum(item["units"] for item in execution_plan)
    progress_state = {
        "start_time": time.time(),
        "completed_units": 0,
        "total_units": total_units,
    }
    print("Planned stages:")
    for item in execution_plan:
        print(f"  - {item['name']}: {item['units']} units")
    print(f"Total planned work units: {total_units}")
    progress_bar = None
    if tqdm is not None and total_units > 0:
        progress_bar = tqdm(total=total_units, desc="Full run progress", unit="unit")
        progress_bar.set_postfix_str("eta=estimating...")

    manifest_path = output_root / "run_manifest.json"
    manifest = {
        "created_at": now_iso(),
        "profile": args.profile,
        "data_dir": str(Path(args.data_dir).resolve()),
        "resolved_dataset_dir": str(dataset_path),
        "output_root": str(output_root),
        "amp_enabled": resolve_amp(args.amp_mode),
        "config": {
            "proposed_epochs": args.proposed_epochs,
            "baseline_epochs": args.baseline_epochs,
            "ablation_epochs": args.ablation_epochs,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "base_filters": args.base_filters,
            "bottleneck_channels": args.bottleneck_channels,
            "learning_rate": args.lr,
            "validate_every": args.validate_every,
            "workers": args.workers,
            "prefetch_factor": args.prefetch_factor,
        },
        "manifest_path": str(manifest_path),
        "stages": [],
    }
    write_json(manifest_path, manifest)

    if "proposed" in selected_stages:
        proposed_dir = runs_dir / "proposed_50ep"
        cmd = build_base_command(args, proposed_dir, args.proposed_epochs) + ["--skip_baselines"]
        run_stage("proposed", cmd, logs_dir, manifest, proposed_dir, progress_bar, args.proposed_epochs, progress_state)

    if "baselines" in selected_stages:
        baseline_dir_name = "baselines_priority_50ep" if baseline_names else "baselines_50ep"
        baseline_dir = runs_dir / baseline_dir_name
        cmd = build_base_command(args, baseline_dir, args.baseline_epochs) + ["--baselines_first", "--only_baselines"]
        if baseline_names:
            cmd += ["--baseline_names", baseline_names]
        baseline_count = len([x for x in baseline_names.split(",") if x.strip()]) if baseline_names else 4
        run_stage("baselines", cmd, logs_dir, manifest, baseline_dir, progress_bar, args.baseline_epochs * baseline_count, progress_state)

    if "attention" in selected_stages and attention_separate:
        attention_dir = runs_dir / "baseline_attention_50ep"
        cmd = build_base_command(args, attention_dir, args.baseline_epochs) + [
            "--baselines_first",
            "--only_baselines",
            "--baseline_names",
            "attention",
        ]
        run_stage("attention", cmd, logs_dir, manifest, attention_dir, progress_bar, args.baseline_epochs, progress_state)

    if "ablations" in selected_stages:
        ablation_root = runs_dir / "ablations_25ep"
        ablation_root.mkdir(parents=True, exist_ok=True)
        ablations = [
            ("ablation_no_sepconv", "no_sepconv", "--disable_separable"),
            ("ablation_no_scannorm", "no_scannorm", "--disable_scanner_norm"),
            ("ablation_no_csa", "no_csa", "--disable_csa"),
            ("ablation_no_ssfb", "no_ssfb", "--disable_ssfb"),
        ]
        for stage_name, folder, flag in ablations:
            run_dir = ablation_root / folder
            cmd = build_base_command(args, run_dir, args.ablation_epochs) + ["--skip_baselines", flag]
            run_stage(stage_name, cmd, logs_dir, manifest, run_dir, progress_bar, args.ablation_epochs, progress_state)

    if "figures" in selected_stages:
        canonical = sync_canonical_results(output_root, runs_dir)
        sync_into_repo_results(canonical)
        ensure_figure_inputs(canonical)
        run_stage("figures_results", [sys.executable, str(RESULTS_SCRIPT)], logs_dir, manifest, progress_bar=progress_bar, stage_units=3, progress_state=progress_state)

        ablation_root = runs_dir / "ablations_25ep"
        ablation_ready = (runs_dir / "proposed_50ep" / "training_history.json").exists() and all(
            (ablation_root / key / "training_history.json").exists() for key in ABLATION_KEYS
        )
        if ablation_ready:
            ablation_args = [
                sys.executable,
                str(ABLATION_SCRIPT),
                "--run",
                f"full={runs_dir / 'proposed_50ep'}",
                "--run",
                f"no_sepconv={ablation_root / 'no_sepconv'}",
                "--run",
                f"no_scannorm={ablation_root / 'no_scannorm'}",
                "--run",
                f"no_csa={ablation_root / 'no_csa'}",
                "--run",
                f"no_ssfb={ablation_root / 'no_ssfb'}",
            ]
            run_stage("figures_ablation", ablation_args, logs_dir, manifest, progress_bar=progress_bar, stage_units=2, progress_state=progress_state)
        else:
            print("Skipping ablation figure generation because one or more ablation runs are missing.")
            manifest["stages"].append(
                {
                    "name": "figures_ablation",
                    "started_at": now_iso(),
                    "finished_at": now_iso(),
                    "status": "skipped",
                    "reason": "Missing one or more ablation training histories.",
                }
            )
            write_json(manifest_path, manifest)
            if progress_bar is not None:
                progress_bar.update(2)
                progress_state["completed_units"] += 2
                progress_bar.set_postfix_str(
                    f"eta={estimate_remaining_time(progress_state['start_time'], progress_state['completed_units'], progress_state['total_units'])}"
                )

        figures_out = output_root / "paper_figures"
        figures_out.mkdir(parents=True, exist_ok=True)
        src_fig_root = ROOT / "publication" / "paper" / "figures"
        shutil.copytree(src_fig_root, figures_out, dirs_exist_ok=True)

    if "qualitative" in selected_stages:
        canonical = sync_canonical_results(output_root, runs_dir)
        sync_into_repo_results(canonical)
        ensure_qualitative_inputs(canonical)
        run_stage("qualitative", [sys.executable, str(QUAL_SCRIPT)], logs_dir, manifest, progress_bar=progress_bar, stage_units=3, progress_state=progress_state)
        figures_out = output_root / "paper_figures"
        figures_out.mkdir(parents=True, exist_ok=True)
        src_fig_root = ROOT / "publication" / "paper" / "figures"
        shutil.copytree(src_fig_root, figures_out, dirs_exist_ok=True)

    canonical = sync_canonical_results(output_root, runs_dir)
    summary = {
        "generated_at": now_iso(),
        "canonical_results": str(canonical),
        "proposed": summarize_proposed(runs_dir / "proposed_50ep"),
        "baselines": summarize_baselines(runs_dir / "baselines_50ep") if (runs_dir / "baselines_50ep").exists() else summarize_baselines(runs_dir / "baselines_priority_50ep"),
        "attention": summarize_baselines(runs_dir / "baseline_attention_50ep") if (runs_dir / "baseline_attention_50ep").exists() else None,
        "ablations": {
            key: summarize_proposed(runs_dir / "ablations_25ep" / key)
            for key in ["no_sepconv", "no_scannorm", "no_csa", "no_ssfb"]
            if (runs_dir / "ablations_25ep" / key).exists()
        },
    }
    write_json(output_root / "experiment_summary.json", summary)
    if progress_bar is not None:
        progress_bar.close()
    print(f"\nAll requested stages finished. Results saved under: {output_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {output_root / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
