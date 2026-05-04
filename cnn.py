"""
Domain-Adaptive Lightweight 3D CNN for Brain Tumor Segmentation
Single-file implementation for Google Colab with automatic BraTS download

Usage on Colab:
    # Cell 1: Upload this file or clone repo
    # Cell 2: Run everything
    !python colab_full_training.py --download_brats --epochs 100
"""

import os
import argparse
import random
import subprocess
import sys
import json
import tarfile
import urllib.request
import importlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, List

# ===================== THIRD-PARTY IMPORTS =====================
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import GradScaler
    from torch.amp import autocast as _autocast
    # Wrapper: use bfloat16 if GPU supports it (H100/A100), else float16
    def _get_amp_dtype():
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    _amp_dtype = None  # resolved lazily at first use
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from scipy.ndimage import distance_transform_edt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError as e:
    print("\nERROR: Missing Python dependency:", e)
    print(
        "\nPlease install the required packages, for example:\n"
        "  pip install numpy torch torchvision nibabel SimpleITK tensorboard tqdm scipy matplotlib\n"
    )
    sys.exit(1)

# ===================== PATH HELPERS =====================
def resolve_brats_path(data_dir: str, download_if_missing: bool) -> Optional[str]:
    """
    Resolve BraTS (Medical Decathlon Task01_BrainTumour) folder path.
    Accepts either:
      - a root dir that contains Task01_BrainTumour/
      - the Task01_BrainTumour/ dir itself (with imagesTr/labelsTr)
    Optionally downloads if missing.
    """
    data_root = Path(data_dir).expanduser().resolve()

    # User points directly at Task01_BrainTumour
    if (data_root / "imagesTr").exists() and (data_root / "labelsTr").exists():
        return str(data_root)

    # User points at a root that contains Task01_BrainTumour
    candidate = data_root / "Task01_BrainTumour"
    if candidate.exists():
        return str(candidate)

    if download_if_missing:
        downloaded = download_brats_dataset(str(data_root))
        return downloaded

    return None

def print_results_summary(output_dir: str):
    out = Path(output_dir).resolve()
    print("\n" + "=" * 70)
    print("WHERE TO SEE RESULTS")
    print("=" * 70)
    print(f"Output directory: {out}")

    key_files = [
        out / "training_summary.txt",
        out / "training_history.json",
        out / "best_model.pth",
        out / "final_model.pth",
        out / "comparison_results.json",
    ]

    figures_dir = out / "figures"
    comparison_figs_dir = out / "comparison_figures"
    logs_dir = out / "logs"

    existing = [p for p in key_files if p.exists()]
    if existing:
        print("\nKey files:")
        for p in existing:
            print(f"  - {p}")

    if figures_dir.exists():
        pngs = sorted(figures_dir.glob("*.png"))
        print(f"\nFigures: {figures_dir} ({len(pngs)} png)")
        for p in pngs[:12]:
            print(f"  - {p.name}")
        if len(pngs) > 12:
            print(f"  - ... and {len(pngs) - 12} more")

    if comparison_figs_dir.exists():
        pngs = sorted(comparison_figs_dir.glob("*.png"))
        print(f"\nComparison figures: {comparison_figs_dir} ({len(pngs)} png)")
        for p in pngs[:12]:
            print(f"  - {p.name}")
        if len(pngs) > 12:
            print(f"  - ... and {len(pngs) - 12} more")

    if logs_dir.exists():
        print(f"\nTensorBoard logs: {logs_dir}")
        print("To view:")
        print(f"  tensorboard --logdir \"{logs_dir}\"")

    print("=" * 70 + "\n")

# ===================== DOWNLOAD BRATS =====================
def _download_file_with_progress(url: str, dst_path: str, chunk_mb: int = 8) -> None:
    """Download a large file with a simple text progress bar."""
    print(f"  → Downloading from: {url}")
    print(f"  → Saving to: {dst_path}")
    chunk_size = chunk_mb * 1024 * 1024

    with urllib.request.urlopen(url) as resp, open(dst_path, "wb") as f:
        total = resp.getheader("Content-Length")
        total = int(total) if total is not None else None
        downloaded = 0
        last_percent = -1

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            if total:
                percent = int(downloaded * 100 / total)
                if percent != last_percent:
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    print(f"\r  Downloaded {mb:7.1f}/{total_mb:.1f} MB ({percent:3d}%)", end="", flush=True)
                    last_percent = percent
            else:
                mb = downloaded / (1024 * 1024)
                print(f"\r  Downloaded {mb:7.1f} MB", end="", flush=True)

    print()  # newline after progress bar


def download_brats_dataset(data_dir='/content/data'):
    """
    Download BraTS (Task01_BrainTumour) with explicit progress.

    This uses a direct HTTP download (~4 GB) and shows a simple
    progress bar in the terminal, then extracts the archive.
    """
    print("=" * 60)
    print("Downloading Brain Tumor Dataset (Medical Decathlon Task01)")
    print("=" * 60)
    print("This is ~4 GB and may take a while; progress is shown below.\n")

    os.makedirs(data_dir, exist_ok=True)

    data_path = os.path.join(data_dir, "Task01_BrainTumour")
    if os.path.isdir(data_path):
        print(f"Dataset already present at: {data_path}")
        return data_path

    url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
    tar_path = os.path.join(data_dir, "Task01_BrainTumour.tar")

    try:
        _download_file_with_progress(url, tar_path)

        print("\nExtracting archive (this may also take a few minutes)...")
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            total_members = len(members)
            for i, member in enumerate(members, start=1):
                tar.extract(member, path=data_dir)
                if i % 100 == 0 or i == total_members:
                    print(f"\r  Extracted {i}/{total_members} files/directories", end="", flush=True)
        print()

        os.remove(tar_path)

        if os.path.isdir(data_path):
            print(f"\nDataset extracted to: {data_path}")
            return data_path
        else:
            print("\nExtraction finished, but target folder not found. Falling back to synthetic data.")
            return None

    except Exception as e:
        print(f"\nDownload or extraction failed: {e}")
        print("Using synthetic data instead...")
        return None


# ===================== CONFIG =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===================== LAYERS =====================

class Separable3DConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv3d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv3d(in_ch, out_ch, 1, bias=bias)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class CrossSliceAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.query = nn.Conv1d(channels, mid, 1)
        self.key = nn.Conv1d(channels, mid, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.scale = mid ** -0.5
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        pooled = x.mean(dim=[-2, -1])
        q, k, v = self.query(pooled), self.key(pooled), self.value(pooled)
        # Force float32 for attention to avoid overflow under AMP
        attn = F.softmax((torch.bmm(q.transpose(1, 2), k) * self.scale).float(), dim=-1).to(q.dtype)
        out = torch.bmm(v, attn).unsqueeze(-1).unsqueeze(-1)
        return x + out.expand_as(x)


class ScannerAwareNorm(nn.Module):
    def __init__(self, num_channels, num_groups=8, num_scanners=8):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.scanner_gamma = nn.Embedding(num_scanners, num_channels)
        self.scanner_beta = nn.Embedding(num_scanners, num_channels)
        self.default_gamma = nn.Parameter(torch.ones(num_channels))
        self.default_beta = nn.Parameter(torch.zeros(num_channels))
        nn.init.ones_(self.scanner_gamma.weight)
        nn.init.zeros_(self.scanner_beta.weight)
    
    def forward(self, x, scanner_id=None):
        x = self.group_norm(x)
        if scanner_id is not None:
            gamma, beta = self.scanner_gamma(scanner_id), self.scanner_beta(scanner_id)
        else:
            gamma, beta = self.default_gamma.unsqueeze(0), self.default_beta.unsqueeze(0)
        while gamma.dim() < x.dim():
            gamma, beta = gamma.unsqueeze(-1), beta.unsqueeze(-1)
        return x * gamma + beta


class SE3D(nn.Module):
    """Squeeze-and-Excitation for 3D"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        pooled = self.pool(x).view(B, C)
        scale = torch.sigmoid(self.fc(pooled.float())).to(x.dtype).view(B, C, 1, 1, 1)
        return x * scale


class LightweightBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_separable=True, use_scanner_norm=True, use_attention=True, stride=1):
        super().__init__()
        self.use_attention = use_attention
        Conv = Separable3DConv if use_separable else nn.Conv3d
        self.conv1 = Conv(in_ch, out_ch, 3, stride, 1, bias=False)
        self.conv2 = Conv(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm1 = ScannerAwareNorm(out_ch) if use_scanner_norm else nn.GroupNorm(8, out_ch)
        self.norm2 = ScannerAwareNorm(out_ch) if use_scanner_norm else nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()
        self.se = SE3D(out_ch)
        if use_attention:
            self.attention = CrossSliceAttention(out_ch)
        self.skip = nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, stride, bias=False), nn.GroupNorm(8, out_ch)) \
            if in_ch != out_ch or stride != 1 else nn.Identity()
    
    def forward(self, x, scanner_id=None):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out, scanner_id) if isinstance(self.norm1, ScannerAwareNorm) else self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out, scanner_id) if isinstance(self.norm2, ScannerAwareNorm) else self.norm2(out)
        out = self.se(out)
        if self.use_attention:
            out = self.attention(out)
        return self.act(out + identity)


# ===================== SSFB =====================

class LowRankAttention(nn.Module):
    def __init__(self, channels, rank=8):
        super().__init__()
        self.query_proj = nn.Conv3d(channels, rank, 1)
        self.key_proj = nn.Conv3d(channels, rank, 1)
        self.value_proj = nn.Conv3d(channels, channels, 1)
        self.out_proj = nn.Conv3d(channels, channels, 1)
        self.scale = rank ** -0.5
        
    def forward(self, dec_feat, enc_feat):
        B, C, D, H, W = dec_feat.shape
        q = self.query_proj(dec_feat).flatten(2)
        k = self.key_proj(enc_feat).flatten(2)
        v = self.value_proj(enc_feat).flatten(2)
        # Force float32 for softmax to avoid overflow under AMP
        q = F.softmax((q * self.scale).float(), dim=-1).to(v.dtype)
        k = F.softmax((k * self.scale).float(), dim=1).to(v.dtype)
        context = torch.bmm(k, v.transpose(1, 2))
        out = torch.bmm(q.transpose(1, 2), context).transpose(1, 2).view(B, C, D, H, W)
        return self.out_proj(out)


class ChannelGating(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(nn.Linear(channels * 2, mid), nn.GELU(), nn.Linear(mid, channels), nn.Sigmoid())
    
    def forward(self, dec_feat, enc_feat):
        B, C = dec_feat.shape[:2]
        dec_pool = F.adaptive_avg_pool3d(dec_feat, 1).view(B, -1).float()
        enc_pool = F.adaptive_avg_pool3d(enc_feat, 1).view(B, -1).float()
        gate = self.fc(torch.cat([dec_pool, enc_pool], dim=1)).to(enc_feat.dtype).view(B, C, 1, 1, 1)
        return enc_feat * gate


class SSFB(nn.Module):
    def __init__(self, enc_ch, dec_ch, out_ch, rank=8):
        super().__init__()
        self.align_enc = nn.Conv3d(enc_ch, out_ch, 1) if enc_ch != out_ch else nn.Identity()
        self.align_dec = nn.Conv3d(dec_ch, out_ch, 1) if dec_ch != out_ch else nn.Identity()
        self.low_rank_attn = LowRankAttention(out_ch, rank)
        self.channel_gate = ChannelGating(out_ch)
        self.fusion = nn.Sequential(nn.Conv3d(out_ch * 2, out_ch, 3, padding=1, bias=False), nn.GroupNorm(8, out_ch), nn.GELU())
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, enc_feat, dec_feat):
        enc, dec = self.align_enc(enc_feat), self.align_dec(dec_feat)
        attn_out = self.low_rank_attn(dec, enc)
        gated_out = self.channel_gate(dec, enc)
        alpha = torch.sigmoid(self.alpha)
        combined = alpha * attn_out + (1 - alpha) * gated_out
        return self.fusion(torch.cat([dec, combined], dim=1))


class SimpleSkip(nn.Module):
    def __init__(self, enc_ch, dec_ch, out_ch, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(enc_ch + dec_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(8, out_ch), nn.GELU())
    
    def forward(self, enc_feat, dec_feat):
        return self.conv(torch.cat([enc_feat, dec_feat], dim=1))


# ===================== HEADS =====================

class EvidentialHead(nn.Module):
    def __init__(self, in_ch, num_classes, hidden=64):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hidden, 3, padding=1, bias=False), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv3d(hidden, num_classes, 1), nn.Softplus()
        )
        
    def forward(self, x):
        evidence = self.net(x).float()
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        probs = alpha / S
        epistemic = self.num_classes / S.squeeze(1)
        aleatoric = -(probs * torch.log(probs.clamp(min=1e-7))).sum(dim=1)
        return {'probs': probs, 'alpha': alpha, 'evidence': evidence, 'epistemic': epistemic, 
                'aleatoric': aleatoric, 'uncertainty': epistemic + aleatoric}


class SoftmaxHead(nn.Module):
    def __init__(self, in_ch, num_classes, hidden=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv3d(in_ch, hidden, 3, padding=1, bias=False), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv3d(hidden, num_classes, 1)
        )
    
    def forward(self, x):
        logits = self.head(x)
        # Force float32 for softmax/log to avoid NaN under AMP
        logits_fp32 = logits.float()
        probs = F.softmax(logits_fp32, dim=1)
        uncertainty = -(probs * torch.log(probs.clamp(min=1e-7))).sum(dim=1)
        return {'probs': probs, 'logits': logits_fp32, 'uncertainty': uncertainty, 'epistemic': uncertainty, 'aleatoric': torch.zeros_like(uncertainty)}


# ===================== NETWORK =====================

class Encoder(nn.Module):
    def __init__(self, in_ch, base_filters, num_stages=4, use_separable=True, use_scanner_norm=True, use_csa=True, max_channels=384, bottleneck_channels=None):
        super().__init__()
        self.init_conv = nn.Sequential(nn.Conv3d(in_ch, base_filters, 3, padding=1, bias=False), nn.GroupNorm(8, base_filters), nn.GELU())
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_filters
        for i in range(num_stages):
            if i == num_stages - 1 and bottleneck_channels is not None:
                out_ch = bottleneck_channels
            else:
                out_ch = min(ch * 2, max_channels) if i > 0 else ch
            # Architecturally: use full 3D convs in the first encoder stage for richer low-level features,
            # then switch to separable convs in deeper stages to stay lightweight.
            stage_use_separable = use_separable if i > 0 else False
            self.stages.append(LightweightBlock(ch, out_ch, stage_use_separable, use_scanner_norm, use_attention=(use_csa and i >= 2)))
            if i < num_stages - 1:
                self.downsamples.append(nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1))
            ch = out_ch
        self.out_channels = ch
    
    def forward(self, x, scanner_id=None):
        features = []
        x = self.init_conv(x)
        for i, stage in enumerate(self.stages):
            x = stage(x, scanner_id)
            features.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        return features


class Decoder(nn.Module):
    def __init__(self, enc_channels, base_filters, use_ssfb=True, use_separable=True, use_scanner_norm=True, ssfb_rank=16):
        super().__init__()
        enc_ch = enc_channels[::-1]
        self.upsamples = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()
        self.stages = nn.ModuleList()
        SkipModule = SSFB if use_ssfb else SimpleSkip
        in_ch = enc_ch[0]
        for i, ec in enumerate(enc_ch[1:]):
            out_ch = ec
            self.upsamples.append(nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2))
            # Architecturally: use a simpler skip fusion (SimpleSkip) at the shallowest decoder level
            # to make optimization easier, and SSFB for deeper skips.
            if use_ssfb and i == 0:
                skip = SimpleSkip(ec, out_ch, out_ch)
            else:
                skip = SkipModule(ec, out_ch, out_ch, rank=ssfb_rank)
            self.skip_fusions.append(skip)
            self.stages.append(LightweightBlock(out_ch, out_ch, use_separable, use_scanner_norm, use_attention=False))
            in_ch = out_ch
        self.out_channels = in_ch
    
    def forward(self, enc_features, scanner_id=None):
        features = enc_features[::-1]
        x = features[0]
        for i, (up, skip_fusion, stage) in enumerate(zip(self.upsamples, self.skip_fusions, self.stages)):
            x = up(x)
            enc_feat = features[i + 1]
            if x.shape[2:] != enc_feat.shape[2:]:
                x = F.interpolate(x, size=enc_feat.shape[2:], mode='trilinear', align_corners=False)
            x = skip_fusion(enc_feat, x)
            x = stage(x, scanner_id)
        return x


class LightweightUNet3D(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, base_filters=32, num_stages=4,
                 use_separable=True, use_ssfb=True, use_uncertainty=True, use_scanner_norm=True, use_csa=True, dropout=0.1,
                 max_channels=384, ssfb_rank=16, bottleneck_channels=None):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.encoder = Encoder(in_ch, base_filters, num_stages, use_separable, use_scanner_norm, use_csa, max_channels=max_channels, bottleneck_channels=bottleneck_channels)
        enc_channels = [base_filters]
        ch = base_filters
        for i in range(1, num_stages):
            if i == num_stages - 1 and bottleneck_channels is not None:
                ch = bottleneck_channels
            else:
                ch = min(ch * 2, max_channels)
            enc_channels.append(ch)
        self.decoder = Decoder(enc_channels, base_filters, use_ssfb, use_separable, use_scanner_norm, ssfb_rank=ssfb_rank)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.head = EvidentialHead(self.decoder.out_channels, out_ch) if use_uncertainty else SoftmaxHead(self.decoder.out_channels, out_ch)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
    
    def forward(self, x, scanner_id=None):
        enc_features = self.encoder(x, scanner_id)
        dec_out = self.decoder(enc_features, scanner_id)
        dec_out = self.dropout(dec_out)
        return self.head(dec_out)
    
    def get_prediction(self, outputs):
        return outputs['probs'].argmax(dim=1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===================== BASELINES =====================

class Standard3DUNet(nn.Module):
    """Standard 3D U-Net baseline"""
    def __init__(self, in_ch=4, out_ch=4, base_ch=32):
        super().__init__()
        self.enc1 = self._block(in_ch, base_ch)
        self.enc2 = self._block(base_ch, base_ch * 2)
        self.enc3 = self._block(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = self._block(base_ch * 4, base_ch * 8)
        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = self._block(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = self._block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = self._block(base_ch * 2, base_ch)
        self.final = nn.Conv3d(base_ch, out_ch, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, scanner_id=None):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.final(d1).float()
        return {'probs': F.softmax(logits, dim=1), 'logits': logits}

    def get_prediction(self, outputs):
        return outputs['probs'].argmax(dim=1)


class AttentionGate3D(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.g = nn.Conv3d(g_ch, inter_ch, 1, bias=False)
        self.x = nn.Conv3d(x_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv3d(inter_ch, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, g, x):
        attn = self.psi(F.relu(self.g(g) + self.x(x)))
        return x * attn


class AttentionUNet3D(nn.Module):
    """Attention U-Net baseline"""
    def __init__(self, in_ch=4, out_ch=4, base_ch=32):
        super().__init__()
        self.enc1 = Standard3DUNet._block(self, in_ch, base_ch)
        self.enc2 = Standard3DUNet._block(self, base_ch, base_ch * 2)
        self.enc3 = Standard3DUNet._block(self, base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = Standard3DUNet._block(self, base_ch * 4, base_ch * 8)
        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.att3 = AttentionGate3D(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec3 = Standard3DUNet._block(self, base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.att2 = AttentionGate3D(base_ch * 2, base_ch * 2, base_ch)
        self.dec2 = Standard3DUNet._block(self, base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2)
        self.att1 = AttentionGate3D(base_ch, base_ch, base_ch // 2)
        self.dec1 = Standard3DUNet._block(self, base_ch * 2, base_ch)
        self.final = nn.Conv3d(base_ch, out_ch, 1)

    def forward(self, x, scanner_id=None):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        logits = self.final(d1).float()
        return {'probs': F.softmax(logits, dim=1), 'logits': logits}

    def get_prediction(self, outputs):
        return outputs['probs'].argmax(dim=1)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class ResUNet3D(nn.Module):
    """Residual U-Net baseline"""
    def __init__(self, in_ch=4, out_ch=4, base_ch=32):
        super().__init__()
        self.enc1 = ResBlock3D(in_ch, base_ch)
        self.enc2 = ResBlock3D(base_ch, base_ch * 2)
        self.enc3 = ResBlock3D(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ResBlock3D(base_ch * 4, base_ch * 8)
        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ResBlock3D(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ResBlock3D(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ResBlock3D(base_ch * 2, base_ch)
        self.final = nn.Conv3d(base_ch, out_ch, 1)

    def forward(self, x, scanner_id=None):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.final(d1).float()
        return {'probs': F.softmax(logits, dim=1), 'logits': logits}

    def get_prediction(self, outputs):
        return outputs['probs'].argmax(dim=1)


class VNetBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, num_convs=2):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(num_convs):
            layers.extend([
                nn.Conv3d(ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(8, out_ch),
                nn.PReLU(out_ch),
            ])
            ch = out_ch
        self.block = nn.Sequential(*layers)
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        return self.act(self.block(x) + self.skip(x))


class VNet3D(nn.Module):
    """Compact V-Net style volumetric baseline"""
    def __init__(self, in_ch=4, out_ch=4, base_ch=24):
        super().__init__()
        self.enc1 = VNetBlock3D(in_ch, base_ch, num_convs=1)
        self.down1 = nn.Conv3d(base_ch, base_ch * 2, 2, stride=2, bias=False)
        self.enc2 = VNetBlock3D(base_ch * 2, base_ch * 2, num_convs=2)
        self.down2 = nn.Conv3d(base_ch * 2, base_ch * 4, 2, stride=2, bias=False)
        self.enc3 = VNetBlock3D(base_ch * 4, base_ch * 4, num_convs=2)
        self.down3 = nn.Conv3d(base_ch * 4, base_ch * 8, 2, stride=2, bias=False)
        self.bottleneck = VNetBlock3D(base_ch * 8, base_ch * 8, num_convs=3)

        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, 2, stride=2, bias=False)
        self.dec3 = VNetBlock3D(base_ch * 8, base_ch * 4, num_convs=2)
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2, bias=False)
        self.dec2 = VNetBlock3D(base_ch * 4, base_ch * 2, num_convs=2)
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2, bias=False)
        self.dec1 = VNetBlock3D(base_ch * 2, base_ch, num_convs=1)
        self.final = nn.Conv3d(base_ch, out_ch, 1)

    def forward(self, x, scanner_id=None):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.final(d1).float()
        return {'probs': F.softmax(logits, dim=1), 'logits': logits}

    def get_prediction(self, outputs):
        return outputs['probs'].argmax(dim=1)

# ===================== DATASETS =====================

class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, img, seg):
        for t in self.transforms: img, seg = t(img, seg)
        return img, seg

class RandomFlip:
    def __call__(self, img, seg):
        for axis in [1, 2, 3]:
            if np.random.rand() > 0.5:
                img, seg = torch.flip(img, [axis]), torch.flip(seg, [axis - 1])
        return img, seg

class RandomIntensity:
    def __call__(self, img, seg):
        return img * (0.9 + 0.2 * np.random.rand()) + (np.random.rand() - 0.5) * 0.2, seg

class RandomNoise:
    def __call__(self, img, seg):
        return img + torch.randn_like(img) * 0.05, seg

def get_transforms(split):
    return Compose([RandomFlip(), RandomIntensity(), RandomNoise()]) if split == 'train' else None


class SyntheticBraTSDataset(Dataset):
    def __init__(self, num_samples=100, patch_size=(96, 96, 96), num_classes=4, in_ch=4, transform=None):
        self.num_samples, self.patch_size, self.num_classes, self.in_ch = num_samples, patch_size, num_classes, in_ch
        self.transform = transform
        self.seeds = np.random.randint(0, 100000, size=num_samples)
    
    def __len__(self): return self.num_samples
    
    def __getitem__(self, idx):
        np.random.seed(self.seeds[idx])
        D, H, W = self.patch_size
        image = np.random.randn(self.in_ch, D, H, W).astype(np.float32) * 0.3
        seg = np.zeros((D, H, W), dtype=np.int64)
        center = [np.random.randint(D//4, 3*D//4), np.random.randint(H//4, 3*H//4), np.random.randint(W//4, 3*W//4)]
        coords = np.mgrid[:D, :H, :W]
        for cls in range(self.num_classes - 1, 0, -1):
            radius = [np.random.randint(5, 15) * cls for _ in range(3)]
            dist = sum(((coords[i] - center[i]) / radius[i]) ** 2 for i in range(3))
            mask = dist < 1
            seg[mask] = cls
            for c in range(self.in_ch): image[c][mask] += np.random.randn() * 0.3
        image = (image - image.mean()) / (image.std() + 1e-8)
        image, seg = torch.from_numpy(image), torch.from_numpy(seg)
        if self.transform: image, seg = self.transform(image, seg)
        return {'image': image, 'seg': seg, 'case_id': f'syn_{idx}', 'scanner_id': torch.tensor(idx % 8)}


class DecathlonBraTSDataset(Dataset):
    """Dataset for Medical Decathlon Task01_BrainTumour"""
    
    def __init__(self, data_dir, split='train', patch_size=(96, 96, 96), transform=None, max_cases=None):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.transform = transform
        
        # Find images and labels
        images_dir = self.data_dir / 'imagesTr'
        labels_dir = self.data_dir / 'labelsTr'
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        
        # Get all cases (skip hidden/corrupt files)
        self.cases = []
        for img_file in sorted(images_dir.glob('*.nii.gz')):
            # Skip macOS hidden files and corrupt files
            if img_file.name.startswith('._') or img_file.name.startswith('.'):
                continue
            case_id = img_file.name.replace('.nii.gz', '')
            label_file = labels_dir / img_file.name
            if label_file.exists() and not label_file.name.startswith('._'):
                self.cases.append({
                    'image': str(img_file),
                    'label': str(label_file),
                    'id': case_id
                })
        
        if max_cases:
            self.cases = self.cases[:max_cases]
        
        # Split 80/20
        split_idx = int(len(self.cases) * 0.8)
        if split == 'train':
            self.cases = self.cases[:split_idx]
        else:
            self.cases = self.cases[split_idx:]
        
        print(f"DecathlonBraTS {split}: {len(self.cases)} cases")
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        
        try:
            import nibabel as nib
            # Load 4D image (4 modalities stacked)
            img_nii = nib.load(case['image'])
            image = img_nii.get_fdata().astype(np.float32)
            
            # Handle different image formats
            if image.ndim == 3:
                # Single modality - replicate to 4 channels
                image = np.stack([image] * 4, axis=-1)
            
            # Transpose to (C, D, H, W)
            if image.ndim == 4:
                image = np.transpose(image, (3, 0, 1, 2))
            
            # Load label
            label_nii = nib.load(case['label'])
            seg = label_nii.get_fdata().astype(np.int64)
            
            # Clip labels to valid range
            seg = np.clip(seg, 0, 3)
            
            # Normalize each channel
            for i in range(image.shape[0]):
                mask = image[i] > 0
                if mask.sum() > 0:
                    mean_val = image[i][mask].mean()
                    std_val = image[i][mask].std()
                    if std_val > 1e-8:
                        image[i][mask] = (image[i][mask] - mean_val) / std_val
                    else:
                        image[i][mask] = 0
            
            # Replace NaN/Inf with 0
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Extract random patch
            image, seg = self._extract_patch(image, seg)
            
            image = torch.from_numpy(image.copy())
            seg = torch.from_numpy(seg.copy())
            
            if self.transform:
                image, seg = self.transform(image, seg)
            
            scanner_id = hash(case['id']) % 8
            
            return {
                'image': image,
                'seg': seg,
                'case_id': case['id'],
                'scanner_id': torch.tensor(scanner_id)
            }
        except Exception as e:
            print(f"Error loading {case['id']}: {e}")
            raise e
    
    def _extract_patch(self, image, seg):
        C, D, H, W = image.shape
        pd, ph, pw = self.patch_size
        
        # Try to find a patch with tumor (up to 10 attempts)
        for _ in range(10):
            d = np.random.randint(0, max(1, D - pd)) if D > pd else 0
            h = np.random.randint(0, max(1, H - ph)) if H > ph else 0
            w = np.random.randint(0, max(1, W - pw)) if W > pw else 0
            
            seg_patch = seg[d:d+pd, h:h+ph, w:w+pw]
            
            # Check if patch has tumor (non-zero labels)
            if seg_patch.sum() > 100:  # At least 100 tumor voxels
                break
        
        img_patch = image[:, d:d+pd, h:h+ph, w:w+pw]
        
        # Pad if needed
        if img_patch.shape[1:] != self.patch_size:
            pad_d = pd - img_patch.shape[1]
            pad_h = ph - img_patch.shape[2]
            pad_w = pw - img_patch.shape[3]
            img_patch = np.pad(img_patch, [(0, 0), (0, pad_d), (0, pad_h), (0, pad_w)])
            seg_patch = np.pad(seg_patch, [(0, pad_d), (0, pad_h), (0, pad_w)])
        
        return img_patch, seg_patch
    


# ===================== LOSSES =====================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target.long().clamp(0, num_classes-1), num_classes).permute(0, 4, 1, 2, 3).float()
        dice_scores = []
        for c in range(1, num_classes):
            pred_c, target_c = pred[:, c].flatten(1), target_onehot[:, c].flatten(1)
            intersection = (pred_c * target_c).sum(1)
            union = pred_c.sum(1) + target_c.sum(1)
            dice = (2 * intersection + self.smooth) / (union + self.smooth + 1e-8)
            dice_scores.append(dice)
        if len(dice_scores) == 0:
            return torch.tensor(0.5, device=pred.device)
        return 1 - torch.stack(dice_scores, dim=1).mean().clamp(0, 1)


class EvidentialLoss(nn.Module):
    def __init__(self, num_classes, annealing_epochs=50, lambda_kl=0.1):
        super().__init__()
        self.num_classes, self.annealing_epochs, self.lambda_kl = num_classes, annealing_epochs, lambda_kl
    
    def forward(self, outputs, target, epoch=0):
        alpha = outputs['alpha']
        y = F.one_hot(target.long(), self.num_classes).permute(0, 4, 1, 2, 3).float()
        S = alpha.sum(dim=1, keepdim=True)
        nll = (y * (torch.log(S) - torch.log(alpha))).sum(dim=1).mean()
        alpha_tilde = y + (1 - y) * alpha
        S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        kl = (torch.lgamma(S_tilde.squeeze(1)) - torch.lgamma(alpha_tilde).sum(dim=1)).mean()
        annealing = min(1.0, epoch / self.annealing_epochs)
        return nll + self.lambda_kl * annealing * kl, {'nll': nll.item(), 'kl': kl.item()}


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=4, dice_w=1.0, ce_w=0.5, evid_w=0.1, use_evidential=True):
        super().__init__()
        self.dice_w, self.ce_w, self.evid_w = dice_w, ce_w, evid_w
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.evidential_loss = EvidentialLoss(num_classes) if use_evidential else None
    
    def forward(self, outputs, target, epoch=0):
        probs = outputs['probs']
        
        # Clamp target to valid range
        target = target.long().clamp(0, probs.shape[1] - 1)
        
        dice = self.dice_loss(probs, target)
        
        if 'logits' in outputs:
            ce = self.ce_loss(outputs['logits'], target)
        else:
            ce = F.nll_loss(torch.log(probs.clamp(min=1e-7)), target)
        
        # Clamp losses to prevent explosion
        dice = dice.clamp(0, 2)
        ce = ce.clamp(0, 10)
        
        total = self.dice_w * dice + self.ce_w * ce
        loss_dict = {'dice': dice.item(), 'ce': ce.item()}
        
        if self.evidential_loss and 'alpha' in outputs:
            evid, evid_dict = self.evidential_loss(outputs, target, epoch)
            total += self.evid_w * evid
            loss_dict.update(evid_dict)
        
        loss_dict['total'] = total.item()
        return total, loss_dict


# ===================== METRICS =====================

def compute_dice(pred, target, num_classes=4):
    pred, target = pred.cpu().numpy(), target.cpu().numpy()
    scores = {}
    names = ['BG', 'NCR', 'ED', 'ET']
    for c in range(1, num_classes):
        pred_c, target_c = (pred == c).astype(float), (target == c).astype(float)
        inter, union = (pred_c * target_c).sum(), pred_c.sum() + target_c.sum()
        scores[f'dice_{names[c]}'] = 2 * inter / union if union > 0 else 1.0
    scores['dice_mean'] = np.mean([v for k, v in scores.items()])
    return scores


def compute_ece(probs, target, n_bins=15):
    probs, target = probs.cpu().numpy(), target.cpu().numpy()
    confidence, pred_class = probs.max(axis=1).flatten(), probs.argmax(axis=1).flatten()
    correct = (pred_class == target.flatten()).astype(float)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.sum() > 0:
            ece += np.abs(correct[in_bin].mean() - confidence[in_bin].mean()) * in_bin.mean()
    return ece


# ===================== PUBLICATION METRICS & VISUALIZATION =====================

def evaluate_publication_metrics(model, val_loader, device, output_dir, num_classes=4):
    """
    Compute confusion matrix and per-class metrics over the full validation set
    and generate publication-ready figures + JSON.
    """
    names = ['BG', 'NCR', 'ED', 'ET']
    publication_dir = os.path.join(output_dir, 'publication')
    figures_dir = os.path.join(publication_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    model.eval()
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    dice_accumulator = {f'dice_{n}': [] for n in names[1:]}  # skip BG for dice mean

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Publication eval', leave=False):
            image = _to_device(batch['image'], device, non_blocking=True)
            target = _to_device(batch['seg'], device, non_blocking=True)
            scanner_id = batch.get('scanner_id')
            if scanner_id is not None:
                scanner_id = _to_device(scanner_id, device, non_blocking=True)
                outputs = model(image, scanner_id)
            else:
                outputs = model(image)

            pred = model.get_prediction(outputs)

            # Confusion matrix over all voxels
            true_flat = target.view(-1).cpu().numpy()
            pred_flat = pred.view(-1).cpu().numpy()
            mask = (true_flat >= 0) & (true_flat < num_classes)
            true_flat = true_flat[mask]
            pred_flat = pred_flat[mask]
            conf_mat += np.bincount(
                true_flat * num_classes + pred_flat,
                minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

            # Per-class Dice for tumor classes
            dice_scores = compute_dice(pred, target, num_classes=num_classes)
            for k, v in dice_scores.items():
                if k.startswith('dice_') and k != 'dice_mean':
                    dice_accumulator[k].append(float(v))

    # Per-class metrics from confusion matrix
    per_class_metrics = {}
    total = conf_mat.sum()
    for c in range(num_classes):
        tp = int(conf_mat[c, c])
        fp = int(conf_mat[:, c].sum() - tp)
        fn = int(conf_mat[c, :].sum() - tp)
        tn = int(total - tp - fp - fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        name = names[c] if c < len(names) else f'class_{c}'
        per_class_metrics[name] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'iou': float(iou),
        }

    # Add averaged Dice from accumulator
    for name, values in dice_accumulator.items():
        class_name = name.replace('dice_', '')
        if class_name in per_class_metrics:
            per_class_metrics[class_name]['dice'] = float(np.mean(values)) if values else 0.0

    # Overall metrics (macro averages over tumor classes only)
    tumor_classes = [n for n in names[1:] if n in per_class_metrics]
    def macro_avg(key):
        vals = [per_class_metrics[c][key] for c in tumor_classes]
        return float(np.mean(vals)) if vals else 0.0

    overall_metrics = {
        'macro_precision': macro_avg('precision'),
        'macro_recall': macro_avg('recall'),
        'macro_specificity': macro_avg('specificity'),
        'macro_iou': macro_avg('iou'),
        'macro_dice': macro_avg('dice'),
    }

    # Save confusion matrix figure
    plt.figure(figsize=(6, 5))
    im = plt.imshow(conf_mat, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, names[:num_classes], rotation=45)
    plt.yticks(tick_marks, names[:num_classes])

    thresh = conf_mat.max() / 2.0 if conf_mat.max() > 0 else 0
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j, i, format(conf_mat[i, j], 'd'),
                ha='center', va='center',
                color='white' if conf_mat[i, j] > thresh else 'black',
                fontsize=8,
            )
    plt.ylabel('True label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Per-class bar chart for Dice / IoU
    tumor_metrics = [per_class_metrics[c] for c in tumor_classes]
    x = np.arange(len(tumor_classes))
    width = 0.35
    plt.figure(figsize=(8, 5))
    dice_vals = [m.get('dice', 0.0) for m in tumor_metrics]
    iou_vals = [m.get('iou', 0.0) for m in tumor_metrics]
    plt.bar(x - width / 2, dice_vals, width, label='Dice')
    plt.bar(x + width / 2, iou_vals, width, label='IoU')
    plt.xticks(x, tumor_classes)
    plt.ylim(0, 1.0)
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Per-class Dice and IoU', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'per_class_dice_iou.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics JSON
    metrics_json = {
        'confusion_matrix': conf_mat.tolist(),
        'per_class_metrics': per_class_metrics,
        'overall_metrics': overall_metrics,
    }
    with open(os.path.join(publication_dir, 'publication_metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)

    return metrics_json


# ===================== VISUALIZATION =====================

def generate_visualizations(history, output_dir):
    """Generate research-quality plots and tables"""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    epochs = np.array(history['epochs'])
    train_losses = np.array(history['train_losses'])
    val_dices = np.array([d for d in history['val_dices'] if d is not None])
    val_eces = np.array([e for e in history['val_eces'] if e is not None])
    val_epochs = np.array([e for i, e in enumerate(history['epochs']) if history['val_dices'][i] is not None])
    
    # 1. Training Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs + 1, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Validation Dice Score
    plt.figure(figsize=(10, 6))
    plt.plot(val_epochs + 1, val_dices, 'g-', linewidth=2, marker='o', markersize=4, label='Validation Dice')
    plt.axhline(y=history['best_dice'], color='r', linestyle='--', linewidth=2, label=f'Best Dice: {history["best_dice"]:.4f}')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Dice Score', fontsize=12, fontweight='bold')
    plt.title('Validation Dice Score Over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'validation_dice.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Combined Loss and Dice
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs + 1, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', color='b', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(val_epochs + 1, val_dices, 'g-', linewidth=2, marker='o', markersize=4, label='Validation Dice')
    ax2.set_ylabel('Dice Score', color='g', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim([0, 1])
    
    plt.title('Training Loss and Validation Dice Score', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'loss_and_dice.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Validation ECE (if available)
    if len(val_eces) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(val_epochs + 1, val_eces, 'm-', linewidth=2, marker='s', markersize=4, label='Expected Calibration Error')
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('ECE', fontsize=12, fontweight='bold')
        plt.title('Expected Calibration Error Over Epochs', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'validation_ece.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Summary Table (as image)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Value'],
        ['Best Validation Dice', f"{history['best_dice']:.4f}"],
        ['Total Epochs', f"{history['total_epochs']}"],
        ['Model Parameters', f"{history['model_params']:,}"],
        ['Final Training Loss', f"{train_losses[-1]:.4f}"],
        ['Final Validation Dice', f"{val_dices[-1]:.4f}" if len(val_dices) > 0 else 'N/A'],
        ['Batch Size', f"{history['config']['batch_size']}"],
        ['Patch Size', f"{history['config']['patch_size']}"],
        ['Base Filters', f"{history['config']['base_filters']}"],
        ['Learning Rate', f"{history['config']['learning_rate']}"],
        ['Mixed Precision', 'Yes' if history['config']['use_amp'] else 'No']
    ]
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4C72B0')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Training Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(figures_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Save summary as text file
    summary_file = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Best Validation Dice: {history['best_dice']:.4f}\n")
        f.write(f"Total Epochs: {history['total_epochs']}\n")
        f.write(f"Model Parameters: {history['model_params']:,}\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.4f}\n")
        if len(val_dices) > 0:
            f.write(f"Final Validation Dice: {val_dices[-1]:.4f}\n")
        f.write(f"\nConfiguration:\n")
        for key, value in history['config'].items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")
    
    print(f"  [OK] Generated {len(os.listdir(figures_dir))} visualization files")
    print(f"  [OK] Summary saved to: {summary_file}")


def generate_comparison_plots(results, output_dir):
    """Generate comparison plots for multiple models"""
    figures_dir = os.path.join(output_dir, 'comparison_figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Sort results by Dice score
    sorted_results = sorted(results, key=lambda x: x['dice'], reverse=True)
    names = [r['name'] for r in sorted_results]
    dices = [r['dice'] for r in sorted_results]
    params = [r['params'] / 1e6 for r in sorted_results]  # Convert to millions
    
    # 1. Dice Score Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), dices, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'][:len(names)])
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Dice Score', fontsize=12, fontweight='bold')
    plt.title('Model Comparison: Dice Scores', fontsize=14, fontweight='bold')
    plt.xticks(range(len(names)), names, rotation=15, ha='right')
    plt.ylim([0, max(dices) * 1.2])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, dice) in enumerate(zip(bars, dices)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dices)*0.02,
                f'{dice:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'dice_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Parameters vs Dice Score
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(names))
    width = 0.35
    
    # Dice scores as bars
    bars1 = ax1.bar(x - width/2, dices, width, label='Dice Score', color='#4C72B0', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Dice Score', color='#4C72B0', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#4C72B0')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.set_ylim([0, max(dices) * 1.2])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Parameters on secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, params, width, label='Parameters (M)', color='#55A868', alpha=0.8)
    ax2.set_ylabel('Parameters (Millions)', color='#55A868', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#55A868')
    
    # Add value labels
    for bar, val in zip(bars1, dices):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dices)*0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, val in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.02,
                f'{val:.2f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.title('Model Comparison: Dice Score vs Parameters', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'dice_vs_params.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Efficiency Plot (Dice per Million Parameters)
    efficiency = [d / p for d, p in zip(dices, params)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), efficiency, color=['#C44E52', '#8172B2', '#CCB974', '#64B5CD'][:len(names)])
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Dice Score per Million Parameters', fontsize=12, fontweight='bold')
    plt.title('Model Efficiency: Dice Score per Million Parameters', fontsize=14, fontweight='bold')
    plt.xticks(range(len(names)), names, rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, eff in zip(bars, efficiency):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency)*0.02,
                f'{eff:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'efficiency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Comparison Summary Table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Model', 'Dice Score', 'Parameters (M)', 'Efficiency'],
    ]
    for name, dice, param, eff in zip(names, dices, params, efficiency):
        table_data.append([
            name,
            f'{dice:.4f}',
            f'{param:.2f}',
            f'{eff:.2f}'
        ])
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4C72B0')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best in each column
    best_dice_idx = dices.index(max(dices))
    best_eff_idx = efficiency.index(max(efficiency))
    
    for i in range(1, len(table_data)):
        if i - 1 == best_dice_idx:
            table[(i, 1)].set_facecolor('#90EE90')  # Light green for best Dice
        if i - 1 == best_eff_idx:
            table[(i, 3)].set_facecolor('#FFD700')  # Gold for best efficiency
    
    plt.title('Model Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(figures_dir, 'comparison_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Save comparison results to JSON
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'models': [
            {
                'name': r['name'],
                'dice': r['dice'],
                'params': r['params'],
                'params_millions': r['params'] / 1e6,
                'efficiency': r['dice'] / (r['params'] / 1e6)
            }
            for r in sorted_results
        ]
    }
    
    comparison_file = os.path.join(output_dir, 'comparison_results.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"  [OK] Generated {len(os.listdir(figures_dir))} comparison visualization files")
    print(f"  [OK] Comparison results saved to: {comparison_file}")
    print(f"  [OK] Comparison figures: {figures_dir}")


def generate_sota_comparison_plots(results_list, output_dir):
    """
    Generate comparison figures using real results only.
    results_list: list of dicts, each with 'name', 'dice', and 'params' (int) or 'params_m' (float).
    """
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    methods = []
    for r in results_list:
        params_m = r.get('params_m')
        if params_m is None:
            params_m = r['params'] / 1e6
        methods.append({
            'name': r['name'],
            'dice': float(r['dice']),
            'params_m': float(params_m),
        })

    names = [m['name'] for m in methods]
    dices = [m['dice'] for m in methods]
    params = [m['params_m'] for m in methods]
    efficiency = [d / p if p > 0 else 0 for d, p in zip(dices, params)]

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]

    # 1) Dice comparison (bar)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), dices, color=bar_colors)
    plt.xlabel('Method', fontsize=12, fontweight='bold')
    plt.ylabel('Dice Score', fontsize=12, fontweight='bold')
    plt.title('Model Comparison: Dice Scores', fontsize=14, fontweight='bold')
    plt.xticks(range(len(names)), names, rotation=20, ha='right')
    plt.ylim([0, min(1.0, max(dices) * 1.15) if dices else 1])
    plt.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, dices):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'sota_dice_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2) Dice vs Parameters (dual-axis)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax1.bar(x - width/2, dices, width, label='Dice', color=bar_colors, alpha=0.8)
    ax1.set_ylabel('Dice Score', color='#4C72B0', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#4C72B0')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha='right')
    ax1.set_ylim([0, min(1.0, max(dices) * 1.15) if dices else 1])
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, params, width, label='Params (M)', color='#55A868', alpha=0.8)
    ax2.set_ylabel('Parameters (Millions)', color='#55A868', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#55A868')

    for bar, val in zip(bars1, dices):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(params) or 1) * 0.02,
                 f'{val:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.title('Dice vs Parameters', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'sota_dice_vs_params.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2b) Scatter: tradeoff between parameters and Dice (publication-ready)
    plt.figure(figsize=(8, 6))
    for i, (n, d, p) in enumerate(zip(names, dices, params)):
        plt.scatter(p, d, s=120, c=[bar_colors[i]], edgecolors='black', linewidths=0.5, zorder=2)
        plt.annotate(n, (p, d), xytext=(6, 6), textcoords='offset points', fontsize=10, fontweight='bold', ha='left')
    plt.xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Dice Score', fontsize=12, fontweight='bold')
    plt.title('Accuracy–Efficiency Tradeoff: Dice vs Parameter Count', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(params) * 1.15 if params else 5)
    plt.ylim(0, min(1.0, max(dices) * 1.1) if dices else 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'dice_params_tradeoff_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3) Efficiency (Dice per M params)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), efficiency, color=bar_colors)
    plt.xlabel('Method', fontsize=12, fontweight='bold')
    plt.ylabel('Dice per Million Params', fontsize=12, fontweight='bold')
    plt.title('Efficiency Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(names)), names, rotation=20, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    max_eff = max(efficiency) if efficiency else 0
    for bar, val in zip(bars, efficiency):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_eff * 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'sota_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4) Summary table
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis('tight')
    ax.axis('off')
    table_data = [['Method', 'Dice', 'Params (M)', 'Dice/M']]
    for n, d, p, e in zip(names, dices, params, efficiency):
        table_data.append([n, f'{d:.2f}', f'{p:.1f}', f'{e:.2f}'])
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4C72B0')
        table[(0, i)].set_text_props(weight='bold', color='white')
    plt.title('Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(figures_dir, 'sota_comparison_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

    results_file = os.path.join(output_dir, 'sota_results.json')
    with open(results_file, 'w') as f:
        json.dump({'methods': methods, 'timestamp': datetime.now().isoformat()}, f, indent=2)

    print("  [OK] SOTA comparison figures saved (real values)")
    print(f"  [OK] SOTA results saved to: {results_file}")


def regenerate_paper_figures_from_logs(
    output_dir,
    history_file: Optional[str] = None,
    comparison_file: Optional[str] = None,
    sota_results_file: Optional[str] = None,
):
    """
    Regenerate all research figures from saved logs without re-running training.
    Expects:
      - training_history.json
      - comparison_results.json (optional, for SOTA-style figures)
    """
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # 1) Training curves and summary from training_history.json (or custom file)
    if history_file is None:
        history_file = os.path.join(output_dir, 'training_history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        print(f"\nRegenerating training visualizations from {history_file}...")
        generate_visualizations(history, output_dir)
    else:
        print(f"\nNo training history found at {history_file}, skipping training plots.")

    # 2) SOTA-style comparison figures
    # Priority: explicit sota_results_file -> explicit comparison_file -> defaults in output_dir
    if sota_results_file is None and comparison_file is None:
        default_sota = os.path.join(output_dir, 'sota_results.json')
        default_comp = os.path.join(output_dir, 'comparison_results.json')
        if os.path.exists(default_sota):
            sota_results_file = default_sota
        elif os.path.exists(default_comp):
            comparison_file = default_comp

    # 2a) From sota_results.json (format: {'methods': [...]})
    if sota_results_file is not None and os.path.exists(sota_results_file):
        with open(sota_results_file, 'r') as f:
            sota_data = json.load(f)
        methods = sota_data.get('methods', [])
        if methods:
            results_list = [
                {
                    'name': m['name'],
                    'dice': m['dice'],
                    # convert params_m (millions) back to absolute params for plotting helper
                    'params': int(m['params_m'] * 1e6),
                }
                for m in methods
            ]
            print(f"Regenerating SOTA comparison figures from {sota_results_file}...")
            generate_sota_comparison_plots(results_list, output_dir)
        else:
            print(f"sota_results.json at {sota_results_file} has no 'methods' entries, skipping SOTA figures.")

    # 2b) Fallback: from comparison_results.json (format: {'models': [...]})
    elif comparison_file is not None and os.path.exists(comparison_file):
        with open(comparison_file, 'r') as f:
            comparison = json.load(f)

        models = comparison.get('models', [])
        if models:
            results_list = [
                {
                    'name': m['name'],
                    'dice': m['dice'],
                    'params': m['params'],
                }
                for m in models
            ]
            print(f"Regenerating SOTA comparison figures from {comparison_file}...")
            generate_sota_comparison_plots(results_list, output_dir)
        else:
            print(f"comparison_results.json at {comparison_file} has no 'models' entries, skipping SOTA figures.")
    else:
        print("No SOTA comparison logs found (neither sota_results.json nor comparison_results.json).")


def print_quick_comparison(sota_results_list: list) -> None:
    """Print a compact table and verdict: is Proposed best by Dice / lightest by params?"""
    if not sota_results_list:
        return
    # Sort by Dice descending for ranking
    by_dice = sorted(sota_results_list, key=lambda x: x['dice'], reverse=True)
    by_params = sorted(sota_results_list, key=lambda x: x['params'])
    proposed = next((x for x in sota_results_list if x['name'] == 'Proposed'), None)
    if not proposed:
        return
    dice_rank = 1 + next(i for i, x in enumerate(by_dice) if x['name'] == 'Proposed')
    is_lightest = by_params[0]['name'] == 'Proposed'
    print("\n" + "=" * 60)
    print("QUICK COMPARISON (Proposed vs baselines)")
    print("=" * 60)
    print(f"{'Model':<20} {'Dice':>8} {'Params':>12}")
    print("-" * 60)
    for r in by_dice:
        params_m = r['params'] / 1e6
        print(f"{r['name']:<20} {r['dice']:>8.4f} {params_m:>10.2f}M")
    print("=" * 60)
    print(f"Proposed: {dice_rank}{'st' if dice_rank == 1 else 'nd' if dice_rank == 2 else 'rd' if dice_rank == 3 else 'th'} by Dice  |  Lightest model: {'Yes' if is_lightest else 'No'}")
    print("=" * 60 + "\n")


def _resolve_baseline_subset(requested_names):
    available = [
        ('Standard UNet', lambda base_ch: Standard3DUNet(in_ch=4, out_ch=4, base_ch=base_ch)),
        ('Attention UNet', lambda base_ch: AttentionUNet3D(in_ch=4, out_ch=4, base_ch=base_ch)),
        ('Residual UNet', lambda base_ch: ResUNet3D(in_ch=4, out_ch=4, base_ch=base_ch)),
        ('V-Net', lambda base_ch: VNet3D(in_ch=4, out_ch=4, base_ch=base_ch)),
    ]
    if not requested_names:
        return available

    alias_map = {
        'standard': 'Standard UNet',
        'standard unet': 'Standard UNet',
        'unet': 'Standard UNet',
        'attention': 'Attention UNet',
        'attention unet': 'Attention UNet',
        'residual': 'Residual UNet',
        'residual unet': 'Residual UNet',
        'resunet': 'Residual UNet',
        'vnet': 'V-Net',
        'v-net': 'V-Net',
    }
    canonical = []
    for raw in requested_names:
        key = raw.strip().lower()
        if not key:
            continue
        if key in alias_map:
            canonical.append(alias_map[key])
            continue
        raise ValueError(
            f"Unknown baseline '{raw}'. Choose from: Standard UNet, Attention UNet, Residual UNet, V-Net."
        )

    selected = []
    for name, factory in available:
        if name in canonical:
            selected.append((name, factory))
    if not selected:
        raise ValueError("No valid baselines selected.")
    return selected


def run_baseline_sota_evaluation(device, train_loader, val_loader, base_ch=24, num_epochs=10, lr=5e-5, use_amp=False, output_dir=None, baseline_names=None):
    """
    Train each baseline for the same number of epochs as the proposed model (fair comparison).
    Returns list of dicts: [{'name': str, 'dice': float, 'params': int}, ...].
    """
    baselines = _resolve_baseline_subset(baseline_names)
    criterion = CombinedLoss(num_classes=4, use_evidential=False)
    if use_amp and device == 'cuda':
        amp_dtype = _get_amp_dtype()
        bl_scaler = GradScaler() if amp_dtype == torch.float16 else None
    else:
        amp_dtype = None
        bl_scaler = None
    results = []
    histories = []

    print("  Baselines selected:", ", ".join(name for name, _ in baselines))

    for name, model_factory in baselines:
        print(f"  Training baseline: {name} ({num_epochs} epochs, fair comparison)...")
        model = model_factory(base_ch).to(device)
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=1e-6)
        best_dice = 0.0
        train_losses = []
        val_history = []

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f'{name} Epoch {epoch+1}/{num_epochs}', leave=False):
                image = _to_device(batch['image'], device, non_blocking=True)
                target = _to_device(batch['seg'], device, non_blocking=True)
                opt.zero_grad()
                if amp_dtype is not None:
                    with _autocast('cuda', dtype=amp_dtype):
                        out = model(image)
                    out_fp32 = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in out.items()}
                    loss, _ = criterion(out_fp32, target, epoch)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    if bl_scaler:
                        bl_scaler.scale(loss).backward()
                        bl_scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        bl_scaler.step(opt)
                        bl_scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                else:
                    out = model(image)
                    loss, _ = criterion(out, target, epoch)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                epoch_loss += float(loss.item())
            scheduler.step()

            epoch_loss /= max(1, len(train_loader))

            model.eval()
            dices = []
            with torch.no_grad():
                for batch in val_loader:
                    image = _to_device(batch['image'], device, non_blocking=True)
                    target = _to_device(batch['seg'], device, non_blocking=True)
                    if amp_dtype is not None:
                        with _autocast('cuda', dtype=amp_dtype):
                            out = model(image)
                        out = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in out.items()}
                    else:
                        out = model(image)
                    pred = model.get_prediction(out)
                    dices.append(compute_dice(pred, target)['dice_mean'])
            mean_dice = float(np.mean(dices)) if dices else 0.0
            best_dice = max(best_dice, mean_dice)
            print(f"    {name} epoch {epoch+1}/{num_epochs} done, val_dice={mean_dice:.4f}, best={best_dice:.4f}")

            train_losses.append(epoch_loss)
            val_history.append({'epoch': epoch, 'val_dice': mean_dice})

        results.append({'name': name, 'dice': best_dice, 'params': params})
        histories.append({
            'name': name,
            'params': params,
            'train_losses': train_losses,
            'val_dices': [v['val_dice'] for v in val_history],
            'epochs': list(range(num_epochs)),
        })
        print(f"    -> {name} best Dice: {best_dice:.4f}, params: {params:,}")
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Optionally save baseline training histories
    if output_dir is not None:
        try:
            os.makedirs(output_dir, exist_ok=True)
            hist_path = os.path.join(output_dir, 'baseline_histories.json')
            import json as _json
            with open(hist_path, 'w') as f:
                _json.dump(histories, f, indent=2)
            print(f"  [OK] Baseline training histories saved to: {hist_path}")
        except Exception as e:
            print(f"Warning: could not save baseline histories: {e}")

    return results


# ===================== TRAINING =====================

def _resolve_runtime_workers(requested_workers, device='cpu'):
    cpu_count = os.cpu_count() or 4
    if requested_workers and requested_workers > 0:
        return requested_workers
    if device == 'cuda':
        return max(4, min(8, cpu_count - 2))
    return max(2, min(6, cpu_count // 2))


def _configure_runtime(args, device):
    cpu_count = os.cpu_count() or 4
    workers = _resolve_runtime_workers(args.workers, device)

    if device == 'cpu':
        intra_threads = args.torch_threads if args.torch_threads > 0 else max(1, min(8, cpu_count - workers))
        interop_threads = args.torch_interop_threads if args.torch_interop_threads > 0 else min(2, max(1, intra_threads // 4))
        torch.set_num_threads(intra_threads)
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass
    else:
        intra_threads = torch.get_num_threads()
        try:
            interop_threads = torch.get_num_interop_threads()
        except RuntimeError:
            interop_threads = 0

    print(
        f"Runtime config: workers={workers}, "
        f"torch_threads={intra_threads}, "
        f"torch_interop_threads={interop_threads}, "
        f"prefetch_factor={args.prefetch_factor if workers > 0 else 'n/a'}"
    )
    return workers


def _make_loader(dataset, batch_size, shuffle, workers, device, drop_last=False, prefetch_factor=2):
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': workers,
        'pin_memory': device == 'cuda',
        'drop_last': drop_last,
    }
    if workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def _setup_cuda_for_max_utilization():
    """Enable CUDA optimizations when GPU is available."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")


def _to_device(tensor, device, non_blocking=False):
    """Transfer tensor to device; use non_blocking for async GPU transfers when pin_memory is used."""
    if device == 'cuda' and non_blocking:
        return tensor.to(device, non_blocking=True)
    return tensor.to(device)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        _setup_cuda_for_max_utilization()
    workers = _configure_runtime(args, device)
    
    # Resolve/download BraTS automatically if missing
    data_path = resolve_brats_path(args.data_dir, download_if_missing=args.download_brats)
    
    # Create datasets
    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    
    if data_path and os.path.exists(data_path):
        print(f"Using real BraTS data from: {data_path}")
        train_dataset = DecathlonBraTSDataset(data_path, 'train', patch_size, get_transforms('train'), args.max_cases)
        val_dataset = DecathlonBraTSDataset(data_path, 'val', patch_size, None, args.max_cases)
    else:
        raise ValueError(
            "BraTS data not found. Use --download_brats or set --data_dir to the dataset path."
        )
    
    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        workers=workers,
        device=device,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        workers=workers,
        device=device,
        drop_last=False,
        prefetch_factor=args.prefetch_factor,
    )
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Optionally run baselines first (save training data + show results) before training proposed
    if getattr(args, 'baselines_first', False):
        print(f"\nRunning baselines first ({args.epochs} epochs each)...")
        baseline_results_first = run_baseline_sota_evaluation(
            device, train_loader, val_loader,
            base_ch=args.base_filters,
            num_epochs=args.epochs,
            lr=args.lr,
            use_amp=args.amp,
            output_dir=args.output_dir,
            baseline_names=args.baseline_names,
        )
        print("\nBaseline results (best Dice over epochs):")
        for r in sorted(baseline_results_first, key=lambda x: x['dice'], reverse=True):
            print(f"  - {r['name']}: dice={r['dice']:.4f}, params={r['params']:,}")
        if getattr(args, 'only_baselines', False):
            print("\nBaseline-only mode enabled. Skipping proposed model training.")
            return
        print("\nBaselines finished. Continuing with Proposed model training...\n")

    # Proposed model: base_filters=24, bottleneck_channels=432 -> ~2.2M params (lighter than baselines ~3.15M)
    bn_ch = None if args.bottleneck_channels == 0 else args.bottleneck_channels
    proposed_components = {
        'use_separable': not args.disable_separable,
        'use_scanner_norm': not args.disable_scanner_norm,
        'use_csa': not args.disable_csa,
        'use_ssfb': not args.disable_ssfb,
    }
    disabled_components = [
        name for name, enabled in {
            'SepConv': proposed_components['use_separable'],
            'ScannerAwareNorm': proposed_components['use_scanner_norm'],
            'CSA': proposed_components['use_csa'],
            'SSFB': proposed_components['use_ssfb'],
        }.items() if not enabled
    ]
    if disabled_components:
        print(f"Proposed ablation variant: disable {', '.join(disabled_components)}")
    else:
        print("Proposed variant: full DALight-3D")
    model = LightweightUNet3D(
        in_ch=4, out_ch=4, base_filters=args.base_filters, num_stages=args.num_stages,
        use_separable=proposed_components['use_separable'],
        use_ssfb=proposed_components['use_ssfb'],
        use_uncertainty=args.use_uncertainty,
        use_scanner_norm=proposed_components['use_scanner_norm'],
        use_csa=proposed_components['use_csa'],
        max_channels=args.max_channels, ssfb_rank=args.ssfb_rank, bottleneck_channels=bn_ch
    ).to(device)
    print(f"Proposed parameters: {model.count_parameters():,}")

    # Training setup (proposed)
    criterion = CombinedLoss(num_classes=4, use_evidential=args.use_uncertainty)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    if args.amp and device == 'cuda':
        amp_dtype = _get_amp_dtype()
        scaler = GradScaler() if amp_dtype == torch.float16 else None
        print(f"AMP enabled: dtype={amp_dtype}, GradScaler={'on' if scaler else 'off (bfloat16)'}")
    else:
        amp_dtype = None
        scaler = None

    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    best_dice = 0.0
    
    # Store training history for visualization
    training_history = {
        'train_losses': [],
        'val_dices': [],
        'val_eces': [],
        'epochs': []
    }
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            image = _to_device(batch['image'], device, non_blocking=True)
            target = _to_device(batch['seg'], device, non_blocking=True)
            scanner_id = _to_device(batch['scanner_id'], device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if amp_dtype is not None:
                with _autocast('cuda', dtype=amp_dtype):
                    outputs = model(image, scanner_id)
                outputs_fp32 = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
                loss, loss_dict = criterion(outputs_fp32, target, epoch)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected, skipping batch")
                    optimizer.zero_grad()
                    continue
                
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            else:
                outputs = model(image, scanner_id)
                loss, loss_dict = criterion(outputs, target, epoch)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected, skipping batch")
                    optimizer.zero_grad()
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        
        # Validate every N epochs
        if (epoch + 1) % args.validate_every == 0 or epoch == args.epochs - 1:
            model.eval()
            val_dice, val_ece = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validation'):
                    image = _to_device(batch['image'], device, non_blocking=True)
                    target = _to_device(batch['seg'], device, non_blocking=True)
                    scanner_id = _to_device(batch['scanner_id'], device, non_blocking=True)
                    
                    if amp_dtype is not None:
                        with _autocast('cuda', dtype=amp_dtype):
                            outputs = model(image, scanner_id)
                        outputs = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
                    else:
                        outputs = model(image, scanner_id)
                    pred = model.get_prediction(outputs)
                    
                    dice = compute_dice(pred, target)
                    ece = compute_ece(outputs['probs'], target)
                    
                    val_dice.append(dice['dice_mean'])
                    val_ece.append(ece)
            
            mean_dice = np.mean(val_dice)
            mean_ece = np.mean(val_ece)
            
            print(f"Epoch {epoch}: loss={train_loss:.4f}, dice={mean_dice:.4f}, ece={mean_ece:.4f}")
            
            writer.add_scalar('val/dice', mean_dice, epoch)
            writer.add_scalar('val/ece', mean_ece, epoch)
            
            # Store history
            training_history['val_dices'].append(float(mean_dice))
            training_history['val_eces'].append(float(mean_ece))
            training_history['epochs'].append(epoch)
            
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                print(f"  -> New best! Dice: {best_dice:.4f}")
        else:
            print(f"Epoch {epoch}: loss={train_loss:.4f}")
            # Store NaN for non-validation epochs
            training_history['val_dices'].append(None)
            training_history['val_eces'].append(None)
            training_history['epochs'].append(epoch)
        
        training_history['train_losses'].append(float(train_loss))
        writer.add_scalar('train/loss', train_loss, epoch)
        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    writer.close()
    
    # Save training history
    training_history['best_dice'] = float(best_dice)
    training_history['total_epochs'] = args.epochs
    training_history['model_params'] = int(model.count_parameters())
    training_history['config'] = {
        'batch_size': args.batch_size,
        'patch_size': args.patch_size,
        'base_filters': args.base_filters,
        'bottleneck_channels': args.bottleneck_channels if args.bottleneck_channels != 0 else None,
        'learning_rate': args.lr,
        'use_amp': args.amp,
        'use_separable': proposed_components['use_separable'],
        'use_scanner_norm': proposed_components['use_scanner_norm'],
        'use_csa': proposed_components['use_csa'],
        'use_ssfb': proposed_components['use_ssfb'],
    }
    
    history_file = os.path.join(args.output_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        import json
        json.dump(training_history, f, indent=2)
    
    # Generate visualizations
    print("\nGenerating research visualizations...")
    generate_visualizations(training_history, args.output_dir)
    
    proposed = {'name': 'Proposed', 'dice': best_dice, 'params': model.count_parameters()}
    sota_results_list = [proposed]
    if not getattr(args, 'skip_baselines', False):
        # Fair comparison: train all baselines for the same number of epochs as the proposed model
        print(f"\nTraining baseline models for fair comparison ({args.epochs} epochs each)...")
        baseline_results = run_baseline_sota_evaluation(
            device, train_loader, val_loader,
            base_ch=args.base_filters,
            num_epochs=args.epochs,
            lr=args.lr,
            use_amp=args.amp,
            output_dir=args.output_dir,
            baseline_names=args.baseline_names,
        )
        sota_results_list = [proposed] + baseline_results
        generate_sota_comparison_plots(sota_results_list, args.output_dir)

        # Quick verdict: is proposed better than others?
        print_quick_comparison(sota_results_list)

    # Publication metrics on the best validation model (skipped in --quick_compare to save time)
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    if getattr(args, 'quick_compare', False):
        print("\nQuick compare: skipping publication metrics.")
    elif os.path.exists(best_model_path):
        print("\nComputing publication metrics on best model (validation set)...")
        state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state)
        pub_metrics = evaluate_publication_metrics(model, val_loader, device, args.output_dir)
        publication_dir = os.path.join(args.output_dir, 'publication')
        os.makedirs(publication_dir, exist_ok=True)
        summary = {
            'best_dice': best_dice,
            'model_params': int(model.count_parameters()),
            'training_history_file': history_file,
            'sota_results': sota_results_list,
            'publication_metrics': pub_metrics,
        }
        with open(os.path.join(publication_dir, 'publication_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    else:
        print("\nBest model checkpoint not found; skipping publication metrics.")
    
    print(f"\nTraining complete. Best Dice: {best_dice:.4f}")
    print(f"Models saved to: {args.output_dir}")
    print(f"Training history: {history_file}")
    print(f"Visualizations: {os.path.join(args.output_dir, 'figures')}")
    print(f"Publication artifacts: {os.path.join(args.output_dir, 'publication')}")
    print_results_summary(args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Training')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory (will download here if missing)')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--download_brats', action='store_true', help='Download BraTS dataset if missing (auto-enabled if not found)')
    parser.add_argument('--compare', action='store_true', help='Quick comparison of all models')
    parser.add_argument(
        '--visualize_only',
        action='store_true',
        help='Regenerate figures from existing logs without any training',
    )
    parser.add_argument(
        '--history_file',
        type=str,
        default=None,
        help='Path to a specific training_history.json to use for visualization',
    )
    parser.add_argument(
        '--sota_results_file',
        type=str,
        default=None,
        help='Path to a specific sota_results.json to use for SOTA comparison figures',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument('--batch_size', type=int, default=1)  # Reduced for memory
    parser.add_argument('--patch_size', type=int, default=64)  # Smaller patches
    parser.add_argument('--lr', type=float, default=5e-5)  # Lower LR for stability
    parser.add_argument('--base_filters', type=int, default=24)
    parser.add_argument('--max_channels', type=int, default=384, help='Max channels in encoder/decoder (increase params without changing base_filters)')
    parser.add_argument('--ssfb_rank', type=int, default=8, help='SSFB low-rank attention rank (lower rank = simpler, more stable attention)')
    parser.add_argument('--num_stages', type=int, default=4, help='Number of encoder/decoder stages (4: ~1.5M params, 5: ~6M params)')
    parser.add_argument('--bottleneck_channels', type=int, default=432, help='Last encoder stage channels; 432 gives ~2.2M params with base_filters=24. Use 0 for default progression (smaller model)')
    parser.add_argument('--use_uncertainty', action='store_true', default=False)  # Disabled - can cause NaN
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision (disabled for stability)')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers. Use 0 for auto-tuning on the current machine')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='Prefetch batches per worker when num_workers > 0')
    parser.add_argument('--torch_threads', type=int, default=0, help='PyTorch intra-op CPU threads. Use 0 for auto-tuning')
    parser.add_argument('--torch_interop_threads', type=int, default=0, help='PyTorch inter-op CPU threads. Use 0 for auto-tuning')
    parser.add_argument('--max_cases', type=int, default=None)
    parser.add_argument('--disable_separable', action='store_true', help='Ablation: replace separable convolutions with standard convolutions in deeper blocks')
    parser.add_argument('--disable_scanner_norm', action='store_true', help='Ablation: replace ScannerAwareNorm with GroupNorm')
    parser.add_argument('--disable_csa', action='store_true', help='Ablation: disable cross-slice attention in deep encoder stages')
    parser.add_argument('--disable_ssfb', action='store_true', help='Ablation: replace SSFB with simple skip fusion at all decoder levels')
    parser.add_argument(
        '--baselines_first',
        action='store_true',
        help='Run baseline models first (save their training histories + show results), then train the proposed model',
    )
    parser.add_argument(
        '--only_baselines',
        action='store_true',
        help='Run only the baseline suite and skip proposed model training',
    )
    parser.add_argument(
        '--skip_baselines',
        action='store_true',
        help='Train only the proposed model and skip the automatic baseline suite at the end',
    )
    parser.add_argument(
        '--baseline_names',
        type=str,
        default='',
        help='Comma-separated baselines to run: standard, attention, residual, vnet. Empty runs all baselines.',
    )
    parser.add_argument(
        '--validate_every',
        type=int,
        default=5,
        help='Validate every N epochs (default: 5, to mimic common paper protocols)'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Short demo: 2 epochs, validate every epoch, max 60 cases (quick Dice check)'
    )
    parser.add_argument(
        '--quick_compare',
        action='store_true',
        help='Quick test: 2 epochs, 60 cases, compare Proposed vs baselines; skip publication metrics; prints verdict'
    )
    args = parser.parse_args()

    if args.demo:
        args.epochs = 2
        args.validate_every = 1
        if args.max_cases is None:
            args.max_cases = 60
        print("Demo mode: 2 epochs, validate every 1, max_cases=%s" % args.max_cases)
    if args.quick_compare:
        args.epochs = 2
        args.validate_every = 1
        if args.max_cases is None:
            args.max_cases = 60
        args.quick_compare = True
        print("Quick compare: 2 epochs, max_cases=%s (Proposed vs baselines; publication metrics skipped)" % args.max_cases)
    args.baseline_names = [x.strip() for x in args.baseline_names.split(',') if x.strip()]

    # Auto-download if dataset isn't present yet
    if not resolve_brats_path(args.data_dir, download_if_missing=False):
        args.download_brats = True
    
    if args.visualize_only:
        regenerate_paper_figures_from_logs(
            args.output_dir,
            history_file=args.history_file,
            comparison_file=None,
            sota_results_file=args.sota_results_file,
        )
    elif args.compare:
        # Quick comparison mode
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n" + "="*60)
        print("MODEL COMPARISON (quick demo)")
        print("="*60)
        print(f"Device: {device}")
        _setup_cuda_for_max_utilization(device)
        workers = _configure_runtime(args, device)
        
        compare_epochs = min(args.epochs, 3)
        print(f"Comparison epochs: {compare_epochs}")
        
        # Select dataset (auto-download if missing)
        data_path = resolve_brats_path(args.data_dir, download_if_missing=args.download_brats)
        
        patch_size = (args.patch_size, args.patch_size, args.patch_size)
        
        if data_path and os.path.exists(data_path):
            print(f"Using real BraTS data from: {data_path}")
            train_ds = DecathlonBraTSDataset(data_path, 'train', patch_size, get_transforms('train'), args.max_cases)
            val_ds = DecathlonBraTSDataset(data_path, 'val', patch_size, None, args.max_cases)
        else:
            raise ValueError(
                "BraTS data not found for comparison. Use --download_brats or set --data_dir."
            )
        
        train_ldr = _make_loader(
            train_ds,
            batch_size=1,
            shuffle=True,
            workers=workers,
            device=device,
            drop_last=False,
            prefetch_factor=args.prefetch_factor,
        )
        val_ldr = _make_loader(
            val_ds,
            batch_size=1,
            shuffle=False,
            workers=workers,
            device=device,
            drop_last=False,
            prefetch_factor=args.prefetch_factor,
        )
        
        models = {
            'Proposed': lambda: LightweightUNet3D(in_ch=4, out_ch=4, base_filters=16, use_uncertainty=False),
            'Standard UNet': lambda: Standard3DUNet(in_ch=4, out_ch=4, base_ch=16),
            'Attention UNet': lambda: AttentionUNet3D(in_ch=4, out_ch=4, base_ch=16),
            'ResUNet': lambda: ResUNet3D(in_ch=4, out_ch=4, base_ch=16),
            'V-Net': lambda: VNet3D(in_ch=4, out_ch=4, base_ch=16),
        }
        
        results = []
        criterion = CombinedLoss(num_classes=4, use_evidential=False)
        
        for name, factory in models.items():
            print(f"\nTraining {name}...")
            model = factory().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            best = 0.0
            
            for epoch in range(compare_epochs):
                model.train()
                pbar = tqdm(train_ldr, desc=f'{name} Epoch {epoch+1}/{compare_epochs}', leave=False)
                for b in pbar:
                    img = _to_device(b['image'], device, non_blocking=True)
                    tgt = _to_device(b['seg'], device, non_blocking=True)
                    opt.zero_grad()
                    out = model(img)
                    loss, _ = criterion(out, tgt, epoch)
                    if not torch.isnan(loss):
                        loss.backward()
                        opt.step()
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                model.eval()
                dices = []
                with torch.no_grad():
                    for b in tqdm(val_ldr, desc='Validating', leave=False):
                        img = _to_device(b['image'], device, non_blocking=True)
                        tgt = _to_device(b['seg'], device, non_blocking=True)
                        out = model(img)
                        pred = model.get_prediction(out)
                        dices.append(compute_dice(pred, tgt)['dice_mean'])
                best = max(best, np.mean(dices) if dices else 0)
            
            results.append({'name': name, 'dice': best, 'params': sum(p.numel() for p in model.parameters())})
            del model
            torch.cuda.empty_cache() if device == 'cuda' else None
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        for r in sorted(results, key=lambda x: x['dice'], reverse=True):
            print(f"{r['name']:<20} Dice: {r['dice']:.4f}  Params: {r['params']:,}")
        print("="*60)
        
        # Generate comparison visualizations
        print("\nGenerating comparison visualizations...")
        generate_comparison_plots(results, args.output_dir)
    else:
        train(args)