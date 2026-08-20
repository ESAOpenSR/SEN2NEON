#!/usr/bin/env python3
"""
download_data.py
----------------
Download the SEN2NEON dataset from Hugging Face into a local folder, preserving
the on-hub layout (metadata indexes, s2_l2a_10m/, neon_2.5m_linearized/).

Usage:
  python download_data.py \
    --repo-id simon-donike/SEN2NEON \
    --out-dir ./data/sen2neon_val

Notes:
- Public datasets don't require a token. If your repo is private, set HF_TOKEN.
- By default we only fetch the needed files (allow_patterns). Use --all to grab everything.
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

DEFAULT_PATTERNS = [
    ".gitattributes",
    "metadata.jsonl",
    "metadata.csv",
    "metadata.parquet",
    "README.md",
    "DATASET_RELEASE_NOTES.md",
    "s2_l2a_10m.sha256",
    "s2_l2a_10m/**",
    "neon_2.5m_linearized/**",
]

def parse_args():
    p = argparse.ArgumentParser(description="Download SEN2NEON from Hugging Face Hub.")
    p.add_argument("--repo-id", type=str, default="simon-donike/SEN2NEON",
                   help="Hugging Face dataset repo id (owner/name).")
    p.add_argument("--out-dir", type=str, default="data/sen2neon_val",
                   help="Local folder to populate with the dataset.")
    p.add_argument("--all", action="store_true",
                   help="Download all files in the repository (ignore allow_patterns).")
    p.add_argument("--high-performance", action="store_true",
                   help="Enable high-performance hf-xet transfers.")
    return p.parse_args()

def main():
    args = parse_args()

    if args.high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = None if args.all else DEFAULT_PATTERNS

    print(f"[HF] Downloading '{args.repo_id}' → {out_dir}")
    if allow_patterns:
        print(f"[HF] Allow patterns: {allow_patterns}")

    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=None,
        # token is auto-read from HF_TOKEN if needed
    )

    # Quick sanity info
    meta = out_dir / "metadata.parquet"
    csv_path = out_dir / "metadata.csv"
    lr_dir = out_dir / "s2_l2a_10m"
    hr_dir = out_dir / "neon_2.5m_linearized"

    n_lr = len(list(lr_dir.glob("**/*.tif"))) if lr_dir.exists() else 0
    n_hr = len(list(hr_dir.glob("**/*.tif"))) if hr_dir.exists() else 0

    print(f"[OK] Local dataset root : {out_dir}")
    print(f"[OK] Cached snapshot    : {local_path}")
    print(f"[OK] metadata.parquet   : {'found' if meta.exists() else 'missing'}")
    print(f"[OK] LR TIFFs            : {n_lr}")
    print(f"[OK] HR TIFFs            : {n_hr}")

    print("\nNext steps:")
    if n_lr != n_hr:
        raise RuntimeError(f"Incomplete download: found {n_lr} LR and {n_hr} HR TIFFs")
    print("  from data.dataset import SEN2NEON")
    print(f"  ds = SEN2NEON(csv_path='{csv_path}', root_dir='{out_dir}')")
    print("  # Or load the metadata index directly with Hugging Face datasets.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
