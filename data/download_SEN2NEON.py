#!/usr/bin/env python3
"""
download_data.py
----------------
Download the SEN2NEON dataset from Hugging Face into a local folder, preserving
the on-hub layout. The canonical 2.5 m HR product is downloaded by default;
the supplementary 1 m workflow product can be selected explicitly.

Usage:
  python download_data.py \
    --repo-id isp-uv-es/SEN2NEON \
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

BASE_PATTERNS = [
    ".gitattributes",
    "metadata.jsonl",
    "metadata.csv",
    "metadata.parquet",
    "README.md",
    "DATASET_RELEASE_NOTES.md",
    "s2_l2a_10m.sha256",
    "s2_l2a_10m/**",
]

HR_DIRECTORIES = {
    "2.5": "neon_2.5m_linearized",
    "1": "neon_1m_linearized",
}

def parse_args():
    p = argparse.ArgumentParser(description="Download SEN2NEON from Hugging Face Hub.")
    p.add_argument("--repo-id", type=str, default="isp-uv-es/SEN2NEON",
                   help="Hugging Face dataset repo id (owner/name).")
    p.add_argument("--out-dir", type=str, default="data/sen2neon_val",
                   help="Local folder to populate with the dataset.")
    p.add_argument("--all", action="store_true",
                   help="Download all files in the repository (ignore allow_patterns).")
    p.add_argument(
        "--hr-resolution",
        choices=("2.5", "1", "both"),
        default="2.5",
        help=(
            "HR product to download: canonical paper product '2.5' (default), "
            "supplementary workflow product '1', or 'both'."
        ),
    )
    p.add_argument("--high-performance", action="store_true",
                   help="Enable high-performance hf-xet transfers.")
    return p.parse_args()

def main():
    args = parse_args()

    if args.high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_resolutions = (
        tuple(HR_DIRECTORIES) if args.all or args.hr_resolution == "both"
        else (args.hr_resolution,)
    )
    allow_patterns = None if args.all else [
        *BASE_PATTERNS,
        *(f"{HR_DIRECTORIES[resolution]}/**" for resolution in selected_resolutions),
    ]

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
    n_lr = len(list(lr_dir.glob("**/*.tif"))) if lr_dir.exists() else 0
    hr_counts = {}
    for resolution in selected_resolutions:
        hr_dir = out_dir / HR_DIRECTORIES[resolution]
        hr_counts[resolution] = len(list(hr_dir.glob("**/*.tif"))) if hr_dir.exists() else 0

    print(f"[OK] Local dataset root : {out_dir}")
    print(f"[OK] Cached snapshot    : {local_path}")
    print(f"[OK] metadata.parquet   : {'found' if meta.exists() else 'missing'}")
    print(f"[OK] LR TIFFs            : {n_lr}")
    for resolution, n_hr in hr_counts.items():
        role = "canonical" if resolution == "2.5" else "supplementary"
        print(f"[OK] HR TIFFs ({resolution} m, {role}): {n_hr}")

    print("\nNext steps:")
    mismatches = {
        resolution: n_hr for resolution, n_hr in hr_counts.items() if n_lr != n_hr
    }
    if mismatches:
        counts = ", ".join(f"{resolution} m={count}" for resolution, count in mismatches.items())
        raise RuntimeError(f"Incomplete download: found LR={n_lr}, {counts} HR TIFFs")
    print("  from data.dataset import SEN2NEON")
    example_resolution = selected_resolutions[0]
    print(
        f"  ds = SEN2NEON(csv_path='{csv_path}', root_dir='{out_dir}', "
        f"hr_resolution={example_resolution})"
    )
    print("  # Or load the metadata index directly with Hugging Face datasets.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
