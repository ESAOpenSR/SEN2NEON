#!/usr/bin/env python3
"""Compare SEN2NEON HF LR to same-date Sentinel-2 L2A from Planetary Computer.

Downloads the HF sample (if missing), fetches matching S2 from PC, warps to the HF
grid, applies HF mask (any band > 0), writes metrics + RGB figures.

  pip install numpy rasterio matplotlib pandas pystac-client planetary-computer huggingface_hub

  python compare_sen2neon_lr_vs_planetary_computer.py \\
    --sample-id 2018_MLBS_3__1_1 --out-dir ./out
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

HF_REPO = "simon-donike/SEN2NEON"
METADATA_URL = (
    "https://raw.githubusercontent.com/ESAOpenSR/SEN2NEON/main/data/metadata/sen2neon_metadata.csv"
)
BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
]
RGB = (3, 2, 1)  # B04, B03, B02


def find_footprints(root: Path) -> Path:
    for p in (
        root / "resources" / "sen2neon_footprints.csv",
        root / "resources" / "sen2neon_foorprints.csv",
        root / "resources" / "sen2neon_metadata_full.csv",
        root / "sen2neon_footprints.csv",
    ):
        if p.is_file() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError(f"No footprints/metadata CSV under {root}")


def ensure_sen2neon(root: Path, sample_id: str) -> Path:
    """Download sample metadata + HF LR/HR tiles if missing."""
    from huggingface_hub import hf_hub_download

    root.mkdir(parents=True, exist_ok=True)
    stem = Path(sample_id).stem

    try:
        meta = find_footprints(root)
    except FileNotFoundError:
        meta = root / "resources" / "sen2neon_metadata_full.csv"
        meta.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading metadata → {meta}")
        urllib.request.urlretrieve(METADATA_URL, meta)

    for rel in (
        f"neon_10m_linearized/{stem}.tif",
        f"neon_2.5m_linearized/{stem}.tif",
    ):
        dest = root / rel
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        print(f"Downloading HF {rel}")
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=rel,
            local_dir=str(root),
        )
    return meta


def footprint_row(csv_path: Path, sample_id: str) -> dict:
    df = pd.read_csv(csv_path)
    stem = Path(sample_id).stem
    if "id" in df.columns:
        hit = df[df["id"].astype(str).str.replace(r"\.tif$", "", regex=True) == stem]
    elif "name" in df.columns:
        hit = df[df["name"].astype(str).str.replace(r"\.tif$", "", regex=True) == stem]
    else:
        raise KeyError(f"{csv_path} needs 'id' or 'name' column")
    if hit.empty:
        raise KeyError(f"{stem!r} not in {csv_path}")
    return hit.iloc[0].to_dict()


def s2_date(row: dict) -> str:
    if row.get("s2_date") and str(row["s2_date"]) not in ("nan", "None"):
        return str(row["s2_date"])[:10]
    m = re.search(r"(20\d{6})T", str(row.get("s2_full_asset_id") or row.get("s2_l2a_gee") or ""))
    if not m:
        raise ValueError("no s2_date in footprints row")
    d = m.group(1)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def query_pc(bbox, date: str, asset_hint: str):
    import planetary_computer as pc
    from pystac_client import Client

    day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    items = list(
        catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=list(bbox),
            datetime=f"{(day - timedelta(days=1)).date()}/{(day + timedelta(days=1)).date()}",
            max_items=50,
        ).items()
    )
    if not items:
        raise RuntimeError(f"no PC S2 items for {date}")

    token = (asset_hint or "").split("/")[-1].upper()
    for it in items:
        if token and token in it.id.upper():
            return pc.sign(it)

    same = [
        it for it in items
        if it.datetime and it.datetime.astimezone(timezone.utc).date().isoformat() == date
    ]
    pool = same or items
    pool.sort(key=lambda it: float(it.properties.get("eo:cloud_cover", 100)))
    return pc.sign(pool[0])


def warp_to_grid(item, transform, crs, h: int, w: int) -> np.ndarray:
    import planetary_computer as pc

    out = np.zeros((len(BANDS), h, w), dtype=np.float32)
    for i, band in enumerate(BANDS):
        href = pc.sign(item.assets[band].href)
        with rasterio.open(href) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=out[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
    return out


def corr(a, b) -> float:
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def metrics(hf, pc, mask) -> dict:
    per = {}
    for i, name in enumerate(BANDS):
        a, b = hf[i, mask], pc[i, mask]
        per[name] = {
            "corr": corr(a, b),
            "mae": float(np.mean(np.abs(a - b))),
            "hf_mean": float(a.mean()),
            "pc_mean": float(b.mean()),
        }
    return {
        "valid_fraction": float(mask.mean()),
        "corr_all": corr(hf[:, mask].ravel(), pc[:, mask].ravel()),
        "mae_all": float(np.mean(np.abs(hf[:, mask] - pc[:, mask]))),
        "per_band": per,
    }


def hr_vs_lr(root: Path, sample_id: str, hf_lr, mask) -> dict | None:
    path = root / "neon_2.5m_linearized" / f"{Path(sample_id).stem}.tif"
    if not path.is_file():
        return None
    hr = rasterio.open(path).read().astype(np.float32)
    c, h, w = hr.shape
    if c != hf_lr.shape[0] or h % 4 or w % 4:
        return None
    ds = hr.reshape(c, h // 4, 4, w // 4, 4).mean(axis=(2, 4))
    m = mask & np.any(ds > 0, axis=0)
    corrs = {BANDS[i]: corr(ds[i, m], hf_lr[i, m]) for i in range(c)}
    return {"mean_corr": float(np.mean(list(corrs.values()))), "per_band_corr": corrs}


def rgb(stack):
    return np.stack([stack[i] for i in RGB], axis=-1)


def show(rgb_arr, mask, scale: float):
    x = np.clip(rgb_arr / scale, 0, 1).copy()
    x[~mask] = 0
    return x


def save_figures(
    out_dir: Path,
    stem: str,
    hf_lr,
    pc,
    mask,
    *,
    hf_hr=None,
    s2_date,
    neon_date,
    pc_id,
) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lr_rgb, pc_rgb = rgb(hf_lr), rgb(pc)
    paths = {}

    # LR vs PC
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, im, title in zip(
        axes,
        [show(lr_rgb, mask, 3000), show(pc_rgb, mask, 3000)],
        ["HF SEN2NEON LR", "Planetary Computer S2 L2A"],
    ):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{stem} | S2={s2_date} | NEON={neon_date} | raw/3000 | HF mask\n{pc_id}")
    fig.tight_layout()
    p = out_dir / f"{stem}_hf_vs_pc.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["lr_vs_pc"] = str(p)

    # Zoom LR vs PC
    cy, cx = hf_lr.shape[1] // 2, hf_lr.shape[2] // 2
    sl = (slice(cy - 48, cy + 48), slice(cx - 48, cx + 48))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, im, title in zip(
        axes,
        [show(lr_rgb, mask, 3000), show(pc_rgb, mask, 3000)],
        ["HF LR zoom", "PC zoom"],
    ):
        ax.imshow(im[sl])
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{stem} center 96x96 | raw/3000 | HF mask")
    fig.tight_layout()
    p = out_dir / f"{stem}_hf_vs_pc_zoom.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["zoom"] = str(p)

    if hf_hr is None:
        return paths

    hr_rgb = rgb(hf_hr)
    # Native: LR | HR | PC
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, im, title in zip(
        axes,
        [show(lr_rgb, mask, 3000), np.clip(hr_rgb / 3000, 0, 1), show(pc_rgb, mask, 3000)],
        ["HF LR (10 m)", "HF HR (2.5 m)", "PC S2 (LR grid)"],
    ):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{stem} | LR / HR / PC | raw/3000")
    fig.tight_layout()
    p = out_dir / f"{stem}_hf_lr_hr_pc.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["lr_hr_pc"] = str(p)

    # Same 10 m: LR | HR↓4x | PC
    c, hh, ww = hf_hr.shape
    if hh == hf_lr.shape[1] * 4 and ww == hf_lr.shape[2] * 4:
        hr_ds = hf_hr.reshape(c, hh // 4, 4, ww // 4, 4).mean(axis=(2, 4))
        hr_ds_rgb = rgb(hr_ds)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for ax, im, title in zip(
            axes,
            [show(lr_rgb, mask, 3000), show(hr_ds_rgb, mask, 3000), show(pc_rgb, mask, 3000)],
            ["HF LR (10 m)", "HF HR → 10 m (area ↓4×)", "PC S2 (LR grid)"],
        ):
            ax.imshow(im)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{stem} | same 10 m grid | raw/3000 | HF mask")
        fig.tight_layout()
        p = out_dir / f"{stem}_hf_lr_hrds_pc.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["lr_hrds_pc"] = str(p)

    return paths


def load_hr(root: Path, sample_id: str):
    path = root / "neon_2.5m_linearized" / f"{Path(sample_id).stem}.tif"
    if not path.is_file():
        return None
    return rasterio.open(path).read().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sen2neon-root", type=Path, default=Path("data/sen2neon"))
    ap.add_argument(
        "--sample-id",
        default="2018_MLBS_3__1_1",
        help="SEN2NEON tile id (default: clean sample with no nodata holes)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("sen2neon_vs_pc_compare"))
    ap.add_argument("--footprints-csv", type=Path, default=None)
    ap.add_argument("--no-download", action="store_true", help="Require local HF files only")
    args = ap.parse_args()

    sample_id = Path(args.sample_id).stem
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.no_download:
        footprints = args.footprints_csv or find_footprints(args.sen2neon_root)
    else:
        footprints = args.footprints_csv or ensure_sen2neon(args.sen2neon_root, sample_id)

    row = footprint_row(footprints, sample_id)
    lr_path = args.sen2neon_root / "neon_10m_linearized" / f"{sample_id}.tif"
    if not lr_path.is_file():
        raise FileNotFoundError(lr_path)

    date = s2_date(row)
    neon = str(row.get("neon_date") or "")[:10] or None
    asset = str(row.get("s2_full_asset_id") or row.get("s2_l2a_gee") or "")

    with rasterio.open(lr_path) as ds:
        hf = ds.read().astype(np.float32)
        transform, crs = ds.transform, ds.crs
        h, w = ds.height, ds.width
        profile = ds.profile.copy()
        bbox = list(transform_bounds(ds.crs, "EPSG:4326", *ds.bounds))

    item = query_pc(bbox, date, asset)
    print(f"{sample_id}: S2={date} NEON={neon} PC={item.id}")

    pc = warp_to_grid(item, transform, crs, h, w)
    mask = np.any(hf > 0, axis=0)
    hf_hr = load_hr(args.sen2neon_root, sample_id)
    result = metrics(hf, pc, mask)
    result.update(
        sample_id=sample_id,
        s2_date=date,
        neon_date=neon,
        s2_asset=asset,
        pc_item=item.id,
        hf_hr_vs_lr=hr_vs_lr(args.sen2neon_root, sample_id, hf, mask),
        figures=save_figures(
            out,
            sample_id,
            hf,
            pc,
            mask,
            hf_hr=hf_hr,
            s2_date=date,
            neon_date=neon,
            pc_id=item.id,
        ),
    )

    profile.update(dtype="float32", count=len(BANDS), compress="deflate")
    tif = out / f"{sample_id}_pc_on_hf_grid.tif"
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(pc)
    result["pc_geotiff"] = str(tif)

    (out / f"{sample_id}_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"valid={result['valid_fraction']:.3f} corr_all={result['corr_all']:.4f} mae={result['mae_all']:.1f}")
    for b, d in result["per_band"].items():
        print(f"  {b:4s} corr={d['corr']:+.3f} mae={d['mae']:.1f}")
    if result["hf_hr_vs_lr"]:
        print(f"HF-HR→10m vs HF-LR mean corr={result['hf_hr_vs_lr']['mean_corr']:.4f}")
    for k, v in result["figures"].items():
        print(f"figure[{k}]={v}")


if __name__ == "__main__":
    main()
