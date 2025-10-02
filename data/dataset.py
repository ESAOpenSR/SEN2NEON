import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import rasterio as rio
import random
from matplotlib import pyplot as plt
import os
import json

class SEN2NEON(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        *,
        crop_size_lr: int | None = None,   # in LR pixels; HR crop inferred
        dtype: torch.dtype = torch.float32,
        allow_nan: bool = False,
        pattern_check: bool = True,        # verify HR is integer multiple of LR
    ):
        self.root_dir = Path(root_dir)
        self.df = pd.read_csv(csv_path)
        # Normalize column names for paths
        if {"lr", "hr"}.issubset(self.df.columns):
            self.lr_col, self.hr_col = "lr", "hr"
        elif {"lr_path", "hr_path"}.issubset(self.df.columns):
            self.lr_col, self.hr_col = "lr_path", "hr_path"
        else:
            raise ValueError("CSV must have columns ('lr','hr') or ('lr_path','hr_path').")

        self.crop_size_lr = crop_size_lr
        self.dtype = dtype
        self.allow_nan = allow_nan
        self.pattern_check = pattern_check

        # Pre-resolve paths; keep as Path objects
        self.lr_paths = [self.root_dir / p for p in self.df[self.lr_col].astype(str).tolist()]
        self.hr_paths = [self.root_dir / p for p in self.df[self.hr_col].astype(str).tolist()]

        # Optional “name” field
        self.names = self.df["name"].astype(str).tolist() if "name" in self.df.columns else [
            Path(p).name for p in self.df[self.lr_col].astype(str).tolist()
        ]

        # Meta keys (everything except the path columns)
        self.meta_cols = [c for c in self.df.columns if c not in {self.lr_col, self.hr_col}]

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _read_tiff(path: Path) -> np.ndarray:
        with rio.open(path) as ds:
            arr = ds.read()  # (C,H,W)
            nodata = ds.nodata
        arr = arr.astype(np.float32, copy=False)
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        return arr

    @staticmethod
    def _aligned_crop(lr: np.ndarray, hr: np.ndarray, cs_lr: int) -> tuple[np.ndarray, np.ndarray]:
        C, Hlr, Wlr = lr.shape
        _, Hhr, Whr = hr.shape
        if Hlr == 0 or Wlr == 0 or Hhr == 0 or Whr == 0:
            raise RuntimeError("Empty image.")

        s_h = Hhr // Hlr
        s_w = Whr // Wlr
        if s_h != s_w or (Hhr % Hlr) or (Whr % Wlr):
            raise RuntimeError(f"Non-integer scale or anisotropic scale: LR={Hlr}x{Wlr}, HR={Hhr}x{Whr}.")
        s = s_h

        if cs_lr is None or cs_lr <= 0 or cs_lr > Hlr or cs_lr > Wlr:
            return lr, hr

        y = random.randint(0, Hlr - cs_lr)
        x = random.randint(0, Wlr - cs_lr)
        lr_crop = lr[:, y:y+cs_lr, x:x+cs_lr]
        hr_crop = hr[:, y*s:(y+cs_lr)*s, x*s:(x+cs_lr)*s]
        return lr_crop, hr_crop

    def __getitem__(self, idx: int):
        lr_p = self.lr_paths[idx]
        hr_p = self.hr_paths[idx]
        if not lr_p.exists() or not hr_p.exists():
            raise FileNotFoundError(f"Missing file(s): {lr_p} or {hr_p}")

        lr = self._read_tiff(lr_p)
        hr = self._read_tiff(hr_p)

        if self.pattern_check:
            Hlr, Wlr = lr.shape[-2:]
            Hhr, Whr = hr.shape[-2:]
            if (Hhr % Hlr) or (Whr % Wlr) or (Hhr // Hlr != Whr // Wlr):
                raise RuntimeError(f"Sizes not integer-multiple: LR {Hlr}x{Wlr}, HR {Hhr}x{Whr}, file {lr_p.name}")

        if self.crop_size_lr:
            lr, hr = self._aligned_crop(lr, hr, self.crop_size_lr)

        lr_t = torch.from_numpy(lr).to(self.dtype)
        hr_t = torch.from_numpy(hr).to(self.dtype)
        if not self.allow_nan:
            lr_t = torch.nan_to_num(lr_t)
            hr_t = torch.nan_to_num(hr_t)

        # metadata dict for this row
        row = self.df.iloc[idx]
        wanted = [
            "name",
            "lon", "lat",
            "LC_detail_id", "LC_superclass_id",
            "LC_superclass_text", "LC_detail_text"
        ]

        meta = {}
        for c in wanted:
            if c in row.index:
                v = row[c]
                if isinstance(v, float) and np.isnan(v):
                    v = None
                meta[c] = v

        sample = {
            "lr": lr_t,
            "hr": hr_t,
            "meta": meta
        }
        return sample

    def save_example(
        self,
        out_path: str | None = None,
        *,
        idx: int | None = None,
        k_bands: int | None = None,
        percentiles: tuple[float, float] = (2, 98),
        seed: int | None = None,
    ) -> str:
        """
        Plot and save a PNG with 2 rows:
        - Top row: LR and HR in RGB (bands 0,1,2 if available)
        - Bottom row: LR and HR in random bands

        Subtitles show chosen bands and LC class (if available).
        """
        if seed is not None:
            random.seed(seed)

        # pick random sample if idx not provided
        if idx is None:
            idx = random.randrange(len(self.lr_paths))

        sample = self[idx]               # <- reuse __getitem__
        lr = sample["lr"].numpy()        # [C,H,W]
        hr = sample["hr"].numpy()        # [C,H,W]
        meta = sample.get("meta", {})
        name = meta.get("name", f"sample_{idx}")

        C = min(lr.shape[0], hr.shape[0])

        # random band selection
        k = k_bands or min(3, C)
        bands_rand = sorted(random.sample(range(C), k))

        # always use RGB = first 3 if available
        bands_rgb = [3, 2, 1] if C >= 3 else [0]

        def _stretch(arr_chw, p=(2, 98)):
            k, H, W = arr_chw.shape
            out = np.zeros((k, H, W), dtype=np.float32)
            for i in range(k):
                a = arr_chw[i]
                finite = np.isfinite(a)
                if not finite.any():
                    continue
                lo, hi = np.nanpercentile(a, p[0]), np.nanpercentile(a, p[1])
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = np.nanmin(a), np.nanmax(a)
                b = np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)
                b[~finite] = 0
                out[i] = b
            return out

        def _to_rgb(chw):
            if chw.shape[0] == 1:
                return np.repeat(chw, 3, axis=0)
            elif chw.shape[0] >= 3:
                return chw[:3]
            else:
                pad = np.zeros_like(chw[0:1])
                return np.concatenate([chw, pad], axis=0)

        def _prep_image(arr, bands):
            sub = arr[bands]
            sub = _stretch(sub, percentiles)
            return _to_rgb(sub).transpose(1, 2, 0)

        lr_rgb = _prep_image(lr, bands_rgb)
        hr_rgb = _prep_image(hr, bands_rgb)
        lr_rand = _prep_image(lr, bands_rand)
        hr_rand = _prep_image(hr, bands_rand)

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        def show(ax, img, title, subtitle=None):
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, fontsize=14, weight="bold", pad=12)
            if subtitle:
                ax.text(
                    0.5, -0.08, subtitle,
                    transform=ax.transAxes,
                    ha="center", va="top", fontsize=10
                )

        # top row: RGB
        show(axs[0,0], lr_rgb, "LR (RGB)")
        show(axs[0,1], hr_rgb, "HR (RGB)")

        # bottom row: random bands
        sub = f"Bands {bands_rand}"
        show(axs[1,0], lr_rand, "LR (multispec)", sub)
        show(axs[1,1], hr_rand, "HR (multispec)", sub)

        plt.tight_layout()

        if out_path is None:
            name = Path(lr_path).stem
            out_path = Path.cwd() / f"example_{name}.png"
        out_path = str(Path(out_path))
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        return out_path

    def export_hf_metadata_jsonl(
            self,
            out_path: str | Path,
            *,
            split: str = "val",
            extra_keys: list[str] | None = None,
        ) -> str:
            """
            Write a Hugging Face-friendly JSONL with relative paths from the CSV.

            Record schema (columns included only if present in CSV):
            {
                "id": <stem of lr>,
                "lr": <relative path from CSV>,
                "hr": <relative path from CSV>,
                "split": "val",
                "name": ...,
                "lon": ..., "lat": ...,
                "LC_detail_id": ..., "LC_superclass_id": ...,
                "LC_superclass_text": ..., "LC_detail_text": ...
            }

            Args:
            out_path: where to write metadata.jsonl
            split: split tag to assign to all samples (default: "val")
            extra_keys: optional list of more CSV columns to include (if present)
            """
            lr_col, hr_col = self.lr_col, self.hr_col
            want_default = [
                "name", "lon", "lat",
                "LC_detail_id", "LC_superclass_id",
                "LC_superclass_text", "LC_detail_text",
                "LC_text" 
            ]
            wanted = want_default + (extra_keys or [])

            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with out_path.open("w", encoding="utf-8") as f:
                for _, row in self.df.iterrows():
                    lr_rel = str(row[lr_col])
                    hr_rel = str(row[hr_col])
                    rec = {
                        "id": Path(lr_rel).stem,
                        "lr": lr_rel,
                        "hr": hr_rel,
                        "split": split,
                    }
                    # attach optional columns if present
                    for k in wanted:
                        if k in self.df.columns:
                            v = row[k]
                            if isinstance(v, float) and np.isnan(v):
                                v = None
                            rec[k] = v
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"Wrote {len(self.df)} records → {out_path}")
            return str(out_path)


# ---------- testing ----------
if __name__ == "__main__":
    csv_path = "/data3/SEN2NEON/sen2neon_metadata.csv"  # your slim/clean CSV is fine too
    root     = "/data3/SEN2NEON/"   # dataset root (contains neon_10m_linearized/, neon_2.5m_linearized/)

    ds = SEN2NEON(
        csv_path=csv_path,
        root_dir=root,
        crop_size_lr=64,       # or None
        dtype=torch.float32,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Dataset size: {len(ds)} samples")
    batch = next(iter(loader))
    print(f"Batch LR shape: {batch['lr'].shape}, HR shape: {batch['hr'].shape}")
