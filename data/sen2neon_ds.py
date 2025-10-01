# pip install rasterio
import os, glob, random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio as rio
import csv
from rasterio.warp import transform
import pandas as pd
from rasterio.windows import from_bounds
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import random


class SEN2NEON(Dataset):
    """
    Pairs LR/HR TIFFs by filename and returns {'lr':Tensor[C,H,W], 'hr':Tensor[C,H,W], 'name':str}.
    Optional aligned random crops controlled by crop_size_lr (in LR pixels).
    """
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        pattern: str = "*.tif",
        crop_size_lr: int | None = None,
        dtype: torch.dtype = torch.float32,
        allow_nan: bool = True,
    ):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.crop_size_lr = crop_size_lr
        self.dtype = dtype
        self.allow_nan = allow_nan

        lr_files = {Path(p).name: Path(p) for p in glob.glob(str(self.lr_dir / pattern))}
        hr_files = {Path(p).name: Path(p) for p in glob.glob(str(self.hr_dir / pattern))}
        common = sorted(set(lr_files.keys()) & set(hr_files.keys()))
        if not common:
            raise RuntimeError("No matching filenames between LR and HR folders.")
        self.pairs = [(lr_files[n], hr_files[n]) for n in common]
        
        # check for centroid info and LC info
        self.try_to_read_info()

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _read_tiff(p: Path) -> np.ndarray:
        # Returns array in [C,H,W], dtype float32
        with rio.open(p) as ds:
            arr = ds.read()  # (C,H,W)
            nodata = ds.nodata
        arr = arr.astype(np.float32, copy=False)
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        return arr

    def _rand_crop(self, lr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # lr: [C,H,W], hr: [C,H,W]; crop aligned by inferred scale
        C, Hlr, Wlr = lr.shape
        Chr, Hhr, Whr = hr.shape
        scale_h = Hhr // Hlr
        scale_w = Whr // Wlr
        if scale_h != scale_w:
            raise RuntimeError(f"Non-uniform scale: {scale_h} vs {scale_w}")
        s = scale_h

        cs = self.crop_size_lr
        if cs is None or cs <= 0 or cs > Hlr or cs > Wlr:
            return lr, hr  # no crop or too large; return full images

        y = random.randint(0, Hlr - cs)
        x = random.randint(0, Wlr - cs)
        lr_crop = lr[:, y:y+cs, x:x+cs]
        hr_crop = hr[:, y*s:(y+cs)*s, x*s:(x+cs)*s]
        return lr_crop, hr_crop

    def __getitem__(self, idx: int):
        lr_path, hr_path = self.pairs[idx]
        lr = self._read_tiff(lr_path)
        hr = self._read_tiff(hr_path)

        # basic sanity: spatial scale
        Hlr, Wlr = lr.shape[-2:]
        Hhr, Whr = hr.shape[-2:]
        if Hlr == 0 or Wlr == 0 or Hhr == 0 or Whr == 0:
            raise RuntimeError(f"Empty image for {lr_path.name}")
        if Hhr % Hlr != 0 or Whr % Wlr != 0:
            raise RuntimeError(f"Sizes not integer-multiple: LR {Hlr}x{Wlr}, HR {Hhr}x{Whr}, file {lr_path.name}")

        # optional crop
        if self.crop_size_lr:
            lr, hr = self._rand_crop(lr, hr)

        # convert to tensors
        lr_t = torch.from_numpy(lr).to(self.dtype)
        hr_t = torch.from_numpy(hr).to(self.dtype)

        if not self.allow_nan:
            # replace NaNs with 0 if desired
            lr_t = torch.nan_to_num(lr_t)
            hr_t = torch.nan_to_num(hr_t)

        return {"lr": lr_t, "hr": hr_t, "name": lr_path.name}
    
    def try_to_read_info(self):
        csv_path = "/data3/SEN2NEON/processed/sen2neon_centroids.csv"
        lc_path  = "/data3/SEN2NEON/processed/centroids_with_LC.csv"
        cleaned_lc_path = "/data1/simon/GitHub/sen2neon_val/centroids_with_LC_cleaned.csv"
        if Path(cleaned_lc_path).exists():
            df = pd.read_csv(cleaned_lc_path)
            print("Centroid + Cleaned LandCover info exists, loading to dataset.")
            self.centroids = df
        elif Path(lc_path).exists():
            df = pd.read_csv(lc_path)
            print("Centroid + LandCover info exists, loading to dataset.")
            self.centroids = df
        elif Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            print("Centroid info exists, loading to dataset.")
            self.centroids = df
        else:
            print("No centroid info found. Create with create_csv().")
            
    def extract_centroid(
        self,
        lr_dir: str,
        hr_dir: str,
        out_csv: str,
        pattern: str = "*.tif",
    ) -> str:
        """
        Compute centroids for all paired LR/HR TIFFs (by identical filename) and save to CSV.

        CSV columns:
        name, lr_path, hr_path, crs, x, y, lon, lat
        - (x, y) are centroid coords in native CRS
        - (lon, lat) are centroid in EPSG:4326 if CRS available, else NaN

        Returns:
        The output CSV path.
        """
        lr_dir = Path(lr_dir)
        hr_dir = Path(hr_dir)

        lr_files = {Path(p).name: Path(p) for p in glob.glob(str(lr_dir / pattern))}
        hr_files = {Path(p).name: Path(p) for p in glob.glob(str(hr_dir / pattern))}
        common = sorted(set(lr_files) & set(hr_files))
        if not common:
            raise RuntimeError("No matching filenames between LR and HR folders.")

        rows = []
        for name in common:
            lr_path = lr_files[name]
            hr_path = hr_files[name]

            with rio.open(lr_path) as ds:
                b = ds.bounds  # left, bottom, right, top
                cx = (b.left + b.right) * 0.5
                cy = (b.bottom + b.top) * 0.5
                crs = ds.crs

                if crs is not None:
                    try:
                        # to EPSG:4326
                        lon, lat = transform(crs, "EPSG:4326", [cx], [cy])
                        lon, lat = lon[0], lat[0]
                    except Exception:
                        lon = float("nan")
                        lat = float("nan")
                else:
                    lon = float("nan")
                    lat = float("nan")

            rows.append({
                "name": name,
                "lr_path": str(lr_path),
                "hr_path": str(hr_path),
                "crs": str(crs) if crs is not None else "",
                "x": cx,
                "y": cy,
                "lon": lon,
                "lat": lat,
            })

        out_csv = str(Path(out_csv))
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["name","lr_path","hr_path","crs","x","y","lon","lat"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print("Wrote centroids for", len(rows), "images to", out_csv)

    def create_csv(self, out_csv: str, pattern: str = "*.tif") -> str:
        """
        Wrapper to call extract_centroid with instance's LR/HR dirs.

        Returns:
        The output CSV path.
        """
        self.extract_centroid(
            self.lr_dir, self.hr_dir, out_csv, pattern
        )
        
    def read_centroid_csv(self, csv_path: str) -> list[dict]:
        """
        Read a centroid CSV file created by create_csv().

        Returns:
        A list of dicts with keys: name, lr_path, hr_path, crs, x, y, lon, lat
        """
        self.centroids = pd.read_csv(csv_path)
        print("Read", len(self.centroids), "centroids from", csv_path)

    def add_landcover_mode(
        self,
        csv_in: str,
        landcover_tif: str,
        out_csv: str | None = None,
        lr_path_col: str = "lr_path",
        out_col: str = "LC_class",
        ignore_values: list[int] | None = None,
    ) -> str:
        """
        For each row in csv_in (must include lr_path), open the LR TIFF, get its bounds,
        reproject to landcover CRS, read landcover window, compute mode (most frequent class),
        and write it as `out_col`. Saves to out_csv (or overwrites csv_in if out_csv=None).

        Args
        ----
        csv_in : path to input CSV (has 'lr_path')
        landcover_tif : categorical raster (e.g., NLCD/USGS) with integer classes
        out_csv : output CSV path (default: overwrite input)
        lr_path_col : column name with LR tiff path
        out_col : new column name for dominant class
        ignore_values : class ids to ignore (e.g., [0]) in addition to NoData

        Returns
        -------
        str : path to the written CSV
        """
        df = pd.read_csv(csv_in)
        if lr_path_col not in df.columns:
            raise ValueError(f"CSV missing required column '{lr_path_col}'")

        out_csv = out_csv or csv_in
        ignore_values = set(ignore_values or [])

        # Open landcover once
        with rio.open(landcover_tif) as lc_ds:
            lc_crs = lc_ds.crs
            lc_nodata = lc_ds.nodata

            modes: list[float] = []
            for i, row in tqdm(df.iterrows(), total=df.shape[0]):
                lr_path = row[lr_path_col]
                if not isinstance(lr_path, str) or not Path(lr_path).exists():
                    modes.append(np.nan)
                    continue

                try:
                    # Footprint from LR image
                    with rio.open(lr_path) as lr_ds:
                        lr_bounds = lr_ds.bounds      # in lr_ds.crs
                        lr_crs = lr_ds.crs

                    # Transform bounds to landcover CRS
                    if lr_crs is not None and lc_crs is not None and lr_crs != lc_crs:
                        xs, ys = transform(
                            lr_crs, lc_crs,
                            [lr_bounds.left,  lr_bounds.right, lr_bounds.left,  lr_bounds.right],
                            [lr_bounds.bottom, lr_bounds.bottom, lr_bounds.top, lr_bounds.top]
                        )
                        minx, maxx = min(xs), max(xs)
                        miny, maxy = min(ys), max(ys)
                    else:
                        # Same CRS or missing CRS (assume same)
                        minx, miny, maxx, maxy = lr_bounds.left, lr_bounds.bottom, lr_bounds.right, lr_bounds.top

                    # Build window and read landcover patch
                    win = from_bounds(minx, miny, maxx, maxy, transform=lc_ds.transform)
                    patch = lc_ds.read(1, window=win, boundless=True, fill_value=lc_nodata)

                    # Compute mode (ignore NoData and optional ignore_values)
                    vals = patch.ravel()
                    mask = np.ones_like(vals, dtype=bool)
                    if lc_nodata is not None:
                        mask &= (vals != lc_nodata)
                    if ignore_values:
                        for iv in ignore_values:
                            mask &= (vals != iv)

                    valid = vals[mask]
                    if valid.size == 0:
                        modes.append(np.nan)
                    else:
                        uniq, cnts = np.unique(valid, return_counts=True)
                        modes.append(float(uniq[np.argmax(cnts)]))

                except Exception:
                    modes.append(np.nan)

        df[out_col] = modes
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        return out_csv

    def clean_landcover_mode(
        self,
        legend_csv: str,
        lc_col: str = "LC_class",
    ) -> pd.DataFrame:
        """
        # /data3/landcover_s2naip/csvs/legend.csv
        Merge land-cover legend onto self.centroids using LC code.

        legend_csv columns required:
          Number, SuperNumber, Detail, Class, SuperClass

        Args:
          legend_csv: path to legend CSV.
          lc_col: column in self.centroids with numeric LC code.
          inplace: if True, write back to self.centroids; returns the merged df either way.
        """
        if getattr(self, "centroids", None) is None:
            raise ValueError("self.centroids is None. Load it first (e.g., self.read_centroid_csv(...)).")
        if lc_col not in self.centroids.columns:
            raise ValueError(f"'{lc_col}' not found in self.centroids.")

        df = self.centroids.copy()

        map_df = pd.read_csv(legend_csv, sep=";")
        required = {"Number", "SuperNumber", "Detail", "Class", "SuperClass"}
        missing = required - set(map_df.columns)
        if missing:
            raise ValueError(f"Legend CSV missing columns: {missing}")

        # robust numeric keys (handles floats/NaNs gracefully)
        key = "__lc_key__"
        df[key] = pd.to_numeric(df[lc_col], errors="coerce").astype("Int64")
        map_df["Number"] = pd.to_numeric(map_df["Number"], errors="coerce").astype("Int64")

        # rename legend columns -> LC_* namespace
        legend_renamed = map_df.rename(columns={
            "Number": "LC_Number",
            "SuperNumber": "LC_SuperNumber",
            "Detail": "LC_Detail",
            "Class": "LC_Class",
            "SuperClass": "LC_SuperClass",
        })

        merged = df.merge(legend_renamed, left_on=key, right_on="LC_Number", how="left")
        merged.drop(columns=[key], inplace=True)

        # human-friendly text label
        merged["LC_text"] = merged["LC_Detail"]
        merged.to_csv("/data1/simon/GitHub/sen2neon_val/centroids_with_LC_cleaned.csv", index=False)
    
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
            idx = random.randrange(len(self.pairs))

        lr_path, hr_path = self.pairs[idx]
        lr = self._read_tiff(lr_path)
        hr = self._read_tiff(hr_path)

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

        # land cover class if present
        lc_class = None
        if hasattr(self, "centroids") and "LC_SuperClass" in self.centroids.columns:
            row = self.centroids[self.centroids["lr_path"] == str(lr_path)]
            if not row.empty:
                lc_class = row["LC_SuperClass"].iloc[0]

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
        if lc_class:
            sub += f" | LC: {lc_class}"
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


# ---------- Example usage ----------
if __name__ == "__main__":
    LR_DIR = "/data3/SEN2NEON/processed/neon_10m_linearized"
    HR_DIR = "/data3/SEN2NEON/processed/neon_2.5m_linearized"

    ds = SEN2NEON(
        LR_DIR, HR_DIR,
        pattern="*.tif",
        crop_size_lr=128,   # None for full images; 128 for aligned LR crops (HR crop auto-scales)
        dtype=torch.float32,
        allow_nan=True
    )
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    batch = next(iter(loader))
    print(batch["lr"].shape, batch["hr"].shape, batch["name"])
    
    # helper to create/read centroid CSV
    #ds.create_csv(out_csv="/data3/SEN2NEON/sen2neon_centroids.csv")
    #ds.read_centroid_csv("/data3/SEN2NEON/sen2neon_centroids.csv")
    
    #ds.add_landcover_mode(
    #    csv_in="/data3/SEN2NEON/sen2neon_centroids.csv",
    #    landcover_tif="/data3/landcover_s2naip/landcover/us_LC.tif",
    #    out_csv="/data3/SEN2NEON/processed/centroids_with_LC.csv",
    #    lr_path_col="lr_path",
    #    out_col="LC_class",
    #    ignore_values=None,
    #)

    #ds.clean_landcover_mode(
    #    legend_csv="/data3/landcover_s2naip/csvs/legend.csv",
    #    lc_col="LC_class",
    #)
    #ds.centroids.to_csv("/data1/simon/GitHub/sen2neon_val/centroids_with_LC.csv", index=False)
    
    # Save example PNG
    for i in range(10):
        ds.save_example(out_path=f"sen2neon_examples/example_{i}.png")
