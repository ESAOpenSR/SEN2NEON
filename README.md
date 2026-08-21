> **Release correction — 20 August 2026:** A previous dataset upload mistakenly provided a linearized 10 m NEON product as `lr`. This was corrected on 20 August 2026. The canonical LR files are now in `s2_l2a_10m/` and contain the original Sentinel-2 observations exported through Google Earth Engine.

<p align="center">
  <img src="resources/sen2neon_banner.png" alt="SEN2NEON example" width="100%">
</p>


<img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/opensr_logo.png?raw=true" width="250"/>



# SEN2NEON - Dataset for Multispectral Sentinel-2 Super-Resolution 

SEN2NEON is a PyTorch-based dataset and evaluation suite for the **SEN2NEON multispectral super-resolution benchmark**. The project aligns Sentinel-2 Level-2A reflectances with co-registered 1 m AVIRIS-NG hyperspectral imagery, creating the first large-scale reference set for the Sentinel-2 20 m bands under real reflectance conditions.

> **HR resolution:** The **2.5 m HR product is the canonical SEN2NEON product described and evaluated in the paper**. We also provide spatially aligned 1 m HR tiles because they are produced as an intermediate output of our processing workflow and may be useful to others. The 1 m product is supplementary; it does not redefine the paper benchmark or its reported results.

## Highlights

- **Harmonized dataset** with 2,269 spatially aligned tiles (1024×1024 pixels at 2.5 m) spanning 12 Sentinel-2-equivalent bands (B1–B9, B8A, B11–B12).
- **Land-cover aware metadata** (centroids, fine/coarse classes) for stratified analyses.
- **Ready-to-use PyTorch Dataset/DataModule** that pairs observed Sentinel-2 LR (all bands on an aligned 10 m grid) with canonical 2.5 m or supplementary 1 m NEON-derived HR GeoTIFF tiles.
- **Validation pipeline** with visualization, histogram matching, and metric logging that integrates OpenSR-Test semantics.
- **Baseline model adapters** for SEN2SR variants, LDSR-S2, and SRGAN to reproduce the benchmarks from the accompanying paper.

---

## Dataset at a glance

- **LR product:** Original Sentinel-2 Level-2A observations exported from GEE `COPERNICUS/S2_SR_HARMONIZED`. The 12 bands are stored as a common 10 m, 256×256 aligned stack; this convenience product does not preserve separate native 10/20/60 m grids.
- **Canonical HR product (paper):** AVIRIS-NG hyperspectral imagery harmonized to Sentinel-2A/B spectral response functions and provided at 2.5 m (1024×1024), excluding B10. This is the default throughout the code and the target used for the published benchmark and metrics.
- **Supplementary HR product:** The same aligned, harmonized data at 1 m (2560×2560). We include these tiles only because our production workflow creates them and they may support other research; they are not the HR definition evaluated in the SEN2NEON paper.
- **Coverage:** 2,269 tiles from 119 NEON acquisition assets (110 calendar dates, 2018–2024), spanning forests, grasslands, shrublands, croplands, water, barren land, and built-up areas.
- **Metadata:** Per-tile CSV/JSON metadata listing relative paths, projected and geographic centroids, and both detailed and superclass land-cover labels for stratified evaluation.

These characteristics make SEN2NEON suitable for benchmarking super-resolution models that target the Sentinel-2 20 m bands without relying on synthetic degradations.

---

## 1. Environment setup

Create a Python environment (≥3.10 recommended) and install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick the wheel for your system
pip install pandas numpy rasterio matplotlib tqdm huggingface_hub datasets opensr_test
```

> Recent `huggingface_hub` releases use `hf-xet` for large-file transfers. Install GPU-specific PyTorch wheels as needed.

---

## 2. Download the dataset from Hugging Face

The dataset is hosted on the Hugging Face Hub at [`isp-uv-es/SEN2NEON`](https://huggingface.co/datasets/isp-uv-es/SEN2NEON/). Use the helper script to keep the on-hub directory layout:

```bash
python data/download_SEN2NEON.py \
  --repo-id isp-uv-es/SEN2NEON \
  --out-dir ./data/sen2neon \
  --hr-resolution 2.5 \
  --high-performance     # optional, enables high-performance hf-xet transfers
```

The command above downloads the canonical 2.5 m HR product. To select the supplementary 1 m product instead, pass `--hr-resolution 1`; use `--hr-resolution both` to download both. Metadata indexes are downloaded in every case.

Key options:

- `--hr-resolution {2.5,1,both}`: choose the HR files to download; the default is canonical `2.5`.
- `--all`: download every file in the repository instead of the curated subset.
- `--high-performance`: enable `HF_XET_HIGH_PERFORMANCE` for faster transfers.
- Set `HF_TOKEN` if you need to authenticate to private mirrors.

After the command finishes you should see the following structure:

```
./data/sen2neon
├── metadata.parquet            # default Hugging Face dataset index
├── metadata.jsonl              # portable equivalent index
├── metadata.csv                # CSV index used by the PyTorch loader
├── s2_l2a_10m/                 # canonical observed S2 LR, 12×256×256
├── s2_l2a_10m.sha256           # checksums for all canonical LR TIFFs
├── neon_2.5m_linearized/       # canonical paper HR, 12×1024×1024
├── neon_1m_linearized/         # supplementary workflow HR, 12×2560×2560
└── DATASET_RELEASE_NOTES.md
```

Only the HR directory or directories selected with `--hr-resolution` will be present.

The CSV, JSONL, and Parquet indexes contain the same one-row-per-tile metadata. `hr` remains a backward-compatible alias for the canonical 2.5 m path. Explicit fields include `hr_2_5m_path`, `hr_1m_path`, `hr_available_resolutions_m`, `hr_canonical_resolution_m`, and per-product width, height, pixel size, nodata value, transform, and status fields (`hr_2_5m_*` and `hr_1m_*`). The indexes also include exact S2 and NEON asset provenance, acquisition times, LR source/grid fields, land-cover metadata, and projected/geographic centroids. The historical `neon_10m_linearized` product is NEON-derived and is not the canonical LR.

Load the lightweight metadata index without downloading all GeoTIFFs:

```python
from datasets import load_dataset

index = load_dataset("isp-uv-es/SEN2NEON", split="validation")
print(index[0]["lr"])
print(index[0]["hr"])             # canonical 2.5 m alias
print(index[0]["hr_1m_path"])     # supplementary 1 m path
```

The path fields remain relative paths because PIL cannot faithfully decode these 12-band GeoTIFFs. Use `rasterio` through the PyTorch loader below, or download individual paths with `huggingface_hub.hf_hub_download`.

---

## 3. Working with the Dataset API

### Basic Dataset

```python
from data.dataset import SEN2NEON
import torch

DATA_ROOT = "./data/sen2neon"
CSV_PATH = f"{DATA_ROOT}/metadata.csv"

ds = SEN2NEON(
    csv_path=CSV_PATH,
    root_dir=DATA_ROOT,
    hr_resolution=2.5,       # canonical paper product (default); use 1 for supplementary HR
    crop_size_lr=128,        # None keeps full tiles; value in LR pixels
    dtype=torch.float32,
    allow_nan=False,
)

sample = ds[0]
print(sample["lr"].shape, sample["hr"].shape)
print(sample["meta"])       # dict with name, lon/lat, land-cover labels
```

To use the supplementary product, download it and set `hr_resolution=1`. Full-tile shapes are `[12, 256, 256]` for LR and `[12, 2560, 2560]` for 1 m HR. With the default `hr_resolution=2.5`, the HR shape is `[12, 1024, 1024]`.

- **Scaling:** GeoTIFFs store Sentinel-2 reflectance scaled by 10 000 (uint16). Convert to `[0,1]` by dividing by 10 000 before feeding a model.
- **Cropping:** `crop_size_lr` performs aligned random crops, automatically scaling the HR crop to match the LR patch.
- **Metadata:** Land-cover detail/superclass IDs and human-readable labels are included for stratified evaluation.

### PyTorch Lightning-style DataModule

```python
from data.datamodule import SEN2NEONDataModule

datamodule = SEN2NEONDataModule(
    csv_path=CSV_PATH,
    root_dir=DATA_ROOT,
    hr_resolution=2.5,       # canonical default; 1 selects supplementary HR
    batch_size=4,
    num_workers=8,
    crop_size_lr=128,
)

datamodule.setup(stage="predict")
predict_loader = datamodule.predict_dataloader()
for batch in predict_loader:
    lr = batch["lr"].float()
    hr = batch["hr"].float()
    break
```

### Visualization helpers

```python
ds.save_example(out_path="example.png", seed=123)
```

This saves a 2×2 panel containing RGB and random-band comparisons between LR and HR tiles.

---

## 4. Validation pipeline

`validate.py` demonstrates how to reproduce the benchmarking experiments. Reproduction of the paper results uses `hr_resolution=2.5`; the supplied baseline models and reported metrics target that canonical grid, not the supplementary 1 m product.

1. **Load data:** instantiate `SEN2NEONDataModule`, call `setup()`, and iterate over the prediction loader.
2. **Select models:** `models/model_selector.py` exposes adapters for `srgan`, `sen2sr`, `lite_sen2sr`, and `ldsrs2`, defining their input bands and prediction calls.
3. **Run inference:** normalize inputs to `[0,1]`, slice the requested bands, and call the adapter’s `predict` function in `torch.no_grad()` mode.
4. **Post-process:** select the SR band subset, apply histogram matching to the LR reference, and ensure SR/HR tensors align.
5. **Score results:** `metrics.validator.SREvaluator` computes PSNR, SSIM, SAM, and OpenSR-Test metrics; `MetricsSink` logs CSV summaries while optional quick-look figures are saved to disk.

To evaluate your own model, add a new entry to `models_configs` describing the required input bands, the index positions of the 20 m outputs, and a callable that performs inference. The rest of the loop remains unchanged.

---

## 5. Benchmark performance

The paper reports the following validation metrics on the SEN2NEON evaluation split (mean ± std):

| Model            | PSNR ↑ | SSIM ↑ | SAM ↓ | Refl. Consistency ↓ | Spectral Consistency ↓ | Spatial Consistency ↓ |
|------------------|-------:|-------:|------:|---------------------:|------------------------:|-----------------------:|
| S2SR-LDSR-S2     | 31.98 ± 4.21 | 0.78 ± 0.12 | 0.05 ± 0.03 | 0.0087 ± 0.000 | 2.20 ± 1.33 | 0.019 ± 0.05 |
| **S2SR-Lite**    | **32.56 ± 3.28** | 0.80 ± 0.10 | **0.04 ± 0.02** | **0.0054 ± 0.000** | **1.52 ± 0.91** | **0.004 ± 0.02** |
| S2SR-Mamba       | 32.44 ± 4.18 | **0.81 ± 0.10** | 0.05 ± 0.03 | 0.0081 ± 0.000 | 2.20 ± 1.25 | 0.006 ± 0.04 |
| SRGAN            | 31.40 ± 3.85 | 0.73 ± 0.13 | 0.05 ± 0.03 | 0.0067 ± 0.000 | **1.21 ± 0.79** | 0.020 ± 0.03 |

- Frequency-transfer approaches (SEN2SR variants) deliver the best balance between sharpness and spectral fidelity.
- Diffusion-based S2SR-LD introduces additional detail but at the cost of hallucinations, while SRGAN favors perceptual sharpness with weaker physical consistency.

---

## 6. Land-cover-aware analysis

Each metadata row carries detailed (`LC_detail_*`) and superclass (`LC_superclass_*`) labels derived from a categorical land-cover raster. Use them to filter tiles, compute stratified metrics, or balance batches across ecosystems. Missing values are stored as empty strings/NaNs and can be dropped or imputed downstream.

---

## 7. Roadmap

- [x] Release dataset & dataloaders
- [x] Integrate land-cover annotations
- [x] Provide visualization utilities
- [x] Add SEN2SR / LDSR-S2 / SRGAN adapters
- [x] Publish benchmarking pipeline and metrics integration

---

## Citation

If you use SEN2NEON in your research, please cite:

```
@article{article,
author = {Donike, Simon and Aybar, Cesar and Contreras, Julio and Gómez-Chova, Luis},
year = {2026},
month = {01},
pages = {6013905-6013905},
title = {SEN2NEON: Enabling Quantitative Benchmarking of Sentinel-2 Superresolution for All Multispectral Bands},
volume = {23},
journal = {IEEE Geoscience and Remote Sensing Letters},
doi = {10.1109/LGRS.2026.3703947}
}```
