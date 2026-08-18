<p align="center">
  <img src="resources/sen2neon_banner.png" alt="SEN2NEON example" width="100%">
</p>


<img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/opensr_logo.png?raw=true" width="250"/>



# SEN2NEON - Dataset for Multispectral Sentinel-2 Super-Resolution 

SEN2NEON is a PyTorch-based dataset and evaluation suite for the **SEN2NEON multispectral super-resolution benchmark**. The project aligns Sentinel-2 Level-2A reflectances with co-registered 1 m AVIRIS-NG hyperspectral imagery, creating the first large-scale reference set for the Sentinel-2 20 m bands under real reflectance conditions.

## Highlights

- **Harmonized dataset** with 2,269 spatially aligned tiles (1024×1024 pixels at 2.5 m) spanning 12 Sentinel-2-equivalent bands (B1–B9, B8A, B11–B12).
- **Land-cover aware metadata** (centroids, fine/coarse classes) for stratified analyses.
- **Ready-to-use PyTorch Dataset/DataModule** that pairs LR (10 m) and HR (2.5 m) GeoTIFF tiles with consistent scaling and NaN handling.
- **Validation pipeline** with visualization, histogram matching, and metric logging that integrates OpenSR-Test semantics.
- **Baseline model adapters** for SEN2SR variants, LDSR-S2, and SRGAN to reproduce the benchmarks from the accompanying paper.

---

## Dataset at a glance

- **Source data:** 1 m AVIRIS-NG hyperspectral imagery harmonized to Sentinel-2A/B spectral response functions, excluding the cirrus band (B10) which is not part of Level-2A reflectance products.
- **Reference products:** Bilinearly downsampled tiles at 2.5 m (1024×1024) for 8× SR and at 10 m (256×256) for 2× SR experiments, stored as 16-bit integers scaled by 10 000.
- **Coverage:** 2,269 tiles spanning 127 acquisition dates (2017–2024) and diverse U.S. land-cover types, including forests, grasslands, shrublands, croplands, water, barren land, and built-up areas.
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
pip install pandas numpy rasterio matplotlib tqdm huggingface_hub hf_transfer opensr_test
```

> `hf_transfer` accelerates downloads (optional). Install GPU-specific PyTorch wheels as needed.

---

## 2. Download the dataset from Hugging Face

The dataset is hosted on the Hugging Face Hub at [`simon-donike/SEN2NEON`](https://huggingface.co/datasets/simon-donike/SEN2NEON). Use the helper script to keep the on-hub directory layout:

```bash
python data/download_SEN2NEON.py \
  --repo-id simon-donike/SEN2NEON \
  --out-dir ./data/sen2neon \
  --use-hf-transfer      # optional, enables HF_HUB_ENABLE_HF_TRANSFER
```

Key options:

- `--all`: download every file in the repository instead of the curated subset.
- `--no-symlinks`: force real copies instead of symlinks (useful on network storage).
- Set `HF_TOKEN` if you need to authenticate to private mirrors.

After the command finishes you should see the following structure:

```
./data/sen2neon
├── metadata.jsonl
├── neon_10m_linearized/        # 256×256 LR GeoTIFFs (scaled by 10 000)
├── neon_2.5m_linearized/       # 1024×1024 HR GeoTIFFs (scaled by 10 000)
└── sen2neon_metadata.csv
```

The CSV contains one row per tile with relative paths, land-cover metadata, projected and geographic centroids, and (optionally) split labels.

---

## 3. Working with the Dataset API

### Basic Dataset

```python
from data.dataset import SEN2NEON
import torch

DATA_ROOT = "./data/sen2neon"
CSV_PATH = f"{DATA_ROOT}/sen2neon_metadata.csv"

ds = SEN2NEON(
    csv_path=CSV_PATH,
    root_dir=DATA_ROOT,
    crop_size_lr=128,        # None keeps full tiles; value in LR pixels
    dtype=torch.float32,
    allow_nan=False,
)

sample = ds[0]
print(sample["lr"].shape, sample["hr"].shape)
print(sample["meta"])       # dict with name, lon/lat, land-cover labels
```

- **Scaling:** GeoTIFFs store Sentinel-2 reflectance scaled by 10 000 (uint16). Convert to `[0,1]` by dividing by 10 000 before feeding a model.
- **Cropping:** `crop_size_lr` performs aligned random crops, automatically scaling the HR crop to match the LR patch.
- **Metadata:** Land-cover detail/superclass IDs and human-readable labels are included for stratified evaluation.

### PyTorch Lightning-style DataModule

```python
from data.datamodule import SEN2NEONDataModule

datamodule = SEN2NEONDataModule(
    csv_path=CSV_PATH,
    root_dir=DATA_ROOT,
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

`validate.py` demonstrates how to reproduce the benchmarking experiments:

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

## Compare Hugging Face LR to Planetary Computer Sentinel-2

Standalone check that a `neon_10m_linearized` tile matches same-date Sentinel-2 L2A from Microsoft Planetary Computer. It downloads the HF sample if needed, warps PC S2 onto the HF grid, and writes metrics plus RGB figures (LR vs PC, zoom, LR/HR/PC).

```bash
pip install numpy rasterio matplotlib pandas pystac-client planetary-computer huggingface_hub

python compare_sen2neon_lr_vs_planetary_computer.py \
  --sample-id 2018_MLBS_3__1_1 \
  --out-dir ./out
```

Default sample `2018_MLBS_3__1_1` has no nodata holes. Override with `--sample-id`. Use `--no-download` if the GeoTIFFs are already local. Needs network (Hugging Face, GitHub metadata, Planetary Computer).

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
