import pandas as pd
import torch
import numpy as np  # ← add this
from datetime import datetime
import os
from typing import Any, Dict, List
from typing import Tuple


def _to_python(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
        return x.tolist()
    # NEW: handle numpy arrays
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.reshape(-1)[0])
        return x.tolist()
    return x


def _meta_item(meta: Dict[str, Any], i: int) -> Dict[str, Any]:
    out = {}
    for k, v in meta.items():
        v = _to_python(v)
        if isinstance(v, (list, tuple)):
            out[k] = v[i]
        else:
            out[k] = v
    return out


def _normalize_scalar_or_vector(v: Any, B: int) -> List[float]:
    """Return list length B of floats (no band dimension)."""
    v = _to_python(v)
    if isinstance(v, (list, tuple)):
        if len(v) == B:
            return [float(x) for x in v]
        if len(v) == 1:
            return [float(v[0])] * B
        raise ValueError(f"Metric length {len(v)} != batch size {B}")
    return [float(v)] * B


def _normalize_band_metric(v: Any, B: int) -> Tuple[List[List[float]], int]:
    """
    For metrics with shape (B, C_bands). Returns:
      - per_item: list of length B, each elem is list length C_bands
      - C_bands: number of bands
    """
    v = _to_python(v)
    if not isinstance(v, (list, tuple)) or len(v) != B:
        raise ValueError(
            f"Band metric expected shape (B, C), got {type(v)} with len={len(v) if isinstance(v, (list, tuple)) else 'n/a'}"
        )

    # each row should be list/tuple of length C
    if not isinstance(v[0], (list, tuple)):
        raise ValueError(
            "Band metric expected inner lists/tuples per item, e.g. shape (B, C)."
        )

    C = len(v[0])
    per_item = []
    for row in v:
        if len(row) != C:
            raise ValueError("Inconsistent band dimension across items.")
        per_item.append([float(x) for x in row])

    return per_item, C


class MetricsSink:
    def __init__(self, save_dir: str, run_id: str | None = None):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self._records: List[Dict[str, Any]] = []

    def log_batch(
        self,
        model_name: str,
        batch_idx: int,
        meta: Dict[str, Any],
        metrics: Dict[str, Any],
    ):
        # infer batch size
        key0 = "name" if "name" in meta else next(iter(meta.keys()))
        B = len(_to_python(meta[key0]))

        # pre-process metrics: detect bandwise metrics (B, C) vs scalar/vector (B,)
        scalar_metrics: Dict[str, List[float]] = {}
        band_metrics: Dict[str, List[List[float]]] = {}
        band_dims: Dict[str, int] = {}

        for m, v in metrics.items():
            v_py = _to_python(v)

            # bandwise: (B, C) -> list of lists
            if (
                isinstance(v_py, (list, tuple))
                and len(v_py) == B
                and isinstance(v_py[0], (list, tuple))
            ):
                per_item, C = _normalize_band_metric(v_py, B)
                band_metrics[m] = per_item
                band_dims[m] = C
            else:
                scalar_metrics[m] = _normalize_scalar_or_vector(v_py, B)

        # one row per item
        for i in range(B):
            row = {
                "run_id": self.run_id,
                "model": model_name,
                "batch_idx": batch_idx,
                "item_idx": i,
            }
            row.update(_meta_item(meta, i))

            # scalars / single value per item
            for m, arr in scalar_metrics.items():
                row[m] = arr[i]

            # bandwise metrics: add columns like ssim_b0, ssim_b1, ...
            for m, per_item in band_metrics.items():
                C = band_dims[m]
                bands_i = per_item[i]  # list length C
                for b in range(C):
                    row[f"{m}_b{b}"] = bands_i[b]

            self._records.append(row)

    def flush(self, basename: str = "val_metrics", clear: bool = False):
        if not self._records:
            return None
        df = pd.DataFrame(self._records)
        csv_path = os.path.join(self.save_dir, f"{basename}.csv")
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, index=False, mode="a", header=header)
        if clear:
            self._records.clear()
        return {"csv": csv_path, "rows": len(df)}
