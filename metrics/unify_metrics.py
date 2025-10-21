import pandas as pd
import torch
from datetime import datetime
import os
from typing import Any, Dict, List

def _to_python(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
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

def _normalize_metric_per_item(v: Any, B: int) -> List[float]:
    v = _to_python(v)
    if isinstance(v, (list, tuple)):
        if len(v) == B:
            return list(v)
        if len(v) == 1:
            return [float(v[0])] * B
        raise ValueError(f"Metric length {len(v)} != batch size {B}")
    return [float(v)] * B

class MetricsSink:
    def __init__(self, save_dir: str, run_id: str | None = None):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self._records: List[Dict[str, Any]] = []

    def log_batch(self, model_name: str, batch_idx: int, meta: Dict[str, Any], metrics: Dict[str, Any]):
        key0 = 'name' if 'name' in meta else next(iter(meta.keys()))
        B = len(_to_python(meta[key0]))
        per_item_metrics = {m: _normalize_metric_per_item(v, B) for m, v in metrics.items()}

        for i in range(B):
            row = {"run_id": self.run_id, "model": model_name, "batch_idx": batch_idx, "item_idx": i}
            row.update(_meta_item(meta, i))
            for m, arr in per_item_metrics.items():
                row[m] = arr[i]
            self._records.append(row)

    def flush(self, basename: str = "val_metrics", clear: bool = False):
        if not self._records:
            return None
        df = pd.DataFrame(self._records)
        csv_path = os.path.join(self.save_dir, f"{basename}.csv")
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, index=False, mode='a', header=header)
        if clear:
            self._records.clear()
        return {"csv": csv_path, "rows": len(df)}

# Example usage:
# sink = MetricsSink("logs/metrics")
# ...
# sink.log_batch(model_name, batch_idx, batch["meta"], metrics)
# sink.flush(f"val_metrics_{model_name}", clear=True)
