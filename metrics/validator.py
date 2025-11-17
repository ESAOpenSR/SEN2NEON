import torch
import opensr_test
import lpips
from kornia.metrics import psnr, ssim
import numpy as np
from typing import Tuple


class SREvaluator:
    """
    Wraps opensr_test.Metrics for repeated super-resolution evaluation.
    """

    def __init__(self, **kwargs):
        """
        Initialize evaluator.
        kwargs are passed to opensr_test.Metrics() constructor.
        """
        self.metrics = opensr_test.Metrics(**kwargs)

    def opensr_metrics(self, lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor) -> dict:
        """
        Compute metrics for one SR batch.
        
        Args:
            lr: low-resolution tensor, shape [C,H,W] or [B,C,H,W]
            sr: super-resolved tensor, shape [C,H,W] or [B,C,H,W]
            hr: high-resolution tensor, shape [C,H,W] or [B,C,H,W]

        Returns:
            dict with metrics (reflectance, spectral, spatial, synthesis, etc.)
        """
        opensr_metrics = self.metrics.compute(lr=lr, sr=sr, hr=hr)
        return opensr_metrics
    
    def compute_normal_metrics(self, lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor) -> dict:
        """
        Compute standard metrics (PSNR, SSIM, SAM) for one SR batch.
        Assumes inputs in [0,1].

        Returns:
            dict with:
              - PSNR, SSIM, SAM (scalar over all bands)
              - PSNR_b{i}, SSIM_b{i} per band i
        """
        sr_b = self._ensure_bchw(sr)  # [B,C,H,W]
        hr_b = self._ensure_bchw(hr)  # [B,C,H,W]
        B, C, H, W = sr_b.shape

        # ---- scalar PSNR over all bands ----
        psnr_val = psnr(sr_b, hr_b, max_val=1.0)
        if psnr_val.dim() > 0:
            psnr_val = psnr_val.mean()
        psnr_val = torch.nan_to_num(psnr_val, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- scalar SSIM over all bands ----
        ssim_val = ssim(sr_b, hr_b, max_val=1.0, window_size=11)
        if ssim_val.dim() > 0:
            ssim_val = ssim_val.mean()
        ssim_val = torch.nan_to_num(ssim_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        # ---- SAM (scalar) ----
        sam_val = self.sam(sr_b, hr_b)  # already sanitized

        metrics = {
            "PSNR": float(psnr_val),
            "SSIM": float(ssim_val),
            "SAM": float(sam_val),
        }

        # ------------------------------------------------------------------
        # Per-band PSNR / SSIM: PSNR_b0, PSNR_b1, ..., SSIM_b0, SSIM_b1, ...
        # ------------------------------------------------------------------
        for c in range(C):
            # single-band tensors [B,1,H,W]
            sr_c = sr_b[:, c:c+1]
            hr_c = hr_b[:, c:c+1]

            # PSNR per band
            psnr_c = psnr(sr_c, hr_c, max_val=1.0)
            if psnr_c.dim() > 0:
                psnr_c = psnr_c.mean()
            psnr_c = torch.nan_to_num(psnr_c, nan=0.0, posinf=0.0, neginf=0.0)
            metrics[f"PSNR_b{c}"] = float(psnr_c)

            # SSIM per band
            ssim_c = ssim(sr_c, hr_c, max_val=1.0, window_size=11)
            if ssim_c.dim() > 0:
                ssim_c = ssim_c.mean()
            ssim_c = torch.nan_to_num(ssim_c, nan=0.0, posinf=0.0, neginf=0.0)
            metrics[f"SSIM_b{c}"] = float(ssim_c)

        return metrics
    
    @torch.no_grad()
    def evaluate(self, lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor) -> dict:
        """
        Compute all metrics for one SR sample.
        Returns: dict with OpenSR + standard metrics (PSNR, SSIM, SAM)
                 plus per-band PSNR_b{i}, SSIM_b{i}.
        """
        metrics = {}
        metrics.update(self.opensr_metrics(lr, sr, hr))
        metrics.update(self.compute_normal_metrics(lr, sr, hr))

        # final sanitation: replace any lingering NaN/Inf with 0
        cleaned = {}
        for k, v in metrics.items():
            t = torch.tensor(v, dtype=torch.float32)
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            cleaned[k] = float(t)
        return cleaned
    
    def _ensure_bchw(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is [B,C,H,W]."""
        if x.dim() == 3:  # [C,H,W]
            return x.unsqueeze(0)
        elif x.dim() == 4:  # [B,C,H,W]
            return x
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")
    
    @torch.no_grad()
    def sam(self, sr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Spectral Angle Mapper (radians), mean over valid pixels.
        sr, hr: [B,C,H,W] or [C,H,W] in [0,1].
        """
        sr = self._ensure_bchw(sr).to(dtype=torch.float32)
        hr = self._ensure_bchw(hr).to(dtype=torch.float32)

        # Flatten channels as vectors per pixel
        B, C, H, W = sr.shape
        sr_v = sr.view(B, C, -1)  # [B,C,N]
        hr_v = hr.view(B, C, -1)  # [B,C,N]

        dot = (sr_v * hr_v).sum(dim=1)                        # [B,N]
        sr_norm = torch.linalg.norm(sr_v, dim=1)              # [B,N]
        hr_norm = torch.linalg.norm(hr_v, dim=1)              # [B,N]
        denom = sr_norm * hr_norm                             # [B,N]

        valid = denom > eps                                   # mask invalid (both nearly zero)
        safe = torch.zeros_like(denom)
        safe[valid] = dot[valid] / (denom[valid] + eps)

        # Numerical safety for acos
        safe = safe.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        angles = torch.zeros_like(safe)
        angles[valid] = torch.acos(safe[valid])               # radians

        # If no valid pixels in a sample, its mean becomes 0 (safe default)
        per_sample = torch.where(
            valid.any(dim=1),
            angles.sum(dim=1) / (valid.sum(dim=1).clamp(min=1)),
            torch.zeros_like(angles.sum(dim=1))
        )  # [B]

        out = per_sample.mean()                      # radians (tensor scalar)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return float(out.item())
