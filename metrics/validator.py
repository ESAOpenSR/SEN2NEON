import torch
import opensr_test
import lpips
from kornia.metrics import psnr, ssim


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
        Compute standard metrics (PSNR, SSIM, RMSE) for one SR batch.

        Args:
            lr: low-resolution tensor, shape [C,H,W] or [B,C,H,W]
            sr: super-resolved tensor, shape [C,H,W] or [B,C,H,W]
            hr: high-resolution tensor, shape [C,H,W] or [B,C,H,W]
        Returns:
            dict with standard metrics (PSNR, SSIM, LPIPS)
        """        
        results = {}
        # PSNR
        results['PSNR'] = psnr(sr, hr,max_val=1.).item()
        # SSIM
        results['SSIM'] = ssim(sr.unsqueeze(0), hr.unsqueeze(0),max_val=1.,window_size=11).mean().item()
        
        return results
    
    def evaluate(self, lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor) -> dict:
        """
        Compute all metrics for one SR sample.

        Args:
            lr: low-resolution tensor, shape [C,H,W]
            sr: super-resolved tensor, shape [C,H,W]
            hr: high-resolution tensor, shape [C,H,W]

        Returns:
            dict with all metrics
        """
        metrics = {}
        metrics.update(self.opensr_metrics(lr, sr, hr))
        metrics.update(self.compute_normal_metrics(lr, sr, hr))
        return metrics

if __name__ == "__main__":
    # create once
    evaluator = SREvaluator()

    # your data
    lr = torch.rand(4, 64, 64)
    hr = torch.rand(4, 256, 256)
    sr = torch.rand(4, 256, 256)

    # compute
    opensr_metrics = evaluator.opensr_metrics(lr, sr, hr)
    normal_metrics = evaluator.compute_normal_metrics(lr, sr, hr)
    all_metrics = evaluator.evaluate(lr, sr, hr)
