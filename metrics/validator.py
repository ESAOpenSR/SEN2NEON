import torch
import opensr_test

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

    def evaluate(self, lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor) -> dict:
        """
        Compute metrics for one SR batch.
        
        Args:
            lr: low-resolution tensor, shape [C,H,W] or [B,C,H,W]
            sr: super-resolved tensor, shape [C,H,W] or [B,C,H,W]
            hr: high-resolution tensor, shape [C,H,W] or [B,C,H,W]

        Returns:
            dict with metrics (reflectance, spectral, spatial, synthesis, etc.)
        """
        return self.metrics.compute(lr=lr, sr=sr, hr=hr)

        
            
    

if __name__ == "__main__":
    # create once
    evaluator = SREvaluator()

    # your data
    lr = torch.rand(4, 64, 64)
    hr = torch.rand(4, 256, 256)
    sr = torch.rand(4, 256, 256)

    # compute
    results = evaluator.evaluate(lr, sr, hr)
    print(results)