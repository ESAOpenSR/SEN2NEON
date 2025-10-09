import os, re
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.SRGAN.utils import normalise_10k
from utils.stretching import convention_stretch

@torch.no_grad()
def save_batch_visualizations(
    lr, sr, hr, meta: dict, model_name: str,
    out_root: str = "visualizations/inf",
    bands=None, max_per_batch=None, name_key: str = "name"
):
    """
    Saves one PNG per sample:
    visualizations/inf/{model_name}_{ID_wo_ext}.png

    Args:
        lr, sr, hr: torch.Tensors (B,C,H,W).
        meta: dict with a 'name' column (list-like per batch).
        model_name: str, added to filename.
    """
    os.makedirs(out_root, exist_ok=True)

    # bring to 0..1
    if lr.max() > 2:
        lr = normalise_10k(lr,stage="norm")
        sr = normalise_10k(sr,stage="norm")
        hr = normalise_10k(hr,stage="norm")

    # Stretch for Viz
    lr = convention_stretch(lr)
    sr = convention_stretch(sr)
    hr = convention_stretch(hr)

    # interpolate LR to HR size with NN
    lr = F.interpolate(lr, size=(hr.shape[2], hr.shape[3]), mode="nearest")

    # get 256 middle crop for all
    crop_size = 256
    Hs, Ws = hr.shape[2], hr.shape[3]
    if Hs > crop_size and Ws > crop_size:
        h1 = (Hs - crop_size) // 2
        w1 = (Ws - crop_size) // 2
        hr = hr[:, :, h1:h1+crop_size, w1:w1+crop_size]
        sr = sr[:, :, h1:h1+crop_size, w1:w1+crop_size]
        lr = lr[:, :, h1:h1+crop_size, w1:w1+crop_size]


    B, C, Hs, Ws = sr.shape
    K = min(B, max_per_batch) if isinstance(max_per_batch, int) and max_per_batch > 0 else B

    if bands is None:
        bands = (0, 1, 2) if C >= 3 else tuple(range(min(3, C)))

    def _sanitize(s: str) -> str:
        # drop path + extension, then keep safe chars
        base = os.path.basename(str(s))
        base_no_ext = os.path.splitext(base)[0]
        return re.sub(r"[^\w\-]+", "_", base_no_ext)

    def _get_id(k: int) -> str:
        v = meta.get(name_key, None)
        if v is None:
            return f"noID"
        try:
            val = v[k]
            if isinstance(val, torch.Tensor):
                val = val.item() if val.numel() == 1 else str(val)
            return _sanitize(val)
        except Exception:
            return f"noID"

    def _prep_img(t: torch.Tensor):
        t = t.clamp(0, 1).cpu()
        if t.shape[0] >= 3:
            return t[bands, :, :].permute(1, 2, 0).numpy()
        else:
            return t[0:1, :, :].permute(1, 2, 0).numpy().squeeze(-1)

    for k in range(K):
        mid = _get_id(k)
        fname = f"{model_name}_{mid}.png"
        fpath = os.path.join(out_root, fname)

        lr_k, sr_k, hr_k = lr[k], sr[k], hr[k]
        if lr_k.shape[-2:] != (Hs, Ws):
            lr_k = F.interpolate(lr_k.unsqueeze(0), size=(Hs, Ws), mode="bilinear", align_corners=False).squeeze(0)

        lr_np, sr_np, hr_np = _prep_img(lr_k), _prep_img(sr_k), _prep_img(hr_k)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(lr_np); axes[0].set_title("LR"); axes[0].axis("off")
        axes[1].imshow(sr_np); axes[1].set_title("SR"); axes[1].axis("off")
        axes[2].imshow(hr_np); axes[2].set_title("HR"); axes[2].axis("off")
        plt.suptitle(f"{model_name} • {mid}")
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {fpath}")
