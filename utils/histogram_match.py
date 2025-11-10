import numpy as np
import torch
from skimage.exposure import match_histograms
from skimage.transform import resize


def hist_match_to_reference(target, reference):
    """
    Histogram-match `target` (e.g. SR) to `reference` (e.g. LR) band-wise.

    - Spatially upsamples `reference` to `target`'s H,W.
    - Matches each band of `target` to the corresponding band of upsampled `reference`.
    - Supports batched inputs.

    Args:
        target:    torch.Tensor or np.ndarray
                   shape (C,H,W) or (B,C,H,W)
        reference: same type, shape (C,h,w) or (B,C,h,w).
                   If B_ref == 1 and B_tgt > 1, reference is broadcast over batch.

    Returns:
        matched: same type as `target`, shape as `target`
    """
    # --- type handling ---
    tgt_is_tensor = isinstance(target, torch.Tensor)
    ref_is_tensor = isinstance(reference, torch.Tensor)

    if tgt_is_tensor:
        tgt_device = target.device
        tgt_dtype  = target.dtype
        tgt_np = target.detach().cpu().numpy()
    else:
        tgt_np = np.asarray(target)
        tgt_dtype = tgt_np.dtype
        tgt_device = None

    if ref_is_tensor:
        ref_np = reference.detach().cpu().numpy()
    else:
        ref_np = np.asarray(reference)

    # --- ensure batch dim ---
    if tgt_np.ndim == 3:
        tgt_np = tgt_np[None, ...]   # (1,C,H,W)
    elif tgt_np.ndim != 4:
        raise ValueError("`target` must be (C,H,W) or (B,C,H,W)")

    if ref_np.ndim == 3:
        ref_np = ref_np[None, ...]
    elif ref_np.ndim != 4:
        raise ValueError("`reference` must be (C,h,w) or (B,C,h,w)")

    B_t, C_t, H_t, W_t = tgt_np.shape
    B_r, C_r, H_r, W_r = ref_np.shape

    if C_t != C_r:
        raise ValueError(f"Channel mismatch: target has {C_t}, reference has {C_r}")

    # broadcast reference over batch if needed
    if B_r == 1 and B_t > 1:
        ref_np = np.repeat(ref_np, B_t, axis=0)
        B_r = B_t

    if B_t != B_r:
        raise ValueError(f"Batch mismatch: target B={B_t}, reference B={B_r}")

    # --- upsample reference to target size & hist-match target to it ---
    out = np.empty((B_t, C_t, H_t, W_t), dtype=np.float32)

    for b in range(B_t):
        for c in range(C_t):
            tgt_band = tgt_np[b, c]      # (H_t, W_t)
            ref_band = ref_np[b, c]      # (H_r, W_r)

            # upsample reference band to target size
            ref_up = resize(
                ref_band,
                (H_t, W_t),
                order=1,              # bilinear
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            )

            # match target band to upsampled reference band
            out[b, c] = match_histograms(
                tgt_band,
                ref_up,
                channel_axis=None
            ).astype(np.float32)

    # --- drop batch dim if original target was 3D ---
    if target.ndim == 3:
        out = out[0]

    # --- back to original type ---
    if tgt_is_tensor:
        out_t = torch.from_numpy(out).to(tgt_device)
        out_t = out_t.to(tgt_dtype)
        return out_t
    else:
        return out.astype(tgt_dtype, copy=False)
