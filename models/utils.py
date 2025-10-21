import torch
import torch.nn.functional as F
import numpy as np
from skimage import exposure


def normalise_s2(im,stage="norm"):
    assert stage in ["norm","denorm"]
    value = 3.
    if stage == "norm":
        im = im*(10./value)
        im = (im*2)-1
        im = torch.clamp(im,-1,1)
    if stage=="denorm":
        im = (im+1)/2
        im = im*(value/10.)
        im = torch.clamp(im,0,1)
    return(im)

def normalise_10k(im,stage="norm"):
    assert stage in ["norm","denorm"]
    if stage == "norm":
        im = (im/10000.)
        im = torch.clamp(im,0,1)
    if stage=="denorm":
        im = im*10000.
        im = torch.clamp(im,0,10000)
    return(im)



# HISTOGRAM MATCHING
def histogram_match(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Per-channel histogram match of `target` -> `reference`.

    Supports (C,H,W) or (B,C,H,W).  
    If batch size differs, reference with B=1 is broadcast; else paired by batch.  
    Number of channels must be the same for reference and target.

    Returns tensor on target's device/dtype.
    """
    assert target.ndim in (3,4) and reference.ndim in (3,4), "Expected (C,H,W) or (B,C,H,W)"
    device, dtype = target.device, target.dtype

    # normalize to BCHW
    ref = reference.unsqueeze(0) if reference.ndim == 3 else reference
    tgt = target.unsqueeze(0) if target.ndim == 3 else target

    B_ref, C_ref, H_ref, W_ref = ref.shape
    B_tgt, C_tgt, H_tgt, W_tgt = tgt.shape
    assert C_ref == C_tgt, f"channel mismatch: reference={C_ref}, target={C_tgt}"

    # resize reference to target spatial size (bilinear, no align_corners)
    if (H_ref, W_ref) != (H_tgt, W_tgt):
        ref = F.interpolate(ref.to(dtype=torch.float32), size=(H_tgt, W_tgt), mode="bilinear", align_corners=False)

    # numpy buffers
    ref_np = ref.detach().cpu().numpy()
    tgt_np = tgt.detach().cpu().numpy()
    out_np = np.empty_like(tgt_np)

    for b in range(B_tgt):
        rb = b % B_ref  # broadcast if reference has B=1
        for c in range(C_tgt):
            ref_ch = ref_np[rb, c]
            tgt_ch = tgt_np[b, c]

            mask = np.isfinite(tgt_ch) & np.isfinite(ref_ch)
            if mask.any():
                matched = exposure.match_histograms(tgt_ch[mask], ref_ch[mask])
                out = tgt_ch.copy()
                out[mask] = matched
                out_np[b, c] = out
            else:
                out_np[b, c] = tgt_ch

    out = torch.from_numpy(out_np).to(device=device, dtype=dtype)
    return out[0] if target.ndim == 3 else out
