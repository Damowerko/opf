from math import log, pi
import torch

def metrics(loss, u: torch.Tensor, eps: float):
    # Consider two values equal if within numerical error.
    violated_mask = u >= eps
    violated = u[violated_mask]

    if violated.numel() == 0:
        violated = torch.zeros(1, device=violated.device, dtype=violated.dtype)

    assert u.numel() > 0
    # if u.numel() == 0:
    #     loss = torch.zeros(1, device=u.device, dtype=u.dtype).squeeze()

    return dict(
        loss=loss,
        rate=(violated_mask.sum() / violated_mask.numel()).nan_to_num(),
        error_mean=violated.abs().mean(),
        error_max=violated.abs().max(),
    )

def equality(value, target, eps):
    u = (value - target).abs()
    loss = u.square().mean()
    return metrics(loss, u, eps)

def inequality(value, low_bound, high_bound, eps):
    # definitely wrong because all these values are complex
    if value >= high_bound:
        u = (value - high_bound).abs()
    elif value <= low_bound:
        u = (value - high_bound).abs()
    else:
        u = torch.zeros_like(value)
    loss = u.square().mean()
    return metrics(loss, u, eps)