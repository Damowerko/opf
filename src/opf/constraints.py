import torch
from math import log, pi


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


def fix_angle(u: torch.Tensor):
    u = torch.fmod(u, 2 * pi)
    u[u > 2 * pi] = 2 * pi - u[u > 2 * pi]
    return u


def _compute_mask(mask, constraint):
    if mask is None:
        mask = torch.isfinite(constraint)
    else:
        mask = mask & torch.isfinite(constraint)
    return mask


def truncated_log(u, s, t):
    threshold = -1 / (s * t)
    return torch.where(
        u <= threshold,
        -torch.log(-u) / t,
        (-log(-threshold) / t) + (u - threshold) * s,
    )


def equality(x, y, eps=1e-4, angle=False):
    u = (x - y).abs()
    if angle:
        u = fix_angle(u)
    loss = u.square().mean()

    # normalize the value to get a sense of scale
    normalization = torch.mean(torch.stack((x, y)).abs())
    # Do not divide by zero
    if normalization > eps:
        u = u / normalization
    else:
        print("Avoided dividing by zero.")

    assert not torch.isnan(u).any()
    assert not torch.isinf(u).any()
    return metrics(loss, u, eps)


def inequality(value, lower_bound, upper_bound, s, t, eps=1e-4, angle=False):
    # To properly normalize the results we do not want any of these to be inf.
    assert not torch.isinf(upper_bound).any()
    assert not torch.isinf(lower_bound).any()

    band = (upper_bound - lower_bound).abs()

    mask_equality = band < eps
    mask_lower = _compute_mask(~mask_equality, lower_bound)
    mask_upper = _compute_mask(~mask_equality, upper_bound)

    u_equal = (lower_bound[mask_equality] - value[:, mask_equality]).abs()
    u_lower = lower_bound[mask_lower] - value[:, mask_lower]
    u_upper = value[:, mask_upper] - upper_bound[mask_upper]

    if angle:
        u_equal = fix_angle(u_equal)
        u_lower = fix_angle(u_lower)
        u_upper = fix_angle(u_upper)

    u_lower /= band[mask_lower]
    u_upper /= band[mask_upper]

    # we normalize u_equal by the mean difference between the upper and lower constraints
    # this allows us to compare values of different scales
    u_inequality = torch.cat((u_lower.flatten(), u_upper.flatten()))
    loss = u_equal.square().sum() + truncated_log(u_inequality, s, t).sum()
    loss /= value.numel()

    # The tensor elements should be bounded.
    assert not torch.isinf(u_equal).any()
    assert not torch.isnan(u_equal).any()
    assert not torch.isinf(u_inequality).any()
    assert not torch.isnan(u_inequality).any()
    return metrics(loss, torch.cat((u_equal.flatten() / band[~mask_equality].mean(), u_inequality)), eps)
