from math import log, pi

import torch


def metrics(loss, u: torch.Tensor, eps: float, multiplier: torch.Tensor):
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
        multiplier_mean=multiplier.mean(),
        multiplier_max=multiplier.max(),
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
    assert not u.isnan().any()
    threshold = -1 / (s * t)
    below = u <= threshold
    v = torch.zeros_like(u)
    v[below] = -torch.log(-u[below]) / t
    v[~below] = (-log(-threshold) / t) + (u[~below] - threshold) * s
    return v


def equality(
    multiplier: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps=1e-4,
    angle=False,
):
    """
    Computes the equality constraint between two tensors `x` and `y`.

    Args:
        x (torch.Tensor): The first tensor.
        y (torch.Tensor): The second tensor.
        mask (torch.Tensor | None, optional): A boolean mask tensor. If provided, only the elements where `mask` is `True` will be considered. Defaults to `None`.
        eps (float, optional): A small constant used to avoid division by zero. Defaults to `1e-4`.
        angle (bool, optional): If `True`, the input tensors are treated as angles in radians and the difference is wrapped to the range `[-pi, pi]`. Defaults to `False`.

    Returns:
        A tuple containing the loss value, the constraint violation vector, and the number of violated constraints.
    """
    
    if mask is not None:
        x = x[..., mask]
        y = y[..., mask]
    u = (x - y).abs()
    if angle:
        u = fix_angle(u)

    loss_tensor = u @ multiplier
    loss = loss_tensor.mean()

    assert not torch.isnan(u).any()
    assert not torch.isinf(u).any()
    return metrics(loss, u, eps, multiplier)


def inequality(
    multiplier: torch.Tensor,
    value: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    eps=1e-4,
    angle=False,
):
    """
    Computes the inequality constraint loss for a given tensor.

    Args:
        multiplier (torch.Tensor): The loss multiplier.
        value (torch.Tensor): The tensor to be constrained.
        lower_bound (torch.Tensor): The lower bound tensor.
        upper_bound (torch.Tensor): The upper bound tensor.
        eps (float, optional): The epsilon value. Defaults to 1e-4.
        angle (bool, optional): Whether to fix the angle. Defaults to False.

    Returns:
        tuple: A tuple containing the loss and the metrics.
    """
    # TODO:
    # Make changes to inequality for equality case

    # To properly normalize the results we do not want any of these to be inf.
    assert not torch.isinf(upper_bound).any()
    assert not torch.isinf(lower_bound).any()

    band = (upper_bound - lower_bound).abs()

    mask_equality = band < eps   

    mask_lower = _compute_mask(~mask_equality, lower_bound)
    mask_upper = _compute_mask(~mask_equality, upper_bound)

    u_equal = lower_bound[mask_equality] - value[:, mask_equality]
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

    loss = ((lower_bound - value)[:, mask_lower] @ multiplier[mask_lower,0]).sum() + ((value - upper_bound)[:, mask_upper] @ multiplier[mask_upper,1]).sum()

    # The tensor elements should be bounded.
    assert not torch.isinf(u_equal).any()
    assert not torch.isnan(u_equal).any()
    assert not torch.isinf(u_inequality).any()
    assert not torch.isnan(u_inequality).any()
    return metrics(
        loss,
        torch.cat((u_equal.flatten() / band[~mask_equality].mean(), u_inequality)),
        eps,
        multiplier,
    )
