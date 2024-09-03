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


def wrap_angle(u: torch.Tensor):
    """
    Wrap angle to the range [-pi, pi].
    """
    return torch.fmod(u, torch.pi)


def _compute_mask(mask, constraint):
    if mask is None:
        mask = torch.isfinite(constraint)
    else:
        mask = mask & torch.isfinite(constraint)
    return mask


def equality(
    x: torch.Tensor,
    y: torch.Tensor,
    multiplier: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps=1e-4,
    angle=False,
):
    """
    Computes the equality constraint between two tensors `x` and `y`.

    Args:
        x (torch.Tensor): The first tensor.
        y (torch.Tensor): The second tensor.
        multiplier (torch.Tensor): A multiplier value that can be any real number.
        mask (torch.Tensor | None, optional): A boolean mask tensor. If provided, only the elements where `mask` is `True` will be considered. Defaults to `None`.
        eps (float, optional): A small constant used to avoid division by zero. Defaults to `1e-4`.
        angle (bool, optional): If `True`, the input tensors are treated as angles in radians and the difference is wrapped to the range `[-pi, pi]`. Defaults to `False`.

    Returns:
        A tuple containing the loss value, the constraint violation vector, and the number of violated constraints.
    """
    if mask is not None:
        x = x[..., mask]
        y = y[..., mask]
        multiplier = multiplier[..., mask]
    u = x - y if not angle else wrap_angle(x - y)
    # The loss is the dot product of the constraint and the multiplier, averaged over the batch.
    loss = (u @ multiplier).mean(dim=0)

    assert not torch.isnan(u).any()
    assert not torch.isinf(u).any()
    return metrics(loss, u, eps, multiplier)


def inequality(
    value: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    lower_multiplier: torch.Tensor,
    upper_multiplier: torch.Tensor,
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
    # Inequality multipliers should be positive.
    assert torch.all(lower_multiplier >= 0) and torch.all(
        upper_multiplier >= 0
    ), "There are negative values in the inequality multiplier"
    # To properly normalize the results we do not want any of these to be inf.
    assert not torch.isinf(upper_bound).any()
    assert not torch.isinf(lower_bound).any()

    band = (upper_bound - lower_bound).abs()
    mask_equality = band < eps
    mask_lower = _compute_mask(~mask_equality, lower_bound)
    mask_upper = _compute_mask(~mask_equality, upper_bound)

    target = (upper_bound[mask_equality] + lower_bound[mask_equality]) / 2
    u_equal = target - value[:, mask_equality]
    u_lower = lower_bound[mask_lower] - value[:, mask_lower]
    u_upper = value[:, mask_upper] - upper_bound[mask_upper]

    if angle:
        u_equal = wrap_angle(u_equal)
        u_lower = wrap_angle(u_lower)
        u_upper = wrap_angle(u_upper)

    u_lower /= band[mask_lower]
    u_upper /= band[mask_upper]
    u_all = torch.cat((u_lower, u_upper, u_equal), dim=1)
    multiplier_all = torch.cat(
        (
            lower_multiplier[mask_lower],
            upper_multiplier[mask_upper],
            # We represent the equality multiplier (which can be any real number) as the difference between two positive numbers.
            upper_multiplier[mask_equality] - lower_multiplier[mask_equality],
        ),
        dim=0,
    )
    # The loss is the dot product of the constraint and the multiplier, averaged over the batch.
    loss = (u_all @ multiplier_all).mean(dim=0)
    return metrics(loss, u_all, eps, multiplier_all)
