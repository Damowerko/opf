import torch


def metrics(loss, u: torch.Tensor, eps: float, multiplier: torch.Tensor | None):
    # Consider two values equal if within numerical error.
    violated_mask = u >= eps
    violated = u[violated_mask]

    if violated.numel() == 0:
        violated = torch.zeros(1, device=violated.device, dtype=violated.dtype)

    assert u.numel() > 0
    # if u.numel() == 0:
    #     loss = torch.zeros(1, device=u.device, dtype=u.dtype).squeeze()

    metrics = dict(
        rate=(violated_mask.sum() / violated_mask.numel()).nan_to_num(),
        error_mean=violated.abs().mean(),
        error_max=violated.abs().max(),
    )
    if loss is not None:
        metrics["loss"] = loss
    if multiplier is not None:
        metrics["multiplier/mean"] = multiplier.mean()
        metrics["multiplier/max"] = multiplier.max()
        metrics["multiplier/min"] = multiplier.min()
    return metrics


def wrap_angle(u: torch.Tensor):
    """
    Wrap angle to the range [-pi, pi].
    """
    return torch.remainder(u + torch.pi, 2 * torch.pi) - torch.pi


def _compute_mask(mask, constraint):
    if mask is None:
        mask = torch.isfinite(constraint)
    else:
        mask = mask & torch.isfinite(constraint)
    return mask


def loss_equality(u: torch.Tensor, multiplier: torch.Tensor, augmented_weight=0.0):
    loss = (u.unsqueeze(1) @ multiplier.unsqueeze(2)).squeeze(1, 2).mean(dim=0)
    if augmented_weight > 0:
        loss = loss + augmented_weight * u.pow(2).sum(dim=-1).mean(dim=0)
    return loss


def loss_inequality(u: torch.Tensor, multiplier: torch.Tensor, augmented_weight=0.0):
    loss = (u.unsqueeze(1) @ multiplier.unsqueeze(2)).squeeze(1, 2).mean(dim=0)
    # u <= 0, therefore we have violation when u > 0, loss = max(0, u)^2
    if augmented_weight > 0:
        loss = loss + augmented_weight * u.relu().pow(2).sum(dim=-1).mean(dim=0)
    return loss


def equality(
    x: torch.Tensor,
    y: torch.Tensor,
    multiplier: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    eps=1e-4,
    angle=False,
    augmented_weight=0.0,
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
        augmented_weight (float, optional): The weight of the augmented loss. Defaults to `0.0`.

    Returns:
        A tuple containing the loss value, the constraint violation vector, and the number of violated constraints.
    """
    if mask is not None:
        x = x[..., mask]
        y = y[..., mask]
        if multiplier is not None:
            multiplier = multiplier[..., mask]
    u = x - y if not angle else wrap_angle(x - y)
    # The loss is the dot product of the constraint and the multiplier, averaged over the batch.
    loss = (
        loss_equality(u, multiplier, augmented_weight=augmented_weight)
        if multiplier is not None
        else None
    )

    assert not torch.isnan(u).any()
    assert not torch.isinf(u).any()
    return metrics(loss, u.abs(), eps, multiplier)


def inequality(
    value: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    lower_multiplier: torch.Tensor | None = None,
    upper_multiplier: torch.Tensor | None = None,
    eps=1e-4,
    angle=False,
    augmented_weight=0.0,
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
        augmented_weight (float, optional): The augmented weight. Defaults to 0.0.
    Returns:
        tuple: A tuple containing the loss and the metrics.
    """
    assert (lower_multiplier is None) == (
        upper_multiplier is None
    ), "Both or none of the multipliers should be provided."
    # Inequality multipliers should be positive.
    if lower_multiplier is not None:
        assert torch.all(
            lower_multiplier >= 0
        ), "There are negative values in the inequality multiplier"
    if upper_multiplier is not None:
        assert torch.all(
            upper_multiplier >= 0
        ), "There are negative values in the inequality multiplier"

    # To properly normalize the results we do not want any of these to be inf.
    assert not torch.isinf(upper_bound).any()
    assert not torch.isinf(lower_bound).any()

    mask_lower = _compute_mask(None, lower_bound)
    mask_upper = _compute_mask(None, upper_bound)

    u_lower = lower_bound[mask_lower] - value[:, mask_lower]
    u_upper = value[:, mask_upper] - upper_bound[mask_upper]

    if angle:
        u_lower = wrap_angle(u_lower)
        u_upper = wrap_angle(u_upper)

    u_all = torch.cat((u_lower, u_upper), dim=1)

    if lower_multiplier is None or upper_multiplier is None:
        return metrics(None, u_all, eps, None)

    loss_lower = loss_inequality(
        u_lower, lower_multiplier[:, mask_lower], augmented_weight=augmented_weight
    )
    loss_upper = loss_inequality(
        u_upper, upper_multiplier[:, mask_upper], augmented_weight=augmented_weight
    )
    loss = loss_lower + loss_upper
    # this is used for metrics only
    multiplier_all = torch.cat(
        (
            lower_multiplier[:, mask_lower],
            upper_multiplier[:, mask_upper],
        ),
        dim=1,
    )
    return metrics(loss, u_all, eps, multiplier_all)
