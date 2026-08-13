"""
Loss functions for PFPSAM.

The main loss combines a sigmoid binary cross-entropy loss and a Dice loss,
both computed on points sampled according to their uncertainty (PointRend,
Kirillov et al., ICCV 2020). This follows the standard SAM fine-tuning
practice used in MedSAM and HQ-SAM.
"""

from typing import List

import torch
from torch.nn import functional as F


def point_sample(input: torch.Tensor, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
    """Grid-sample features at (possibly batched) point coordinates.

    Wrapper around ``torch.nn.functional.grid_sample`` that supports
    3-D ``point_coords`` tensors and expects coordinates in [0, 1] x [0, 1].
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def cat(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Efficient ``torch.cat`` that avoids a copy for a single element."""
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    uncertainty_func,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> torch.Tensor:
    """Sample points in [0, 1] x [0, 1] based on their uncertainty.

    See the PointRend paper for details.
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1

    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)

    point_uncertainties = uncertainty_func(point_logits)

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]

    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [point_coords, torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)],
            dim=1,
        )
    return point_coords


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    """Dice loss (generalized IoU for masks)."""
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    """Sigmoid binary cross-entropy loss."""
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)


def calculate_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """Estimate uncertainty as the L1 distance between 0 and the logit.

    The most uncertain locations (closest to the decision boundary) have
    the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    return -(torch.abs(logits.clone()))


def loss_masks(
    src_masks: torch.Tensor,
    target_masks: torch.Tensor,
    num_masks: float,
    oversample_ratio: float = 3.0,
):
    """Compute the mask losses: sigmoid CE + Dice on uncertain points.

    Args:
        src_masks: [B, 1, H, W] predicted logits.
        target_masks: [B, 1, H, W] ground-truth masks in [0, 1].
        num_masks: batch size used for normalization.
        oversample_ratio: PointRend oversampling ratio.

    Returns:
        loss_mask: scalar sigmoid CE loss.
        loss_dice: scalar Dice loss.
    """
    with torch.no_grad():
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks,
            lambda logits: calculate_uncertainty(logits),
            112 * 112,
            oversample_ratio,
            0.75,
        )
        point_labels = point_sample(target_masks, point_coords, align_corners=False).squeeze(1)

    point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)

    loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
    loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)

    return loss_mask, loss_dice
