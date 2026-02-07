"""Custom loss functions.

Provides loss functions for handling class imbalance and training strategies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self, gamma: float = 2.0, alpha: torch.Tensor | list[float] | None = None, reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        p_t = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        focal_loss = focal_weight * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Cross entropy loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1, num_classes: int = 4, reduction: str = "mean") -> None:
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(inputs, dim=1)
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), confidence)
        loss = -smooth_targets * log_probs
        loss = loss.sum(dim=1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(cfg: DictConfig) -> nn.Module:
    """Build loss function from configuration."""
    loss_cfg = cfg.get("train", {}).get("loss", {})
    loss_type = loss_cfg.get("type", "cross_entropy")
    class_weights = loss_cfg.get("class_weights")
    if class_weights is not None:
        class_weights = torch.tensor(list(class_weights), dtype=torch.float32)
        logger.info(f"Using class weights: {class_weights.tolist()}")

    if loss_type == "cross_entropy":
        loss = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        logger.info("Using CrossEntropyLoss")
    elif loss_type == "focal":
        gamma = loss_cfg.get("gamma", 2.0)
        loss = FocalLoss(gamma=gamma, alpha=class_weights)
        logger.info(f"Using FocalLoss with gamma={gamma}")
    elif loss_type == "label_smoothing":
        smoothing = loss_cfg.get("smoothing", 0.1)
        num_classes = loss_cfg.get("num_classes", 4)
        loss = LabelSmoothingLoss(smoothing=smoothing, num_classes=num_classes)
        logger.info(f"Using LabelSmoothingLoss with smoothing={smoothing}")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: cross_entropy, focal, label_smoothing")
    return loss
