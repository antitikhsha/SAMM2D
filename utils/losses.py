"""
Loss functions for imbalanced classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits [B, num_classes]
            targets: Ground truth labels [B]
        """
        # Convert to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get class probabilities
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # Compute focal loss
        focal_weight = (1 - pt) ** self.gamma
        
        # Binary cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply alpha weighting
        alpha_weight = self.alpha * targets_one_hot[:, 1] + (1 - self.alpha) * targets_one_hot[:, 0]
        
        loss = alpha_weight * focal_weight * ce_loss
        
        return loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with class weights for imbalanced datasets.
    
    Args:
        pos_weight: Weight for positive class
    """
    def __init__(self, pos_weight=2.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        # Create weight tensor
        weights = torch.ones_like(targets).float()
        weights[targets == 1] = self.pos_weight
        
        # Compute weighted cross entropy
        loss = F.cross_entropy(inputs, targets, reduction='none')
        weighted_loss = loss * weights
        
        return weighted_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for binary classification.
    Often used in medical imaging.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # Compute dice coefficient
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss
        loss = 1 - dice.mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss (e.g., Focal + Dice).
    
    Args:
        loss1: First loss function
        loss2: Second loss function
        weight1: Weight for first loss
        weight2: Weight for second loss
    """
    def __init__(self, loss1, loss2, weight1=0.7, weight2=0.3):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.weight1 = weight1
        self.weight2 = weight2
    
    def forward(self, inputs, targets):
        l1 = self.loss1(inputs, targets)
        l2 = self.loss2(inputs, targets)
        return self.weight1 * l1 + self.weight2 * l2


if __name__ == "__main__":
    # Test losses
    batch_size = 4
    num_classes = 2
    
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 1, 1, 0])
    
    # Focal loss
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal(inputs, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Weighted CE
    weighted_ce = WeightedCrossEntropyLoss(pos_weight=2.0)
    loss = weighted_ce(inputs, targets)
    print(f"Weighted CE Loss: {loss.item():.4f}")
    
    # Dice loss
    dice = DiceLoss()
    loss = dice(inputs, targets)
    print(f"Dice Loss: {loss.item():.4f}")
