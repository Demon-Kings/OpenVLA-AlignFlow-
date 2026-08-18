"""
Embodied Vision-Language Alignment Module.
Implements:
1. Sub-goal Contrastive Learning (InfoNCE Loss between task instruction and future goal visual state)
2. Spatial Affordance-Grounded Cross-Attention Loss (Aligning interactive object masks with text attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class EmbodiedVLAlignmentModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        temperature: float = 0.07,
        affordance_weight: float = 0.5,
    ):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.affordance_weight = affordance_weight
        
        # Projection heads for contrastive space
        self.proj_img = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 256)
        )
        self.proj_text = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 256)
        )

    def compute_subgoal_infonce_loss(
        self,
        text_feat: torch.Tensor,
        goal_img_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        text_feat: (B, hidden_dim)
        goal_img_feat: (B, hidden_dim)
        """
        z_t = F.normalize(self.proj_text(text_feat), dim=-1)
        z_v = F.normalize(self.proj_img(goal_img_feat), dim=-1)
        
        # Cosine similarity matrix: (B, B)
        sim_matrix = torch.matmul(z_t, z_v.T) / torch.clamp(self.temperature, min=0.01, max=1.0)
        
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
        
        loss_t2v = F.cross_entropy(sim_matrix, labels)
        loss_v2t = F.cross_entropy(sim_matrix.T, labels)
        return (loss_t2v + loss_v2t) * 0.5

    def compute_affordance_loss(
        self,
        cross_attn_weights: torch.Tensor,
        affordance_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        cross_attn_weights: (B, 196, L) - Patch-to-text attention
        affordance_masks: (B, 1, 224, 224) - Ground truth interaction area mask
        """
        B = cross_attn_weights.shape[0]
        # Average attention over text tokens: (B, 196) -> (B, 1, 14, 14)
        attn_spatial = cross_attn_weights.mean(dim=-1).view(B, 1, 14, 14)
        
        # Downsample ground-truth affordance mask to 14x14 patch grid
        downsampled_mask = F.adaptive_avg_pool2d(affordance_masks, (14, 14))  # (B, 1, 14, 14)
        
        # Flatten to probability distribution (B, 196)
        p_pred = F.softmax(attn_spatial.flatten(1), dim=-1)
        p_target = F.softmax(downsampled_mask.flatten(1) / 0.1, dim=-1)
        
        # KL-Divergence / MSE loss
        loss = F.kl_div(torch.log(p_pred + 1e-8), p_target, reduction="batchmean")
        return loss

    def forward(
        self,
        text_feat: torch.Tensor,
        goal_img_feat: torch.Tensor,
        cross_attn_weights: torch.Tensor,
        affordance_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the complete vision-language alignment loss.
        """
        loss_infonce = self.compute_subgoal_infonce_loss(text_feat, goal_img_feat)
        loss_aff = self.compute_affordance_loss(cross_attn_weights, affordance_masks)
        
        total_loss = loss_infonce + self.affordance_weight * loss_aff
        return {
            "loss_total_vl": total_loss,
            "loss_infonce": loss_infonce,
            "loss_affordance": loss_aff,
        }
