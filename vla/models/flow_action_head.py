"""
Conditional Flow Matching (CFM) Continuous Action Head.
Learns probability flow ODE velocity fields from standard Gaussian prior x_0 ~ N(0, I)
to true action trajectories x_1 = A_t in R^(k x 7).
Supports fast 4-step Euler numerical integration for real-time inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any, Optional


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) float tensor in [0, 1]
        Returns: (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FlowActionHead(nn.Module):
    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 16,
        context_dim: int = 768,
        hidden_dim: int = 512,
        time_emb_dim: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.total_action_dim = action_dim * chunk_size  # 16 x 7 = 112
        
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        
        # Action chunk input projector
        self.action_proj = nn.Linear(self.total_action_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        # Residual MLP Blocks with LayerNorm & GELU
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.total_action_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_t: (B, 16, 7) or (B, 112) - Noisy action chunk at time t
        t: (B,) - Continuous time in [0, 1]
        context_c: (B, context_dim) - VLM Multimodal Condition
        Returns:
            pred_velocity: (B, 16, 7) - Predicted vector field v_theta(x_t, t, c)
        """
        B = x_t.shape[0]
        x_flat = x_t.reshape(B, self.total_action_dim)
        
        # Embeddings
        h_x = self.action_proj(x_flat)
        h_t = self.time_mlp(self.time_embed(t))
        h_c = self.context_proj(context_c)
        
        h = h_x + h_t + h_c
        
        for block in self.blocks:
            h = h + block(h)
            
        h = self.final_norm(h)
        pred_flat = self.out_proj(h)
        return pred_flat.reshape(B, self.chunk_size, self.action_dim)

    def compute_cfm_loss(
        self,
        x_1: torch.Tensor,
        context_c: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes Conditional Flow Matching loss on linear optimal transport path.
        x_1: (B, 16, 7) - True action chunk
        context_c: (B, context_dim) - Condition
        sample_weights: (B,) - Optional advantage weights from Offline RL
        """
        B = x_1.shape[0]
        device = x_1.device
        
        # Sample prior noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)
        
        # Sample continuous timestep t ~ Uniform(0, 1)
        t = torch.rand(B, device=device)
        
        # Linear interpolation path: x_t = (1 - t) * x_0 + t * x_1
        t_expand = t.view(B, 1, 1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        
        # Target velocity field: u_t = x_1 - x_0
        u_t = x_1 - x_0
        
        # Predict velocity field
        pred_v = self.forward(x_t, t, context_c)
        
        # Mean Squared Error on velocity field
        squared_diff = torch.sum((pred_v - u_t) ** 2, dim=[1, 2])  # (B,)
        
        if sample_weights is not None:
            loss = torch.mean(sample_weights * squared_diff)
        else:
            loss = torch.mean(squared_diff)
            
        return {
            "loss_cfm": loss,
            "pred_velocity": pred_v,
            "target_velocity": u_t,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        context_c: torch.Tensor,
        num_steps: int = 4,
        solver: str = "euler",
    ) -> torch.Tensor:
        """
        Fast Numerical ODE integration from t=0 to t=1.
        Returns: (B, 16, 7) Action Chunk in [-1, 1]
        """
        B = context_c.shape[0]
        device = context_c.device
        
        # Start at noise prior x_0 ~ N(0, I)
        x_curr = torch.randn(B, self.chunk_size, self.action_dim, device=device)
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t_val = step * dt
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)
            
            # Predict velocity
            v_pred = self.forward(x_curr, t_batch, context_c)
            
            # Euler step
            x_curr = x_curr + dt * v_pred
            
        # Clamp to normalized action bounds [-1, 1]
        return torch.clamp(x_curr, -1.0, 1.0)
