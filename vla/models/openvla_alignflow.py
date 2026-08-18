"""
Full End-to-End OpenVLA-AlignFlow Model Architecture.
Integrates:
1. Embodied Vision-Language Multimodal Backbone (with auto local pretrained model loading)
2. Sub-goal & Affordance Fine-Grained Alignment Head
3. Conditional Flow Matching (CFM) Continuous Action Head
4. Offline Reinforcement Learning (IQL Critic: V & Q Networks)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from vla.models.vl_backbone import EmbodiedMultimodalBackbone
from vla.models.vl_alignment import EmbodiedVLAlignmentModule
from vla.models.flow_action_head import FlowActionHead
from vla.configs.config import ProjectConfig, get_default_config


class IQLCriticHead(nn.Module):
    def __init__(self, context_dim: int = 768, action_chunk_dim: int = 112, hidden_dim: int = 512):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q_net = nn.Sequential(
            nn.Linear(context_dim + action_chunk_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_v(self, context_c: torch.Tensor) -> torch.Tensor:
        return self.v_net(context_c).squeeze(-1)

    def forward_q(self, context_c: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        B = context_c.shape[0]
        a_flat = action_chunk.reshape(B, -1)
        qa_input = torch.cat([context_c, a_flat], dim=-1)
        return self.q_net(qa_input).squeeze(-1)


class OpenVLAAlignFlow(nn.Module):
    def __init__(self, config: Optional[ProjectConfig] = None):
        super().__init__()
        self.cfg = config or get_default_config()
        
        # 1. Resolve local pretrained paths if present
        local_vlm_path = self.cfg.model.get_local_model_path(self.cfg.model.vlm_backbone_type)
        local_vis_path = self.cfg.model.get_local_model_path(self.cfg.model.vision_encoder_name)
        
        if local_vlm_path or local_vis_path:
            print("[*] [OpenVLA-AlignFlow] 正在自动挂载本地预训练底座:")
            if local_vlm_path:
                print(f"    - VLM 语言/多模态底座路径: {local_vlm_path}")
            if local_vis_path:
                print(f"    - 视觉编码器底座路径: {local_vis_path}")
        
        # 2. Multimodal Vision-Language Backbone
        self.backbone = EmbodiedMultimodalBackbone(
            hidden_dim=self.cfg.model.hidden_dim,
            local_vision_path=local_vis_path,
            local_text_path=local_vlm_path,
        )
        
        # 3. Vision-Language Alignment Module
        self.vl_aligner = EmbodiedVLAlignmentModule(
            hidden_dim=self.cfg.model.hidden_dim,
            temperature=self.cfg.model.contrastive_temperature,
            affordance_weight=self.cfg.model.affordance_weight,
        )
        
        # 4. Conditional Flow Matching Action Head
        self.action_head = FlowActionHead(
            action_dim=self.cfg.data.action_dim,
            chunk_size=self.cfg.data.action_chunk_size,
            context_dim=self.cfg.model.hidden_dim,
            hidden_dim=self.cfg.model.flow_hidden_dim,
            time_emb_dim=self.cfg.model.time_emb_dim,
            num_layers=self.cfg.model.flow_num_layers,
        )
        
        # 5. Offline RL Critic Networks (IQL)
        self.critic = IQLCriticHead(
            context_dim=self.cfg.model.hidden_dim,
            action_chunk_dim=self.cfg.data.action_dim * self.cfg.data.action_chunk_size,
            hidden_dim=self.cfg.model.flow_hidden_dim,
        )

    def extract_context(
        self,
        images: torch.Tensor,
        instructions: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.backbone(images, instructions)

    def forward_vl_alignment(
        self,
        obs_images: torch.Tensor,
        goal_images: torch.Tensor,
        instructions: Any,
        affordance_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        context_c, patch_tokens, cross_attn_weights = self.backbone(obs_images, instructions)
        goal_cls_feat, _ = self.backbone.vision_encoder(goal_images)
        text_feat, _ = self.backbone.text_encoder(
            self.backbone.text_encoder.tokenize_text(instructions, device=obs_images.device)
        )
        
        align_losses = self.vl_aligner(
            text_feat=text_feat,
            goal_img_feat=goal_cls_feat,
            cross_attn_weights=cross_attn_weights,
            affordance_masks=affordance_masks,
        )
        return align_losses

    def forward_flow_imitation(
        self,
        obs_images: torch.Tensor,
        instructions: Any,
        action_chunks: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        context_c, _, _ = self.backbone(obs_images, instructions)
        flow_losses = self.action_head.compute_cfm_loss(
            x_1=action_chunks,
            context_c=context_c,
            sample_weights=sample_weights,
        )
        return flow_losses

    def forward_offline_iql(
        self,
        obs_images: torch.Tensor,
        instructions: Any,
        action_chunks: torch.Tensor,
        rewards: torch.Tensor,
        expectile_tau: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            context_c, _, _ = self.backbone(obs_images, instructions)
            
        v_pred = self.critic.forward_v(context_c)
        q_pred = self.critic.forward_q(context_c, action_chunks)
        
        adv = q_pred.detach() - v_pred
        weight = torch.where(adv > 0, expectile_tau, (1.0 - expectile_tau))
        loss_v = torch.mean(weight * (adv ** 2))
        
        target_q = rewards + 0.99 * v_pred.detach()
        loss_q = F.mse_loss(q_pred, target_q)
        
        return {
            "loss_v": loss_v,
            "loss_q": loss_q,
            "advantage": (q_pred - v_pred).detach(),
        }

    @torch.no_grad()
    def predict_action_chunk(
        self,
        obs_images: torch.Tensor,
        instructions: Any,
        num_steps: int = 4,
    ) -> torch.Tensor:
        context_c, _, _ = self.backbone(obs_images, instructions)
        pred_actions = self.action_head.sample_actions(
            context_c=context_c,
            num_steps=num_steps,
        )
        return pred_actions
