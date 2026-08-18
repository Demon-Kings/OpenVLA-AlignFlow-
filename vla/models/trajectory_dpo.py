"""
================================================================================
Embodied Trajectory Direct Preference Optimization (Trajectory-DPO) Engine.
Migrated from integrated_pipeline/src/dpo_module.py for Continuous Action Flow.
================================================================================
Includes:
1. Continuous Action Log-Likelihood Proxy via Flow Matching ODE vector field error.
2. 【Cauchy C1 Smooth BNF】: Softplus(0.1 - ΔAdv, beta=10.0) replaces non-smooth ReLU.
3. 【Lagrange KKT Dual Beta】: Dynamic beta update: beta_t+1 = clamp(beta_t + eta*(KL - 0.05)).
4. 【Trajectory Path Efficiency / Length Penalty】: Dual ascent on trajectory duration/slack.
5. 【SFT / BC Auxiliary Regularization】: Anchors expert policy to prevent likelihood displacement.
6. 【Riemann Geodesic Manifold Regularization】: Stabilizes policy logits on continuous manifold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Optional
import copy


class EmbodiedTrajectoryDPOTrainer:
    """
    SOTA Embodied Trajectory DPO Trainer tailored for Flow-VLA policies.
    """
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        config: Any,
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        # Freeze reference model completely
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.cfg = config
        
        # Dual variables and hyperparameter initialization
        self.current_beta = getattr(config.rl, "dpo_beta", 0.1)
        self.current_length_lambda = getattr(config.rl, "initial_length_lambda", 0.005)
        self.target_kl = getattr(config.rl, "target_kl", 0.05)
        
        # Feature toggles
        self.use_cauchy_smoothness = getattr(config.rl, "use_cauchy_smoothness", True)
        self.use_bidirectional_feedback = getattr(config.rl, "use_bidirectional_feedback", True)
        self.use_kkt_dual_ascent = getattr(config.rl, "use_kkt_dual_ascent", True)
        self.use_riemann_geodesic = getattr(config.rl, "use_riemann_geodesic", True)
        self.use_bc_aux_loss = getattr(config.rl, "use_bc_aux_loss", True)

    def compute_action_log_prob_proxy(
        self,
        model: nn.Module,
        obs_images: torch.Tensor,
        instructions: Any,
        action_chunks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the log-likelihood proxy log π(A|s) for continuous actions.
        In Flow Matching, the log-likelihood is directly proportional to
        the negative integrated vector field error: - || v_theta(x_t, t, c) - (x_1 - x_0) ||^2.
        """
        B = action_chunks.shape[0]
        device = action_chunks.device
        
        context_c, _, _ = model.backbone(obs_images, instructions)
        
        # Sample prior noise & timestep
        x_0 = torch.randn_like(action_chunks)
        t = torch.rand(B, device=device)
        t_expand = t.view(B, 1, 1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * action_chunks
        target_v = action_chunks - x_0
        
        pred_v = model.action_head(x_t, t, context_c)
        
        # Negative squared error across all 16x7 action dimensions
        squared_err = torch.sum((pred_v - target_v) ** 2, dim=[1, 2])
        # Proxy log probability
        log_prob_proxy = -squared_err
        return log_prob_proxy

    def update_kkt_dual_beta(self, chosen_adv: torch.Tensor):
        """
        【Lagrange KKT Dual Beta】
        Dynamically adjusts beta to keep policy KL divergence constrained around target_kl (0.05).
        """
        if not self.use_kkt_dual_ascent:
            return
        with torch.no_grad():
            kl_proxy = chosen_adv.detach().abs().mean().item()
            lr_dual = 0.001
            # Dual update: if KL exceeds target, increase beta to penalize drift
            self.current_beta = max(0.02, min(0.50, self.current_beta + lr_dual * (kl_proxy - self.target_kl)))

    def update_kkt_dual_length_lambda(self, chosen_lens: torch.Tensor, rejected_lens: torch.Tensor):
        """
        【Lagrange KKT Dual Length Lambda】
        Penalizes excessive trajectory steps / idle delays if chosen trajectories are longer than necessary.
        """
        if not self.use_kkt_dual_ascent:
            return
        with torch.no_grad():
            avg_len_diff = (chosen_lens - rejected_lens).mean().item()
            target_slack = 0.0
            lr_dual_len = 0.0005
            self.current_length_lambda = max(0.0, min(0.05, self.current_length_lambda + lr_dual_len * (avg_len_diff - target_slack)))

    def compute_trajectory_dpo_loss(
        self,
        obs_images: torch.Tensor,
        instructions: Any,
        chosen_actions: torch.Tensor,
        rejected_actions: torch.Tensor,
        chosen_lengths: Optional[torch.Tensor] = None,
        rejected_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the complete SOTA Trajectory DPO loss with 5 advanced math components.
        """
        B = chosen_actions.shape[0]
        device = chosen_actions.device

        # 1. Policy model log-prob proxies
        pi_chosen_logp = self.compute_action_log_prob_proxy(self.policy, obs_images, instructions, chosen_actions)
        pi_rejected_logp = self.compute_action_log_prob_proxy(self.policy, obs_images, instructions, rejected_actions)

        # 2. Reference model log-prob proxies (no grad)
        with torch.no_grad():
            ref_chosen_logp = self.compute_action_log_prob_proxy(self.ref_model, obs_images, instructions, chosen_actions)
            ref_rejected_logp = self.compute_action_log_prob_proxy(self.ref_model, obs_images, instructions, rejected_actions)

        # 3. Log-ratio & Advantages
        chosen_logratios = pi_chosen_logp - ref_chosen_logp
        rejected_logratios = pi_rejected_logp - ref_rejected_logp

        chosen_advantages = self.current_beta * chosen_logratios
        rejected_advantages = self.current_beta * rejected_logratios

        logits = chosen_advantages - rejected_advantages
        base_dpo_loss = -F.logsigmoid(logits).mean()

        # 4. 【Cauchy C1 Smooth BNF (Bidirectional Negative Feedback)】
        if self.use_bidirectional_feedback:
            delta_adv = chosen_advantages - rejected_advantages
            if self.use_cauchy_smoothness:
                bnf_loss = F.softplus(0.1 - delta_adv, beta=10.0).mean()
            else:
                bnf_loss = F.relu(0.1 - delta_adv).mean()
        else:
            bnf_loss = torch.tensor(0.0, device=device)

        # 5. 【KKT Dynamic Dual Trajectory Length Penalty】
        if chosen_lengths is not None and rejected_lengths is not None:
            len_diff = chosen_lengths - rejected_lengths
            if self.use_cauchy_smoothness:
                length_loss = F.softplus(len_diff, beta=5.0).mean()
            else:
                length_loss = len_diff.clamp(min=0.0).mean()
            length_penalty = self.current_length_lambda * length_loss
        else:
            length_penalty = torch.tensor(0.0, device=device)

        # 6. 【SFT / BC Auxiliary Reconstruction Regularization】
        # Anchors policy on expert demonstration to prevent catastrophic forgetting
        if self.use_bc_aux_loss:
            bc_aux_loss = -pi_chosen_logp.mean()
        else:
            bc_aux_loss = torch.tensor(0.0, device=device)

        # 7. 【Riemann Geodesic Regularization】
        # Prevents log-prob explosion on action manifold
        if self.use_riemann_geodesic:
            riemann_penalty = 0.005 * (pi_chosen_logp.pow(2).mean() + pi_rejected_logp.pow(2).mean())
        else:
            riemann_penalty = torch.tensor(0.0, device=device)

        # Total SOTA Loss composition
        total_loss = (
            base_dpo_loss
            + 0.10 * bnf_loss
            + length_penalty
            + 0.05 * bc_aux_loss
            + riemann_penalty
        )

        # Update KKT dual multipliers
        self.update_kkt_dual_beta(chosen_advantages)
        if chosen_lengths is not None and rejected_lengths is not None:
            self.update_kkt_dual_length_lambda(chosen_lengths, rejected_lengths)

        return {
            "loss_total_dpo": total_loss,
            "loss_base_dpo": base_dpo_loss,
            "loss_bnf": bnf_loss,
            "loss_length_penalty": length_penalty,
            "loss_bc_aux": bc_aux_loss,
            "loss_riemann": riemann_penalty,
            "current_beta": self.current_beta,
            "current_lambda_len": self.current_length_lambda,
            "chosen_advantage": chosen_advantages.detach().mean().item(),
            "rejected_advantage": rejected_advantages.detach().mean().item(),
            "preference_margin": logits.detach().mean().item(),
        }
