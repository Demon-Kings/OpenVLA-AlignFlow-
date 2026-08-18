"""
Stage 3 Training Script: Advanced Embodied Trajectory DPO & IQL Policy Optimization.
Integrates the complete migrated SOTA DPO mechanisms from integrated_pipeline:
- Cauchy C1 Smooth BNF
- Lagrange KKT Dual Beta & Length Lambda dynamic ascent
- SFT / BC Auxiliary Policy Anchoring
- Riemann Geodesic Manifold Regularization
"""

import os
import sys
import copy
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vla.configs.config import ProjectConfig, get_default_config
from vla.models.openvla_alignflow import OpenVLAAlignFlow
from vla.models.trajectory_dpo import EmbodiedTrajectoryDPOTrainer
from vla.data.embodied_dataset import EmbodiedVLADataset, create_synthetic_embodied_dataset


def run_stage3_offline_rl_dpo(
    model: Optional[OpenVLAAlignFlow] = None,
    config: Optional[ProjectConfig] = None,
    custom_dataset: Optional[EmbodiedVLADataset] = None,
    epochs: int = 6,
):
    cfg = config or get_default_config()
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"\n=== [Stage 3: Advanced Trajectory-DPO (SOTA Multi-Mechanism)] Running on {device} ===")
    
    # 1. Dataset & DataLoader (Pairing clean chosen vs noisy rejected)
    if custom_dataset is None:
        trajectories = create_synthetic_embodied_dataset(num_trajectories=60, traj_len=30, noise_ratio=0.3)
        dataset = EmbodiedVLADataset(trajectories, is_train=True)
    else:
        dataset = custom_dataset
        
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)
    
    # 2. Policy Model & Frozen Reference Model
    if model is None:
        model = OpenVLAAlignFlow(cfg).to(device)
    else:
        model = model.to(device)
        
    print("[*] 正在构建不可导 Reference Policy Model (π_ref)...")
    ref_model = copy.deepcopy(model).to(device)
    
    # 3. Initialize Advanced Trajectory DPO Engine
    dpo_trainer = EmbodiedTrajectoryDPOTrainer(
        policy_model=model,
        ref_model=ref_model,
        config=cfg,
    )
    
    # 4. Optimizers
    actor_optimizer = AdamW(
        list(model.backbone.parameters()) + list(model.action_head.parameters()),
        lr=cfg.training.learning_rate * 0.5,
        weight_decay=cfg.training.weight_decay,
    )
    critic_optimizer = AdamW(model.critic.parameters(), lr=cfg.training.learning_rate)
    scheduler = CosineAnnealingLR(actor_optimizer, T_max=epochs)
    
    beta_aw = cfg.rl.advantage_temperature_beta
    clip_max = cfg.rl.max_advantage_clip
    
    print(f"[*] DPO 核心参数: 初始 Beta={dpo_trainer.current_beta:.3f} | 初始 Lambda_Len={dpo_trainer.current_length_lambda:.4f}")
    print(f"[*] 开启特性: Cauchy BNF={dpo_trainer.use_cauchy_smoothness} | KKT 对偶={dpo_trainer.use_kkt_dual_ascent} | 黎曼正则={dpo_trainer.use_riemann_geodesic}")
    
    model.train()
    for epoch in range(epochs):
        epoch_dpo_loss = 0.0
        epoch_bnf_loss = 0.0
        epoch_riemann = 0.0
        epoch_margin = 0.0
        epoch_v_loss = 0.0
        
        for step, batch in enumerate(loader):
            obs_img = batch["obs_image"].to(device)
            instructions = batch["instruction"]
            action_chunk = batch["action_chunk"].to(device)
            rewards = batch["reward"].to(device)
            is_expert = batch["is_expert"].to(device)
            
            # -------------------------------------------------------------
            # Part 1: IQL Critic Value Step
            # -------------------------------------------------------------
            critic_optimizer.zero_grad()
            critic_losses = model.forward_offline_iql(
                obs_images=obs_img,
                instructions=instructions,
                action_chunks=action_chunk,
                rewards=rewards,
                expectile_tau=cfg.rl.iql_expectile_tau,
            )
            total_critic_loss = critic_losses["loss_v"] + critic_losses["loss_q"]
            total_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), cfg.training.max_grad_norm)
            critic_optimizer.step()
            
            # -------------------------------------------------------------
            # Part 2: Advanced SOTA Trajectory DPO Step
            # -------------------------------------------------------------
            actor_optimizer.zero_grad()
            
            # Synthesize/Split paired chosen vs rejected actions
            # For expert samples, synthesize jittery perturbation as rejected; for non-expert, vice versa
            expert_mask = (is_expert > 0.5)
            
            # Clean chosen vs noisy rejected
            chosen_actions = action_chunk.clone()
            # Inject simulated high-frequency teleop jitter to create hard negative rejected pairs
            noise_jitter = torch.randn_like(action_chunk) * 0.15
            rejected_actions = torch.clamp(action_chunk + noise_jitter, -1.0, 1.0)
            
            # Compute advanced DPO loss
            dpo_metrics = dpo_trainer.compute_trajectory_dpo_loss(
                obs_images=obs_img,
                instructions=instructions,
                chosen_actions=chosen_actions,
                rejected_actions=rejected_actions,
            )
            
            total_actor_loss = dpo_metrics["loss_total_dpo"]
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            actor_optimizer.step()
            
            epoch_dpo_loss += total_actor_loss.item()
            epoch_bnf_loss += dpo_metrics["loss_bnf"].item()
            epoch_riemann += dpo_metrics["loss_riemann"].item() if isinstance(dpo_metrics["loss_riemann"], torch.Tensor) else dpo_metrics["loss_riemann"]
            epoch_margin += dpo_metrics["preference_margin"]
            epoch_v_loss += critic_losses["loss_v"].item()
            
        scheduler.step()
        n_steps = max(1, len(loader))
        print(f"[Stage 3 Epoch {epoch+1:02d}/{epochs:02d}] "
              f"DPO Loss: {epoch_dpo_loss/n_steps:.4f} | "
              f"BNF Loss: {epoch_bnf_loss/n_steps:.4f} | "
              f"Margin: {epoch_margin/n_steps:+.3f} | "
              f"Dyn Beta: {dpo_trainer.current_beta:.3f} | "
              f"Dyn Lambda: {dpo_trainer.current_length_lambda:.4f}")
        
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.training.output_dir, "stage3_final_aligned_flow_vla.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[Stage 3] SOTA 迁移对齐后的模型权重已保存至: {save_path}\n")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    run_stage3_offline_rl_dpo(epochs=args.epochs)
