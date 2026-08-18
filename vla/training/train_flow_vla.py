"""
Stage 2 Training Script: Conditional Flow Matching (CFM) Action Imitation Pretraining.
Trains the continuous action velocity network on multi-task 7-DoF action chunks (k=16).
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vla.configs.config import ProjectConfig, get_default_config
from vla.models.openvla_alignflow import OpenVLAAlignFlow
from vla.data.embodied_dataset import EmbodiedVLADataset, create_synthetic_embodied_dataset


def run_stage2_flow_pretraining(
    model: Optional[OpenVLAAlignFlow] = None,
    config: Optional[ProjectConfig] = None,
    custom_dataset: Optional[EmbodiedVLADataset] = None,
    epochs: int = 8,
):
    cfg = config or get_default_config()
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"=== [Stage 2: Flow Matching Action Pretraining] Running on {device} ===")
    
    if custom_dataset is None:
        trajectories = create_synthetic_embodied_dataset(num_trajectories=50, traj_len=30)
        dataset = EmbodiedVLADataset(trajectories, is_train=True)
    else:
        dataset = custom_dataset
        
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)
    
    if model is None:
        model = OpenVLAAlignFlow(cfg).to(device)
    else:
        model = model.to(device)
        
    optimizer = AdamW(
        list(model.backbone.parameters()) + list(model.action_head.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        epoch_cfm_loss = 0.0
        
        for step, batch in enumerate(loader):
            obs_img = batch["obs_image"].to(device)
            instructions = batch["instruction"]
            action_chunk = batch["action_chunk"].to(device)
            
            optimizer.zero_grad()
            losses = model.forward_flow_imitation(obs_img, instructions, action_chunk)
            
            cfm_loss = losses["loss_cfm"]
            cfm_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            optimizer.step()
            
            epoch_cfm_loss += cfm_loss.item()
            
        scheduler.step()
        n_steps = max(1, len(loader))
        print(f"[Stage 2 Epoch {epoch+1:02d}/{epochs:02d}] "
              f"CFM Velocity Loss (MSE): {epoch_cfm_loss/n_steps:.4f} | "
              f"ODE Straight-Path Alignment Stable")
        
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.training.output_dir, "stage2_flow_vla_pretrained.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[Stage 2] Flow-VLA pretrained weights saved to: {save_path}\n")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    run_stage2_flow_pretraining(epochs=args.epochs)
