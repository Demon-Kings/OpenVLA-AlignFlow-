"""
Stage 1 Training Script: Embodied Vision-Language Fine-Grained Alignment.
Trains the multimodal backbone using Sub-goal InfoNCE and Affordance Cross-Attention losses.
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


def run_stage1_vl_alignment(
    config: Optional[ProjectConfig] = None,
    custom_dataset: Optional[EmbodiedVLADataset] = None,
    epochs: int = 5,
):
    cfg = config or get_default_config()
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"=== [Stage 1: Vision-Language Alignment] Running on {device} ===")
    
    if custom_dataset is None:
        trajectories = create_synthetic_embodied_dataset(num_trajectories=40, traj_len=25)
        dataset = EmbodiedVLADataset(trajectories, is_train=True)
    else:
        dataset = custom_dataset
        
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)
    
    model = OpenVLAAlignFlow(cfg).to(device)
    optimizer = AdamW(
        list(model.backbone.parameters()) + list(model.vl_aligner.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_infonce = 0.0
        epoch_aff = 0.0
        
        for step, batch in enumerate(loader):
            obs_img = batch["obs_image"].to(device)
            goal_img = batch["goal_image"].to(device)
            instructions = batch["instruction"]
            aff_mask = batch["affordance_mask"].to(device)
            
            optimizer.zero_grad()
            losses = model.forward_vl_alignment(obs_img, goal_img, instructions, aff_mask)
            
            total_loss = losses["loss_total_vl"]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_infonce += losses["loss_infonce"].item()
            epoch_aff += losses["loss_affordance"].item()
            
        scheduler.step()
        n_steps = max(1, len(loader))
        print(f"[Stage 1 Epoch {epoch+1:02d}/{epochs:02d}] "
              f"Total VL Loss: {epoch_loss/n_steps:.4f} | "
              f"Sub-goal InfoNCE: {epoch_infonce/n_steps:.4f} | "
              f"Affordance KL: {epoch_aff/n_steps:.4f}")
        
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.training.output_dir, "stage1_vl_aligned.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[Stage 1] Alignment weights saved to: {save_path}\n")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    run_stage1_vl_alignment(epochs=args.epochs)
