"""
Embodied VLA PyTorch Dataset and DataLoader Pipeline.
Implements:
1. Temporal Action Chunking (k=16) with sliding windows
2. Sub-goal image pair extraction (I_t, I_goal)
3. Affordance spatial mask conditioning
4. DPO preference labeling (Preferred Clean vs Dispreferred Noisy)
5. Built-in Synthetic Multi-Embodiment Generator for instant benchmarking and verification
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from vla.configs.config import DataConfig, get_default_config
from vla.data.canonicalize import ActionCanonicalizer
from vla.data.vlm_annotator import VLMAnnotator
from vla.data.kinetic_filter import KineticJitterFilter


class EmbodiedVLADataset(Dataset):
    def __init__(
        self,
        trajectories: List[Dict[str, Any]],
        chunk_size: int = 16,
        action_dim: int = 7,
        img_size: Tuple[int, int] = (224, 224),
        canonicalizer: Optional[ActionCanonicalizer] = None,
        is_train: bool = True,
    ):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.img_size = img_size
        self.is_train = is_train
        
        self.canonicalizer = canonicalizer or ActionCanonicalizer()
        self.vlm_annotator = VLMAnnotator()
        
        # Build samples index: (traj_idx, start_t, goal_t)
        self.samples: List[Tuple[int, int, int]] = []
        self.trajectories = trajectories
        
        self._build_index()

    def _build_index(self):
        for traj_idx, traj in enumerate(self.trajectories):
            T = len(traj["actions"])
            if T < 2:
                continue
            subgoals = traj.get("subgoals", [T - 1])
            
            for t in range(T):
                # Find nearest future subgoal
                future_subgoals = [s for s in subgoals if s >= t]
                goal_t = future_subgoals[0] if future_subgoals else (T - 1)
                self.samples.append((traj_idx, t, goal_t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, t, goal_t = self.samples[idx]
        traj = self.trajectories[traj_idx]
        
        # 1. Current Observation & Sub-goal Images
        obs_img = traj["images"][t]  # (H, W, 3) in uint8 or float
        goal_img = traj["images"][goal_t]
        
        if obs_img.dtype == np.uint8:
            obs_img = obs_img.astype(np.float32) / 255.0
            goal_img = goal_img.astype(np.float32) / 255.0
            
        # Convert to Channel-First: (3, H, W)
        obs_tensor = torch.from_numpy(obs_img).permute(2, 0, 1).float()
        goal_tensor = torch.from_numpy(goal_img).permute(2, 0, 1).float()
        
        # 2. Text Instruction
        instructions = traj.get("instructions", ["pick up the target object"])
        if isinstance(instructions, list) and self.is_train:
            text = np.random.choice(instructions)
        elif isinstance(instructions, list):
            text = instructions[0]
        else:
            text = str(instructions)
            
        # 3. Action Chunking (k=16) with tail padding
        T = len(traj["actions"])
        actions_window = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        
        valid_len = min(self.chunk_size, T - t)
        raw_slice = traj["actions"][t : t + valid_len]
        
        # Normalize actions
        norm_slice = self.canonicalizer.normalize(raw_slice)
        actions_window[:valid_len] = norm_slice
        
        # Zero/Repeat padding for trailing steps
        if valid_len < self.chunk_size:
            actions_window[valid_len:] = norm_slice[-1]
            
        action_chunk_tensor = torch.from_numpy(actions_window).float()
        
        # 4. Affordance Spatial Mask
        aff_mask = traj.get("affordance_masks", None)
        if aff_mask is not None and t < len(aff_mask):
            mask_np = aff_mask[t]
        else:
            mask_np = self.vlm_annotator.generate_affordance_mask(self.img_size)
        aff_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()  # (1, H, W)
        
        # 5. Quality / Preference Label (for Trajectory DPO / IQL)
        is_expert = float(traj.get("is_clean", True))
        reward = float(traj.get("return", 1.0 if is_expert else 0.1))
        
        return {
            "obs_image": obs_tensor,
            "goal_image": goal_tensor,
            "instruction": text,
            "action_chunk": action_chunk_tensor,
            "affordance_mask": aff_tensor,
            "is_expert": torch.tensor(is_expert, dtype=torch.float32),
            "reward": torch.tensor(reward, dtype=torch.float32),
        }


def create_synthetic_embodied_dataset(
    num_trajectories: int = 50,
    traj_len: int = 30,
    img_size: Tuple[int, int] = (224, 224),
    noise_ratio: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Generates synthetic robot trajectories mimicking OpenX (Bridge/RT-1/Franka)
    with kinetic variations, noisy teleop examples, and multi-modal actions.
    """
    trajectories = []
    vlm = VLMAnnotator()
    filter_engine = KineticJitterFilter()
    
    tasks = [
        ("pick red block", (0.3, 0.4)),
        ("place cup on plate", (0.6, 0.7)),
        ("push drawer open", (0.5, 0.2)),
        ("rotate valve knob", (0.4, 0.6)),
    ]
    
    for idx in range(num_trajectories):
        task_name, target_center = tasks[idx % len(tasks)]
        is_noisy = (np.random.rand() < noise_ratio)
        
        # Synthetic position trajectory
        t_steps = np.linspace(0, 1, traj_len)
        base_x = np.sin(t_steps * np.pi * 0.5) * 0.2
        base_y = np.cos(t_steps * np.pi * 0.5) * 0.2
        base_z = np.linspace(0.1, 0.3, traj_len)
        
        eef_pos = np.stack([base_x, base_y, base_z], axis=-1)
        
        # Inject jitter if noisy
        if is_noisy:
            eef_pos += np.random.normal(0, 0.08, size=eef_pos.shape)
            
        # Synthetic actions (7-DoF delta + gripper)
        actions = np.zeros((traj_len, 7), dtype=np.float32)
        actions[:-1, :3] = (eef_pos[1:] - eef_pos[:-1]) / 0.1
        actions[-1, :3] = actions[-2, :3]
        actions[:, 3:6] = np.random.normal(0, 0.02, size=(traj_len, 3))
        actions[:, 6] = np.where(t_steps > 0.5, 1.0, -1.0)
        
        # Synthetic RGB images (H, W, 3)
        images = np.zeros((traj_len, img_size[0], img_size[1], 3), dtype=np.uint8)
        for step in range(traj_len):
            # Gradient background + synthetic colored target square
            img = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * int(step * 4)
            cy, cx = int(target_center[0] * img_size[0]), int(target_center[1] * img_size[1])
            img[max(0, cy-15):min(img_size[0], cy+15), max(0, cx-15):min(img_size[1], cx+15)] = [220, 50, 50]
            images[step] = img
            
        instructions = vlm.expand_instruction(task_name)
        subgoals = vlm.extract_subgoal_keyframes(images, actions)
        
        traj_data = {
            "traj_id": f"synthetic_traj_{idx:05d}",
            "embodiment": "widowx" if idx % 2 == 0 else "franka",
            "images": images,
            "actions": actions,
            "eef_pos": eef_pos,
            "instructions": instructions,
            "subgoals": subgoals,
            "is_clean": not is_noisy,
            "return": 1.0 if not is_noisy else 0.15,
        }
        trajectories.append(traj_data)
        
    return trajectories
