"""
Action Space Canonicalization and Normalization Module.
Maps heterogeneous robot embodiment action spaces (joint/cartesian)
into a unified 7-DoF End-Effector (EEF) Relative Delta Pose:
[dx, dy, dz, droll, dpitch, dyaw, gripper] in [-1, 1].
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class ActionCanonicalizer:
    def __init__(
        self,
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        """
        Args:
            quantile_low: Lower quantile for robust normalization
            quantile_high: Upper quantile for robust normalization
            clip_range: Normalized range bounds
        """
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.clip_range = clip_range
        
        # Statistics per dimension for 7-DoF: [dx, dy, dz, dr, dp, dy, gripper]
        self.stats = {
            "mean": np.zeros(7, dtype=np.float32),
            "std": np.ones(7, dtype=np.float32),
            "q_min": -np.ones(7, dtype=np.float32) * 0.1,
            "q_max": np.ones(7, dtype=np.float32) * 0.1,
            "is_fitted": False
        }
        # Gripper default bounds
        self.stats["q_min"][-1] = -1.0
        self.stats["q_max"][-1] = 1.0

    def fit(self, actions_list: np.ndarray):
        """
        Computes robust normalization statistics from a dataset of 7-DoF actions.
        actions_list: (N, 7) array.
        """
        actions = np.asarray(actions_list, dtype=np.float32)
        if len(actions) == 0:
            return
        
        self.stats["mean"] = np.mean(actions, axis=0)
        self.stats["std"] = np.std(actions, axis=0) + 1e-6
        
        # Quantile bounds for pos and rot (dims 0..5)
        for i in range(6):
            self.stats["q_min"][i] = np.quantile(actions[:, i], self.quantile_low)
            self.stats["q_max"][i] = np.quantile(actions[:, i], self.quantile_high)
            # Avoid division by zero
            if abs(self.stats["q_max"][i] - self.stats["q_min"][i]) < 1e-5:
                self.stats["q_min"][i] -= 0.01
                self.stats["q_max"][i] += 0.01
                
        self.stats["is_fitted"] = True

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        """
        Normalizes raw 7-DoF delta actions into [-1, 1].
        actions: (..., 7)
        """
        actions = np.asarray(actions, dtype=np.float32)
        norm_actions = np.zeros_like(actions)
        
        q_min = self.stats["q_min"]
        q_max = self.stats["q_max"]
        
        # Scale to [-1, 1] using min-max on quantiles
        for i in range(6):
            scaled = 2.0 * (actions[..., i] - q_min[i]) / (q_max[i] - q_min[i]) - 1.0
            norm_actions[..., i] = np.clip(scaled, self.clip_range[0], self.clip_range[1])
            
        # Gripper is strictly clamped to {-1.0, 1.0} or [-1, 1]
        norm_actions[..., 6] = np.clip(actions[..., 6], -1.0, 1.0)
        return norm_actions

    def denormalize(self, norm_actions: np.ndarray) -> np.ndarray:
        """
        Restores raw 7-DoF delta actions from normalized space [-1, 1].
        norm_actions: (..., 7)
        """
        norm_actions = np.asarray(norm_actions, dtype=np.float32)
        raw_actions = np.zeros_like(norm_actions)
        
        q_min = self.stats["q_min"]
        q_max = self.stats["q_max"]
        
        for i in range(6):
            clamped = np.clip(norm_actions[..., i], self.clip_range[0], self.clip_range[1])
            raw_actions[..., i] = 0.5 * (clamped + 1.0) * (q_max[i] - q_min[i]) + q_min[i]
            
        # Binary gripper recovery
        raw_actions[..., 6] = np.where(norm_actions[..., 6] > 0.0, 1.0, -1.0)
        return raw_actions

    @staticmethod
    def map_embodiment_action(raw_action: np.ndarray, embodiment_name: str) -> np.ndarray:
        """
        Maps embodiment-specific raw control formats to the unified 7-DoF format.
        Output: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        raw = np.asarray(raw_action, dtype=np.float32)
        target = np.zeros(7, dtype=np.float32)
        
        if "widowx" in embodiment_name.lower() or "bridge" in embodiment_name.lower():
            # Bridge format: [dx, dy, dz, droll, dpitch, dyaw, gripper (0/1)]
            target[:6] = raw[:6]
            target[6] = 1.0 if raw[6] > 0.5 else -1.0
        elif "google" in embodiment_name.lower() or "fractal" in embodiment_name.lower():
            # RT-1 format: [dx, dy, dz, dr, dp, dy, gripper, base_disp, ...]
            target[:6] = raw[:6]
            target[6] = 1.0 if raw[6] > 0.5 else -1.0
        elif "franka" in embodiment_name.lower() or "panda" in embodiment_name.lower():
            # Franka Delta EEF format
            target[:6] = raw[:6]
            target[6] = 1.0 if raw[6] > 0.0 else -1.0
        else:
            # Default pass-through or truncation to 7-DoF
            dim = min(len(raw), 7)
            target[:dim] = raw[:dim]
            
        return target
