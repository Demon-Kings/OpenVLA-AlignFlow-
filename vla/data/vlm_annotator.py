"""
VLM Automated Semantic Expansion & Sub-Goal Keyframe Extraction Module.
Uses vision-language reasoning (simulated / Qwen2-VL API) to:
1. Expand terse teleop instructions into diverse, spatially descriptive commands (5x augmentation).
2. Segment long-horizon manipulation trajectories into sub-goals and anchor keyframes.
3. Generate approximate affordance attention heatmaps / masks for target interactive objects.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Any, Optional


class VLMAnnotator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Expansion templates for manipulation actions
        self.verb_expansions = {
            "pick": ["grasp", "reach out and pick up", "lift", "carefully grab", "take hold of"],
            "place": ["put down", "set gently onto", "position", "deposit", "place steadily into"],
            "push": ["shove forward", "nudge", "slide across the table", "push towards"],
            "open": ["pull open", "unlatch and swing open", "rotate open"],
            "close": ["shut", "push to close", "securely close"],
        }
        
        self.spatial_adjectives = [
            "located on the table", "in front of the robot", "near the center of the workspace",
            "resting on the surface", "situated to the right", "situated to the left"
        ]

    def expand_instruction(self, raw_instruction: str) -> List[str]:
        """
        Generates 5 diverse natural language paraphrases with spatial details.
        """
        raw_clean = raw_instruction.lower().strip().rstrip(".")
        tokens = raw_clean.split()
        
        if not tokens:
            return [raw_instruction] * 5
            
        verb = tokens[0]
        obj = " ".join(tokens[1:]) if len(tokens) > 1 else "target object"
        
        # Find matching verb expansion
        synonyms = self.verb_expansions.get(verb, [verb, f"carefully {verb}"])
        
        expanded_prompts = [raw_instruction]
        for _ in range(4):
            syn_verb = random.choice(synonyms)
            spatial = random.choice(self.spatial_adjectives)
            template_id = random.randint(0, 3)
            
            if template_id == 0:
                prompt = f"{syn_verb} the {obj} {spatial}."
            elif template_id == 1:
                prompt = f"Please {syn_verb} the {obj} smoothly."
            elif template_id == 2:
                prompt = f"Using the robot gripper, {syn_verb} the {obj}."
            else:
                prompt = f"Execute manipulation: {syn_verb} the {obj} {spatial}."
            expanded_prompts.append(prompt)
            
        return expanded_prompts[:5]

    def extract_subgoal_keyframes(self, images: np.ndarray, actions: np.ndarray) -> List[int]:
        """
        Identifies key transition indices (Sub-goal anchors) in a trajectory based on:
        1. Gripper state transitions (open -> close or close -> open).
        2. Visual perceptual velocity divergence (peaks in frame-to-frame image delta).
        images: (T, H, W, C)
        actions: (T, 7) [dx, dy, dz, dr, dp, dy, gripper]
        Returns: list of integer frame indices representing subgoals.
        """
        T = len(images)
        if T <= 4:
            return [T - 1]
            
        subgoals = []
        
        # 1. Detect gripper switch points
        grippers = actions[:, 6]
        for t in range(1, T):
            if (grippers[t] > 0 and grippers[t-1] <= 0) or (grippers[t] <= 0 and grippers[t-1] > 0):
                subgoals.append(t)
                
        # 2. Visual feature difference peaks
        if len(images.shape) == 4:
            img_diffs = np.mean(np.abs(images[1:].astype(float) - images[:-1].astype(float)), axis=(1, 2, 3))
            diff_thresh = np.mean(img_diffs) + 1.2 * np.std(img_diffs)
            peak_indices = np.where(img_diffs > diff_thresh)[0] + 1
            for idx in peak_indices:
                if not any(abs(idx - s) < 4 for s in subgoals):
                    subgoals.append(int(idx))
                    
        # Always include the terminal goal frame
        if (T - 1) not in subgoals:
            subgoals.append(T - 1)
            
        subgoals = sorted(list(set(subgoals)))
        return subgoals

    def generate_affordance_mask(self, img_shape: Tuple[int, int], contact_center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Synthesizes a 2D Gaussian affordance heatmap / binary spatial mask around interaction center.
        img_shape: (H, W)
        contact_center: (norm_y, norm_x) in [0, 1]
        Returns: (H, W) float32 mask in [0, 1]
        """
        H, W = img_shape
        if contact_center is None:
            # Default to center of image
            cy, cx = H // 2, W // 2
        else:
            cy, cx = int(contact_center[0] * H), int(contact_center[1] * W)
            
        y = np.arange(0, H)
        x = np.arange(0, W)
        xx, yy = np.meshgrid(x, y)
        
        sigma = min(H, W) * 0.15
        mask = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
        return mask.astype(np.float32)
