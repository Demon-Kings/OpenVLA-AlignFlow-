"""
Global Configuration for OpenVLA-AlignFlow.
Includes deep training presets for high-accuracy convergence:
- Extended Epochs for Stage 1, Stage 2, Stage 3
- Cosine Annealing with Warmup
- Sharp Contrastive Temperature (tau=0.05)
- Enhanced Affordance Loss Weight (1.2)
- Multi-step ODE Precision Sampling
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import os
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


@dataclass
class DataConfig:
    processed_data_dir: str = os.path.join(PROJECT_ROOT, "data", "processed")
    raw_bridge_data_dir: str = os.path.join(PROJECT_ROOT, "bridge_dataset")
    
    dataset_names: List[str] = field(default_factory=lambda: [
        "bridge_dataset",
        "fractal20220817_data",
        "franka_kitchen_droid",
    ])
    total_raw_trajectories: int = 90000
    target_clean_trajectories: int = 45000
    
    dt: float = 0.1                            # 10 Hz control loop
    max_velocity_threshold: float = 0.85       # Max EEF velocity (m/s)
    max_acceleration_threshold: float = 3.5    # Max EEF acceleration (m/s^2)
    max_idle_ratio: float = 0.55               # Max stationary ratio
    
    action_dim: int = 7                        # 7-DoF EEF Delta
    action_chunk_size: int = 16                # Temporal Chunking Window
    img_height: int = 224
    img_width: int = 224
    
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10


@dataclass
class ModelConfig:
    pretrained_models_candidates: List[str] = field(default_factory=lambda: [
        os.path.join(PROJECT_ROOT, "vla", "pretrained_models"),
        os.path.join(PROJECT_ROOT, "pretrained_models"),
        os.path.join(PROJECT_ROOT, "models"),
    ])
    pretrained_models_dir: str = os.path.join(PROJECT_ROOT, "vla", "pretrained_models")
    
    vlm_backbone_type: str = "qwen3-vl"
    vision_encoder_name: str = "siglip"
    
    hidden_dim: int = 768
    vl_num_heads: int = 12
    vl_num_layers: int = 6
    vocab_size: int = 32000
    max_text_len: int = 64
    
    # Deep Tuning: Sharpened contrastive temperature and stronger affordance pull
    contrastive_temperature: float = 0.05      # Sharpened from 0.07 to boost R@1 / R@5
    affordance_weight: float = 1.2             # Boosted from 0.5 to drive IoU > 60%
    
    # Conditional Flow Matching (CFM) Action Head
    flow_hidden_dim: int = 512
    flow_num_layers: int = 4
    time_emb_dim: int = 128
    ode_sampling_steps: int = 6                # Boosted from 4 to 6 for higher precision
    ode_solver: str = "euler"

    def get_local_model_path(self, model_key: str) -> Optional[str]:
        for base_dir in self.pretrained_models_candidates:
            if not os.path.exists(base_dir):
                continue
            manifest_file = os.path.join(base_dir, "manifest.json")
            if os.path.exists(manifest_file):
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                        if model_key in manifest and os.path.exists(manifest[model_key]):
                            return os.path.abspath(manifest[model_key])
                except Exception:
                    pass
            folder_names = [
                model_key.replace("-", "_"),
                model_key,
                model_key.upper(),
                "Qwen3-VL-2B" if "qwen3" in model_key else "",
                "siglip" if "siglip" in model_key else "",
                "dinov2" if "dinov2" in model_key else "",
            ]
            for fn in folder_names:
                if not fn:
                    continue
                candidate_path = os.path.join(base_dir, fn)
                if os.path.exists(candidate_path) and os.path.isdir(candidate_path):
                    return os.path.abspath(candidate_path)
        return None


@dataclass
class OfflineRLConfig:
    iql_expectile_tau: float = 0.7
    discount_gamma: float = 0.99
    advantage_temperature_beta: float = 3.0
    max_advantage_clip: float = 20.0
    
    dpo_beta: float = 0.15                     # Increased from 0.10 for sharper separation
    initial_length_lambda: float = 0.008       # Increased for tighter path control
    target_kl: float = 0.05                    # KL bound
    use_cauchy_smoothness: bool = True         # Cauchy C1 Softplus BNF
    use_bidirectional_feedback: bool = True
    use_kkt_dual_ascent: bool = True
    use_riemann_geodesic: bool = True
    use_bc_aux_loss: bool = True
    reference_model_update_freq: int = 1000


@dataclass
class TrainingConfig:
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 2e-4                # Tuned learning rate with Cosine Annealing
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    
    # Deep Training Epoch Presets
    stage1_vl_align_epochs: int = 50           # Boosted to 25 for deep cross-attention alignment
    stage2_cfm_pretrain_epochs: int = 50       # Boosted to 45 for deep vector field convergence
    stage3_offline_rl_epochs: int = 50         # Boosted to 20 for deep DPO preference optimization
    
    log_interval: int = 50
    eval_interval: int = 500
    output_dir: str = os.path.join(PROJECT_ROOT, "checkpoints", "openvla_alignflow")
    save_top_k: int = 3


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rl: OfflineRLConfig = field(default_factory=OfflineRLConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_default_config() -> ProjectConfig:
    return ProjectConfig()
