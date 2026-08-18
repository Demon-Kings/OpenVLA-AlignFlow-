"""
OpenVLA-AlignFlow Master Pipeline Entry Point with Deep Training Support.
Executes end-to-end workflow:
1. Auto-discovers local pretrained models in target directory (e.g. ./vla/pretrained_models)
2. Loads extracted real data (./data/processed/)
3. Stage 1: Embodied Vision-Language Multi-modal Alignment (Deep: 25 Epochs)
4. Stage 2: Conditional Flow Matching (CFM) Continuous Action Pretraining (Deep: 45 Epochs)
5. Stage 3: Offline Policy Enhancement (SOTA Trajectory-DPO with 5 Math Components) (Deep: 20 Epochs)
6. Stage 4: Multi-dimensional Offline Benchmark & Ablation Report
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any, List
import torch
import numpy as np

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vla.configs.config import ProjectConfig, get_default_config, PROJECT_ROOT
from vla.data.kinetic_filter import KineticJitterFilter
from vla.data.canonicalize import ActionCanonicalizer
from vla.data.embodied_dataset import EmbodiedVLADataset, create_synthetic_embodied_dataset
from vla.models.openvla_alignflow import OpenVLAAlignFlow
from vla.training.train_vl_align import run_stage1_vl_alignment
from vla.training.train_flow_vla import run_stage2_flow_pretraining
from vla.training.train_offline_rl_dpo import run_stage3_offline_rl_dpo
from vla.evaluation.offline_benchmark import OfflineBenchmarkEvaluator


def execute_full_pipeline(
    mode: str = "full",
    device_name: str = "auto",
    data_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    epochs_s1: Optional[int] = None,
    epochs_s2: Optional[int] = None,
    epochs_s3: Optional[int] = None,
):
    cfg = get_default_config()
    
    # 1. Resolve Data Directory
    resolved_data_dir = os.path.abspath(data_dir) if data_dir else cfg.data.processed_data_dir
    
    # 2. Resolve Models Directory
    if models_dir:
        resolved_models_dir = os.path.abspath(models_dir)
        cfg.model.pretrained_models_dir = resolved_models_dir
        if resolved_models_dir not in cfg.model.pretrained_models_candidates:
            cfg.model.pretrained_models_candidates.insert(0, resolved_models_dir)
    else:
        resolved_models_dir = cfg.model.pretrained_models_dir

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    cfg.training.device = str(device)

    # Resolve Deep Epochs
    s1_epochs = epochs_s1 if epochs_s1 is not None else (cfg.training.stage1_vl_align_epochs if mode == "full" else 2)
    s2_epochs = epochs_s2 if epochs_s2 is not None else (cfg.training.stage2_cfm_pretrain_epochs if mode == "full" else 3)
    s3_epochs = epochs_s3 if epochs_s3 is not None else (cfg.training.stage3_offline_rl_epochs if mode == "full" else 2)

    print("=" * 88)
    print("           OpenVLA-AlignFlow 全链路端到端运行流水线 (深度强化训练模式)")
    print("=" * 88)
    print(f"  [配置] 项目根目录 (PROJECT_ROOT) : {PROJECT_ROOT}")
    print(f"  [配置] 数据加载路径 (DATA_DIR)    : {resolved_data_dir}")
    print(f"  [配置] 预训练底座路径 (MODELS_DIR) : {resolved_models_dir}")
    print(f"  [配置] 计算运行设备 (DEVICE)      : {device}")
    print(f"  [配置] 深度训练轮数 (EPOCHS)      : S1(图文对齐)={s1_epochs} | S2(CFM流匹配)={s2_epochs} | S3(DPO对齐)={s3_epochs}")
    print(f"  [配置] 权重保存路径 (CHECKPOINT)  : {cfg.training.output_dir}")
    print("=" * 88)

    # -------------------------------------------------------------
    # Step 0: Pretrained Models Discovery Check
    # -------------------------------------------------------------
    print(f"\n>>> [Step 0/5] 检查预训练模型目录...")
    vlm_path = cfg.model.get_local_model_path(cfg.model.vlm_backbone_type)
    vis_path = cfg.model.get_local_model_path(cfg.model.vision_encoder_name)
    
    if vlm_path or vis_path:
        print("    [✓] 发现本地预训练底座：")
        if vlm_path:
            print(f"        - VLM 多模态底座: {vlm_path}")
        if vis_path:
            print(f"        - 视觉编码器底座: {vis_path}")
    else:
        print(f"    [*] 提示: 未在预选目录中找到已下载底座，将以轻量初始化模式运行。")

    # -------------------------------------------------------------
    # Step 1: Data Loading & Preparation
    # -------------------------------------------------------------
    train_path = os.path.join(resolved_data_dir, "train_trajectories.npy")
    test_path = os.path.join(resolved_data_dir, "test_trajectories.npy")
    noisy_path = os.path.join(resolved_data_dir, "noisy_preference_trajectories.npy")
    
    use_real_data = (mode != "demo") and os.path.exists(train_path) and os.path.exists(test_path)
    
    if use_real_data:
        print(f"\n>>> [Step 1/5] 正在从 {resolved_data_dir} 加载真实机器人数据集...")
        train_trajs = np.load(train_path, allow_pickle=True).tolist()
        test_trajs = np.load(test_path, allow_pickle=True).tolist()
        noisy_trajs = np.load(noisy_path, allow_pickle=True).tolist() if os.path.exists(noisy_path) else []
        print(f"    [✓] 成功加载真实数据: 训练集 {len(train_trajs)} 条 | 测试集 {len(test_trajs)} 条 | 劣质偏好集 {len(noisy_trajs)} 条")
    else:
        print(f"\n>>> [Step 1/5] 运行【Demo 快速自测模式】(使用内置高保真合成生成器)...")
        num_trajs = 50
        raw_trajectories = create_synthetic_embodied_dataset(num_trajectories=num_trajs, traj_len=30, noise_ratio=0.25)
        filter_engine = KineticJitterFilter(dt=cfg.data.dt)
        clean_trajs, noisy_trajs, _ = filter_engine.batch_filter(raw_trajectories)
        n_train = int(len(clean_trajs) * 0.8)
        train_trajs = clean_trajs[:n_train]
        test_trajs = clean_trajs[n_train:]

    train_dataset = EmbodiedVLADataset(train_trajs, chunk_size=cfg.data.action_chunk_size, is_train=True)
    test_dataset = EmbodiedVLADataset(test_trajs, chunk_size=cfg.data.action_chunk_size, is_train=False)
    print(f"    数据集就绪: 训练 Batch 样本数: {len(train_dataset)}, 离线测试样本数: {len(test_dataset)}")
    
    # -------------------------------------------------------------
    # Step 2: Stage 1 Vision-Language Alignment Training
    # -------------------------------------------------------------
    print(f"\n>>> [Step 2/5] 训练阶段 1: 具身图文细粒度对齐 (Sub-goal InfoNCE + Affordance Loss) [{s1_epochs} 轮]...")
    model = run_stage1_vl_alignment(config=cfg, custom_dataset=train_dataset, epochs=s1_epochs)
    
    # -------------------------------------------------------------
    # Step 3: Stage 2 Flow Matching Action Imitation Pretraining
    # -------------------------------------------------------------
    print(f"\n>>> [Step 3/5] 训练阶段 2: 条件流匹配 (CFM) 连续动作生成头深度模仿训练 [{s2_epochs} 轮]...")
    model = run_stage2_flow_pretraining(model=model, config=cfg, custom_dataset=train_dataset, epochs=s2_epochs)
    
    # -------------------------------------------------------------
    # Step 4: Stage 3 Offline Policy Optimization (IQL + AW-CFM + SOTA Trajectory-DPO)
    # -------------------------------------------------------------
    print(f"\n>>> [Step 4/5] 训练阶段 3: 纯离线策略提升与 SOTA 轨迹 DPO 偏好对齐 [{s3_epochs} 轮]...")
    train_and_noisy = train_trajs + noisy_trajs[:max(1, len(train_trajs)//2)]
    dpo_dataset = EmbodiedVLADataset(train_and_noisy, chunk_size=cfg.data.action_chunk_size, is_train=True)
    model = run_stage3_offline_rl_dpo(model=model, config=cfg, custom_dataset=dpo_dataset, epochs=s3_epochs)
    
    # -------------------------------------------------------------
    # Step 5: Comprehensive Offline Benchmark
    # -------------------------------------------------------------
    print("\n>>> [Step 5/5] 运行多维度纯离线 Benchmark 深度评测 (6 步高精度 Euler ODE)...")
    evaluator = OfflineBenchmarkEvaluator(model, device)
    eval_samples = min(200, len(test_dataset))
    metrics = evaluator.evaluate_dataset(test_dataset, num_samples=eval_samples, ode_steps=cfg.model.ode_sampling_steps)
    evaluator.print_benchmark_report(metrics)
    
    print("=" * 88)
    print("  [SUCCESS] OpenVLA-AlignFlow 深度强化训练全链路执行完成！")
    print(f"  - 最终高精度权重已保存至: {cfg.training.output_dir}")
    print("=" * 88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVLA-AlignFlow Deep Training Pipeline")
    parser.add_argument("--mode", type=str, default="full", choices=["auto", "demo", "full"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default=None)
    parser.add_argument("--epochs_s1", type=int, default=None, help="Stage 1 图文对齐轮数 (默认: 25)")
    parser.add_argument("--epochs_s2", type=int, default=None, help="Stage 2 CFM流匹配轮数 (默认: 45)")
    parser.add_argument("--epochs_s3", type=int, default=None, help="Stage 3 轨迹DPO轮数 (默认: 20)")
    args = parser.parse_args()
    
    execute_full_pipeline(
        mode=args.mode,
        device_name=args.device,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        epochs_s1=args.epochs_s1,
        epochs_s2=args.epochs_s2,
        epochs_s3=args.epochs_s3,
    )
