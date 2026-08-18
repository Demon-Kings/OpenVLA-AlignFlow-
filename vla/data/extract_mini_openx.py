"""
Extract Mini OpenX (BridgeData v2) Dataset Script
优先读取本地已经下载的BridgeData v2 TFRecord，避免国内网络在线流式失败
Supports:
1. Local TFRecord (Recommended, gsutil预先下载bridge_dataset)
2. Domestic Fast Streaming via HF-Mirror (lerobot/bridge_data_v2) - No Google Cloud / VPN required.
3. GCS Anonymous Stream with ADC bypass.
4. Automatic conversion into 7-DoF EEF Delta space and kinetic filtering.
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
import tensorflow as tf

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vla.data.kinetic_filter import KineticJitterFilter
from vla.data.canonicalize import ActionCanonicalizer
from vla.data.vlm_annotator import VLMAnnotator
from vla.data.embodied_dataset import create_synthetic_embodied_dataset


def load_from_local_tfrecord(local_dir: str, num_trajectories: int = 500) -> Optional[List[Dict[str, Any]]]:
    """
    【推荐】读取本地gsutil下载好的bridge_dataset tfrecord文件
    local_dir: tfrecord文件所在文件夹，例如 D:/datasets/bridge_dataset
    """
    print(f"[*] [通道0:本地TFRecord读取] 从本地目录加载: {local_dir}")
    if not os.path.exists(local_dir):
        print(f"[!] 本地目录不存在 {local_dir}")
        return None

    try:
        import tensorflow_datasets as tfds
    except ImportError:
        print("[!] 安装依赖 tensorflow_datasets")
        os.system("pip install tensorflow tfds-nightly")
        import tensorflow_datasets as tfds

    raw_trajectories = []
    vlm = VLMAnnotator()
    # 从本地目录构建tfds builder
    builder = tfds.builder_from_directory(local_dir)
    dataset = builder.as_dataset(split="train")

    for idx, episode in enumerate(dataset):
        if idx >= num_trajectories:
            break
        steps = list(episode["steps"])
        if len(steps) < 6:
            continue
        # 取最多60步
        steps = steps[:60]
        images = np.stack([s["observation"]["image_0"].numpy() for s in steps])
        actions = np.stack([s["action"].numpy() for s in steps])

        actions_7dof = np.stack([
            ActionCanonicalizer.map_embodiment_action(a, "widowx") for a in actions
        ], axis=0)
        eef_pos = np.cumsum(actions_7dof[:, :3] * 0.1, axis=0)
        text = steps[0]["language_instruction"].numpy().decode("utf-8")

        raw_trajectories.append({
            "traj_id": f"bridge_local_{idx:05d}",
            "embodiment": "widowx",
            "images": images,
            "actions": actions_7dof,
            "eef_pos": eef_pos,
            "instructions": vlm.expand_instruction(text),
            "subgoals": vlm.extract_subgoal_keyframes(images, actions_7dof),
        })
        if idx %50 == 0:
            print(f"    [进度] 已读取本地 {idx}/{num_trajectories} 条轨迹")

    print(f"[*] 本地TFRecord读取完成，共加载 {len(raw_trajectories)} 条轨迹")
    return raw_trajectories


def stream_from_hf_mirror(num_trajectories: int = 500) -> Optional[List[Dict[str, Any]]]:
    """
    Streams directly from Hugging Face Mirror (hf-mirror.com) without downloading whole dataset.
    Works domestically in China with zero Google credentials.
    """
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("[*] [通道 1: HF-Mirror 国内直连] 正在连接 BridgeData v2 镜像流式拉取...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("[!] 正在安装 datasets 库以支持高速流式拉取...")
        os.system("pip install datasets pyarrow")
        from datasets import load_dataset

    raw_trajectories = []
    vlm = VLMAnnotator()

    try:
        # Stream without downloading entire repo
        ds = load_dataset("lerobot/bridge_data_v2", split="train", streaming=True)
        print(f"[*] 成功连接！开始流式读取前 {num_trajectories} 条真实机械臂轨迹...")

        current_ep_actions = []
        current_ep_images = []
        current_task = "manipulate the object on table"
        last_ep_idx = None

        count = 0
        for item in ds:
            ep_idx = item.get("episode_index", count)
            action = item.get("action", None)

            # Extract image if present or create placeholder
            if "image" in item:
                img = np.array(item["image"])
            else:
                img = np.zeros((224, 224, 3), dtype=np.uint8)

            if last_ep_idx is not None and ep_idx != last_ep_idx:
                # Flush completed episode
                if len(current_ep_actions) >= 6:
                    actions_np = np.array(current_ep_actions, dtype=np.float32)
                    images_np = np.array(current_ep_images, dtype=np.uint8)
                    actions_7dof = np.stack([
                        ActionCanonicalizer.map_embodiment_action(a, "widowx") for a in actions_np
                    ], axis=0)
                    eef_pos = np.cumsum(actions_7dof[:, :3] * 0.1, axis=0)
                    subgoals = vlm.extract_subgoal_keyframes(images_np, actions_7dof)
                    expanded_inst = vlm.expand_instruction(current_task)

                    raw_trajectories.append({
                        "traj_id": f"bridge_hf_{count:05d}",
                        "embodiment": "widowx",
                        "images": images_np,
                        "actions": actions_7dof,
                        "eef_pos": eef_pos,
                        "instructions": expanded_inst,
                        "subgoals": subgoals,
                    })
                    count += 1
                    if count % 50 == 0 or count == num_trajectories:
                        print(f"    [进度] 已流式获取 {count} / {num_trajectories} 条真实轨迹...")
                    if count >= num_trajectories:
                        break

                current_ep_actions = []
                current_ep_images = []

            last_ep_idx = ep_idx
            if action is not None:
                current_ep_actions.append(action)
                current_ep_images.append(img)
            if "task" in item or "language_instruction" in item:
                current_task = item.get("task", item.get("language_instruction", current_task))

        return raw_trajectories
    except Exception as e:
        print(f"[!] HF-Mirror 流式连接提示: {e}")
        return None


def stream_from_gcs_tfds(num_trajectories: int = 500) -> Optional[List[Dict[str, Any]]]:
    """Streams from Google Cloud Storage TFDS."""
    print("[*] [通道 2: GCS TFDS] 正在尝试 Google Cloud 存储桶...")
    try:
        import tensorflow_datasets as tfds
        os.environ["NO_GCE_CHECK"] = "True"
        gcs_path = "gs://gresearch/robotics/bridge_dataset/1.0.0"
        builder = tfds.builder_from_directory(gcs_path)
        dataset = builder.as_dataset(split=f"train[:{num_trajectories}]")

        raw_trajectories = []
        vlm = VLMAnnotator()
        for idx, episode in enumerate(dataset):
            steps = list(episode["steps"])
            if len(steps) < 6:
                continue
            images = np.stack([s["observation"]["image_0"].numpy() for s in steps[:60]])
            actions = np.stack([s["action"].numpy() for s in steps[:60]])
            actions_7dof = np.stack([
                ActionCanonicalizer.map_embodiment_action(a, "widowx") for a in actions
            ], axis=0)
            eef_pos = np.cumsum(actions_7dof[:, :3] * 0.1, axis=0)
            text = steps[0]["language_instruction"].numpy().decode("utf-8")
            raw_trajectories.append({
                "traj_id": f"bridge_gcs_{idx:05d}",
                "embodiment": "widowx",
                "images": images,
                "actions": actions_7dof,
                "eef_pos": eef_pos,
                "instructions": vlm.expand_instruction(text),
                "subgoals": vlm.extract_subgoal_keyframes(images, actions_7dof),
            })
        return raw_trajectories
    except Exception as e:
        print(f"[!] GCS 连接失败: {e}")
        return None


def extract_mini_bridgedata(
    num_trajectories: int = 500,
    output_dir: str = "./data/processed",
    local_tfrecord_dir: Optional[str] = None
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print(f"  开始提取真实机器人操作数据集 (目标: {num_trajectories} 条)")
    print("=" * 80)

    raw_trajectories = None
    # 【优先级最高】优先读取本地tfrecord
    if local_tfrecord_dir is not None:
        raw_trajectories = load_from_local_tfrecord(local_tfrecord_dir, num_trajectories=num_trajectories)

    # Strategy 1: HF Mirror (Domestic direct fast stream)
    if not raw_trajectories:
        raw_trajectories = stream_from_hf_mirror(num_trajectories=num_trajectories)

    # Strategy 2: GCS Fallback
    if not raw_trajectories:
        raw_trajectories = stream_from_gcs_tfds(num_trajectories=num_trajectories)

    # Strategy 3: Safe Synthetic Generator
    if not raw_trajectories or len(raw_trajectories) == 0:
        print("\n[*] 自动降级为离线真实动力学模拟器生成 (无需外网，即刻就绪)...")
        raw_trajectories = create_synthetic_embodied_dataset(
            num_trajectories=min(num_trajectories, 100), traj_len=35, noise_ratio=0.25
        )

    print(f"\n[1/3] 正在对提取的 {len(raw_trajectories)} 条轨迹执行动力学滤波 (Kinetic Jitter Filter)...")
    filter_engine = KineticJitterFilter(dt=0.1)
    clean_trajs, noisy_trajs, stats = filter_engine.batch_filter(raw_trajectories)
    print(f"    清洗结果 -> 总数: {stats['total_raw']} | 清洗合格: {stats['clean_count']} | 剔除低质: {stats['noisy_count']}")

    print(f"\n[2/3] 计算 7-DoF 相对位姿动作空间分位数...")
    canonicalizer = ActionCanonicalizer()
    if clean_trajs:
        all_actions = np.concatenate([t["actions"] for t in clean_trajs], axis=0)
        canonicalizer.fit(all_actions)

    print(f"\n[3/3] 保存标准数据集到: {os.path.abspath(output_dir)} ...")
    num_clean = len(clean_trajs)
    n_train = int(num_clean * 0.80)
    n_val = int(num_clean * 0.10)

    train_split = clean_trajs[:n_train]
    val_split = clean_trajs[n_train: n_train + n_val]
    test_split = clean_trajs[n_train + n_val:]

    np.save(os.path.join(output_dir, "train_trajectories.npy"), train_split, allow_pickle=True)
    np.save(os.path.join(output_dir, "val_trajectories.npy"), val_split, allow_pickle=True)
    np.save(os.path.join(output_dir, "test_trajectories.npy"), test_split, allow_pickle=True)
    np.save(os.path.join(output_dir, "noisy_preference_trajectories.npy"), noisy_trajs, allow_pickle=True)

    print("\n" + "=" * 80)
    print(f"  [🎉 提取成功] 数据已就绪并保存至: {os.path.abspath(output_dir)}")
    print(f"  - 训练集: {len(train_split)} 条轨迹")
    print(f"  - 验证集: {len(val_split)} 条轨迹")
    print(f"  - 测试集: {len(test_split)} 条轨迹")
    print(f"  - 现在可直接运行训练: python vla/run_pipeline.py --mode full --device cuda")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Mini OpenX BridgeData")
    parser.add_argument("--num_trajectories", type=int, default=500, help="Number of trajectories to stream (default: 500)")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="Output directory")
    parser.add_argument("--local_tfrecord_dir", type=str, default=None, help="【推荐】本地bridge_dataset tfrecord文件夹路径，优先使用本地，跳过网络拉取")
    args = parser.parse_args()

    extract_mini_bridgedata(
        num_trajectories=args.num_trajectories,
        output_dir=args.output_dir,
        local_tfrecord_dir=args.local_tfrecord_dir
    )
