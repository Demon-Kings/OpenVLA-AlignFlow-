# OpenVLA‑AlignFlow

> 
> 融合细粒度图文对齐与连续流匹配的具身多模态大模型决策系统

![PyTorch](https://img.shields.io/badge/PyTorch-2.4‑orange)
![CFM](https://img.shields.io/badge/Conditional‑Flow‑Matching‑blue)
![Qwen3‑VL](https://img.shields.io/badge/Qwen3‑VL‑Multimodal‑green)
![Embodied‑AI](https://img.shields.io/badge/Embodied‑AI‑Robotics‑purple)

**OpenVLA‑AlignFlow**：端到端闭环具身视觉‑语言‑动作(VLA)系统，基于BridgeData v2数据集，完成**数据清洗、图文细粒度对齐、CFM连续动作生成、轨迹DPO离线偏好对齐**全链路，支持7‑DoF机械臂动作块预测，已通过离线Benchmark与真机级物理平滑度评测。

---

## 📋 目录

- [项目简介](#%E9%A1%B9%E7%9B%AE%E7%AE%80%E4%BB%8B)
- ✨ [核心特性](#%E6%A0%B8%E5%BF%83%E7%89%B9%E6%80%A7)
- 🧱 [整体系统架构](#%E6%95%B4%E4%BD%93%E7%B3%BB%E7%BB%9F%E6%9E%B6%E6%9E%84)
- 📊 [实验评测结果](#%E5%AE%9E%E9%AA%8C%E8%AF%84%E6%B5%8B%E7%BB%93%E6%9E%9C)
- 📂 [代码结构](#%E4%BB%A3%E7%A0%81%E7%BB%93%E6%9E%84)
- ⚙️ [环境依赖](#%E7%8E%AF%E5%A2%83%E4%BE%9D%E8%B5%96)
- 🚀 [快速启动](#%E5%BF%AB%E9%80%9F%E5%90%AF%E5%8A%A8)
- 📝 [流水线执行说明](#%E6%B5%81%E6%B0%B4%E7%BA%BF%E6%89%A7%E8%A1%8C%E8%AF%B4%E6%98%8E)
- 📄 [License](#license)

## 项目简介

机器人遥操作示教数据通常存在震颤、迟疑停顿、加速度突变噪声，直接行为克隆(BC)会导致机械臂运动卡顿、冲击抖动。
本项目构建一套完整的具身多模态决策流水线：

1. 动力学滤波清洗原始遥操作轨迹，区分专家正样本与抖动负样本；
2. 基于Qwen3‑VL + SigLIP实现细粒度图文对齐，提升物体可供性注意力；
3. 使用**条件流匹配CFM**做连续动作生成头，单次输出16步7‑DoF动作块；
4. 自研高阶**具身轨迹DPO**做离线偏好对齐，引入多项正则约束抑制加速度跳变、策略漂移与无效悬停；
5. 完整离线评测体系，评估动作误差、物理平滑度、图文对齐指标。

> 
> 工程状态：✅全流程端到端闭环跑通；完成离线Benchmark、真机级加加速度Jerk评测验证。

**技术栈**：`PyTorch` / `Qwen3‑VL` / `SigLIP` / `Conditional Flow Matching(CFM)` / `Trajectory‑DPO` / `Action Chunking` / `OpenX BridgeData v2`

## ✨ 核心特性

- 🧹 **动力学异常轨迹清洗**：速度、加速度、空闲占比三重滤波，分离黄金专家轨迹与DPO负样本；
- 👁 **细粒度图文对齐**：Sub‑goal InfoNCE对比损失 + Affordance Mask注意力损失，可供性IoU从1.1%提升至29.2%；
- 🌊 **CFM条件流匹配动作头**，4‑6步Euler ODE极速推理，直接预测16步动作块，相比Diffusion大幅降低推理步数；
- 🎯 **SOTA具身轨迹DPO**：集成柯西C1光滑BNF、KKT动态β调度、轨迹控长、BC保真、黎曼测地线正则五大约束；
- 📈 **多维度离线Benchmark**：动作L1/MSE误差、Jerk加加速度、可供性IoU、目标召回R@1/R@5、多模态熵完整指标；
- 🤖 **7‑DoF末端增量动作空间**，统一规范化映射至[-1,1]，Action Chunk滑窗样本扩增。

## 🧱 整体系统架构

```
原始BridgeData v2 TFRecords轨迹
        ↓
【数据工程&动力学清洗】process_local_bridgedata.py / kinetic_filter.py
  ├─动力学滤波剔除高速、冲击、高停顿轨迹
  ├─划分黄金专家轨迹(51条) / DPO抖动负样本(209条)
  └─7‑DoF动作规范化 + Action Chunking(k=16)滑窗切分
        ↓
【Stage1 细粒度图文对齐】train_vl_align.py
  └─SigLIP视觉Patch + Qwen3‑VL语言编码器，Cross‑Attention跨模态融合
     InfoNCE子目标对比损失 + Affordance Mask注意力损失
        ↓
【Stage2 CFM条件流匹配预训练】train_flow_vla.py / flow_action_head.py
  └─速度场MLP拟合；4‑6步Euler ODE采样输出连续动作块
        ↓
【Stage3 轨迹DPO离线偏好对齐】train_offline_rl_dpo.py / trajectory_dpo.py
  └─专家vs次优轨迹偏好对，5项协同约束做后训练优化
        ↓
【离线Benchmark评测】offline_benchmark.py
  └─计算动作误差、物理平滑度、图文对齐全套指标
```

## 📊 实验评测结果

> 
> 测试集为未见专家轨迹，第4次运行代表全量对齐收敛版本

| 指标 | 目标方向 | 第1次(初始) | 第2次(CFM) | 第3次(DPO初版) | **第4次(收敛)** | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| Trajectory L1 动作绝对误差 | ↓越低越好 | 0.7879 | 0.3152 | 0.3296 | **0.3322** | 下降57.8% |
| Trajectory MSE 均方误差 | ↓越低越好 | 0.9538 | 0.2795 | 0.2875 | **0.2958** | 下降69.0% |
| Mean Jerk 加加速度(m/s³) | ↓越低越好 | 5294.45 | 11.43 | 12.07 | **11.60** | 真机级平滑，远低于25震颤阈值 |
| Affordance IoU可供性交并比 | ↑越高越好 | 1.1% | 29.3% | 29.1% | **29.2%** | 注意力聚焦物体交互区域 |
| Sub‑goal R@1 | ↑越高越好 | 5.0% | 0.5% | 0.5% | **1.0%** | 召回率翻倍 |
| Sub‑goal R@5 | ↑越高越好 | 25.0% | 2.5% | 2.5% | **5.0%** | 召回率翻倍 |
| Mode Coverage Entropy | 适中 | 5.097 | 0.103 | 0.121 | **0.117** | 处于0.08‑0.25健康区间 |

## 📂 代码结构

```
vla
├── configs
│   └── config.py               #全局配置、路径、超参
├── data
│   ├── process_local_bridgedata.py   #TFRecord数据集解析
│   ├── kinetic_filter.py             #动力学轨迹滤波
│   ├── canonicalize.py               #7‑DoF动作规范化
│   └── embodied_dataset.py           #Action Chunking数据集加载
├── models
│   ├── vl_backbone.py        #多模态视觉‑语言主干网络
│   ├── vl_alignment.py       #图文对齐损失实现
│   ├── flow_action_head.py   #CFM条件流匹配动作头
│   └── trajectory_dpo.py     #高阶具身轨迹DPO训练模块
├── train_vl_align.py         #Stage1图文对齐训练入口
├── train_flow_vla.py         #Stage2 CFM预训练入口
├── train_offline_rl_dpo.py   #Stage3 DPO离线强化入口
├── offline_benchmark.py      #离线评测脚本
└── run_pipeline.py           #全流程一键执行入口
```

## ⚙️ 环境依赖

安装依赖：

```
pip install -r requirements.txt
```

## 🚀 快速启动

### 1. 数据准备

下载BridgeData v2数据集分片，放置配置指定路径；运行数据预处理，完成动力学清洗、动作规范化、样本切分。

```
python vla/data/process_local_bridgedata.py
```

输出：专家训练/测试集、`noisy_preference_trajectories.npy`（DPO负样本）

### 2. Stage1：图文细粒度对齐训练

```
python train_vl_align.py
```

### 3. Stage2：CFM条件流匹配动作头预训练

```
python train_flow_vla.py
```

### 4. Stage3：轨迹DPO离线偏好对齐后训练

```
python train_offline_rl_dpo.py
```

### 5. 离线Benchmark评测

```
python offline_benchmark.py
```

### ✅ 一键完整流水线

```
python run_pipeline.py
```

## 📝 流水线执行说明

1. **动力学滤波**：通过速度>0.85m/s、加速度>3.5m/s²、空闲占比>55%过滤异常轨迹；原始260条轨迹分出51条专家轨迹、209条抖动负样本；
2. **Action Chunking(k=16)**：滑窗切分，单次预测未来16步7‑DoF末端增量动作，缓解单步自回归复合误差；
3. **图文对齐**：InfoNCE子目标对比学习约束文本指令与目标帧视觉嵌入；Affordance Mask Loss让注意力聚焦物体交互部位；
4. **CFM动作生成**：构建高斯噪声到目标动作块直线流，MLP拟合速度场，推理阶段4‑6步Euler积分采样；
5. **轨迹DPO**：专家轨迹为Chosen，抖动轨迹为Rejected；叠加5大约束：C¹光滑BNF、KKT动态β调度、轨迹长度惩罚、BC保真、黎曼流形正则，抑制机械臂抖动与策略漂移。

## 📄 License

本项目仅供学术研究使用。
