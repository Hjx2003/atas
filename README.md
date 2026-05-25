# ATAS (Advanced Token Alignment and Self-Distillation) Training

本项目用于复现 **ATAS** 训练流程：基于预训练 CLIP 架构，使用 ImageNet 无标注图像进行自蒸馏训练。在训练阶段，算法会冻结 Teacher CLIP 模型的全部参数，仅更新 Student 模型的 Image Encoder。通过结合全局与局部表征能力，系统采用 **GLD (Global Language-image Distillation)**、**LLD (Local Language-image Distillation)** 和 **GGD (Global-local Geometric Distillation)** 三个核心损失函数进行联合优化。

---

## 📌 项目结构

根据当前代码库的实际目录组织，项目结构如下：

```
ATAS/
├── checkpoints/             # 模型检查点保存目录（权重、优化器状态等）
├── configs/                 # 配置文件目录
│   ├── detection/           # 检测任务相关配置
│   ├── pretrain/            # 预训练阶段配置
│   │   └── atas_imagenet.yaml  # ImageNet 自蒸馏预训练核心配置文件
│   └── segmention/          # 分割任务相关配置
├── docs/                    # 项目相关文档与说明
├── logs/                    # 训练及评估日志输出目录
├── outputs/                 # 训练中间结果、可视化或预测输出
├── scripts/                 # 执行脚本目录
│   ├── eval_ovseg.py        # 开放词汇分割（Open-Vocabulary Segmentation）评估脚本
│   └── train_atas.py        # ATAS 自蒸馏预训练核心启动脚本
└── src/                     # 项目核心源代码目录
    ├── config/              # 配置解析与管理模块
    │   ├── __init__.py
    │   └── config.py        # 配置文件读取、参数初始化与校验
    ├── datasets/            # 数据加载与增强模块
    │   ├── __init__.py
    │   ├── dataset.py       # 自定义数据集定义与 ImageFolder 封装
    │   └── mosaic.py        # Mosaic 数据增强算法实现（用于多尺度/局部特征增强）
    ├── losses/              # 损失函数核心实现模块
    │   ├── __init__.py
    │   └── atas_losses.py   # 包含 GLD、LLD、GGD 联合损失函数的具体实现
    ├── models/              # 模型架构定义模块
    │   ├── __init__.py
    │   └── atas.py          # ATAS 核心网络结构（Student/Teacher 交互及特征提取）
    └── utils/               # 基础通用工具模块
        └── __init__.py      # 分布式通信、日志输出、权重保存等辅助工具

```

---

## 🛠️ 环境准备

建议在 Linux 环境下使用 Conda 进行环境隔离与管理。

### 1. 创建并激活 Conda 环境

```bash
conda create -n atas python=3.10 -y
conda activate atas

```

### 2. 安装 PyTorch 及相关依赖

根据您的 CUDA 版本（示例为 CUDA 12.1），运行以下命令：

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```

### 3. 安装项目依赖

在项目根目录下，通过 `requirements.txt` 完成其余基础库的安装（如未创建该文件，可根据 `src/` 中的导入需求安装 `yaml`, `tqdm`, `swanlab` 等）：

```bash
# 请确保根目录下存在或自行创建 requirements.txt
pip install -r requirements.txt

```

### 4. 实验跟踪与可视化工具（选填）

如果项目中使用 `swanlab` 记录训练指标，请提前初始化登录：

```bash

# 若使用 swanlab
swanlab login

```

---

## 📂 数据准备

项目默认采用 **ImageNet 训练集** 作为无监督自蒸馏的输入数据。为了兼容 PyTorch 的数据加载机制，目录结构建议保持标准 `ImageFolder` 格式：
根据存储路径修改atas_imagenet.yaml中dataset.root的路径
```txt
data/
└── imagenet/
    └── train/
        ├── n01440764/
        │   ├── xxx.JPEG
        │   └── ...
        ├── n01443537/
        │   ├── xxx.JPEG
        │   └── ...
        └── ...

```
---

## ⚙️ 配置文件说明

核心预训练配置文件位于 `configs/pretrain/atas_imagenet.yaml`。
该文件内集成了超参数控制、数据增强策略以及损失函数权重。主要包含以下核心配置块：

* **model**: 指定 CLIP 骨干网络版本（如 ViT-B/16）、Student 层的初始化配置。
* **train**: 学习率（Learning Rate）、Batch Size、总 Epoch 数、权重衰减（Weight Decay）等。
* **loss**: `gld_weight`、`lld_weight`、`ggd_weight` 三个损失函数的平衡系数。
* **dataset**: 图像输入尺寸、Mosaic 增强概率及相关超参数。

---

## 🚀 模型训练

本项目基于 PyTorch DDP（Distributed Data Parallel）分布式架构设计，推荐使用 `torchrun` 进行多卡并行训练。

### 1. 正常启动预训练

运行以下命令启动多 GPU 并行训练（以 4 卡为例）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --nproc_per_node=4 \
  scripts/train_atas.py \
  --config configs/pretrain/atas_imagenet.yaml

```

### 2. 从断点恢复训练 (Resume)

如果训练因不可抗力中断，可通过指定 `--resume` 参数加载最近的检查点，恢复优化器、学习率调度器及 Epoch 状态：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --nproc_per_node=4 \
  scripts/train_atas.py \
  --config configs/pretrain/atas_imagenet.yaml \
  --resume checkpoints/pretrain/atas_epoch_3.pt

```

---

## 📊 日志与可视化

### 1. 本地日志

训练过程中的文本日志（包含时间戳、各 Loss 细项、学习率、吞吐量等）会实时输出至终端，并持久化保存于：

```txt
logs/atas/train.log

```

### 2. 在线看板 (SwanLab / Wandb)

若在配置文件 `configs/pretrain/atas_imagenet.yaml` 中启用了监控面板（例如 SwanLab）：

```yaml
swanlab:
  enable: true
  project: "ATAS_Pretrain"
  experiment_name: "imagenet_self_distill"

```

训练过程中的总体 `loss`、`gld_loss`、`lld_loss`、`ggd_loss` 以及学习率（`lr`）等关键指标将自动同步至云端看板，便于远程监控训练趋势。

---

## 💾 检查点 (Checkpoint) 说明

模型将根据配置文件中设定的 `save_interval`（如每个 Epoch 或每 N 个 Step）自动在以下路径保存检查点：

```txt
checkpoints/pretrain/

```

每个生成的 `.pt` 权重文件均为一个复合字典，包含恢复完整训练现场所需的全部状态：

* `student_state_dict`: Student 模型 (Image Encoder) 的当前可训练权重。
* `teacher_state_dict`: Teacher 模型权重（保持冻结，主要供校验或多节点同步确认使用）。
* `optimizer_state_dict`: 优化器（如 AdamW）的动量与平方梯度历史状态。
* `scaler_state_dict`: 混合精度训练（AMP）的梯度缩放器状态。
* `epoch`: 当前迭代的完整周期数。
* `global_step`: 全局迭代步数。
* `config`: 训练时所采用的完整配置参数镜像。

保存示例：

```txt
checkpoints/pretrain/atas_epoch_1.pt

```

## 🚀 下游任务

### 1. 开放词汇语义分割VOC2012测试

CUDA_VISIBLE_DEVICES=0 python scripts/eval_ovseg.py   --config configs/segmention/voc2012_ovseg.yaml

---

## ⚠️ 注意事项与 FAQ

1. **非从零训练**：ATAS 算法并非从头开始盲训 CLIP，而是在已有开源预训练 CLIP 权重的良好表征基础上，通过自蒸馏进一步对齐或增强 Image Encoder 的多尺度/局部特征表达能力。
2. **权重冻结**：在预训练全过程中，**Teacher 模型始终处于冻结状态**（`eval()` 模式），不计算梯度，仅提供稳定的表征蒸馏靶向信号。
3. **模型初始化**：训练启动时，Student 模型的 Image Encoder 默认会初始化为 Teacher 模型对应权重的精确拷贝。
4. **Batch Size 的特殊机制**：
* 配置文件中的 `train.batch_size` 机制默认表示 **Mosaic 基础批大小**。
* 由于引入了多尺度局部裁剪与拼接，实际输入到 Dataloader 的单卡物理 Batch Size 为 `batch_size * 4`。
* 如果在训练初始化阶段遭遇显存溢出（OOM），请优先调小配置文件中的 `train.batch_size`。


5. **任务下游评估**：预训练完成后，可使用 `scripts/eval_ovseg.py` 脚本无缝对接到开放词汇分割（Open-Vocabulary Segmentation）等下游稠密预测任务中，用以定量评估局部特征对齐的效果。


