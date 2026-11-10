# OrganRelation3D：基于动态区域关系推理的脑肿瘤多标签分割

**简体中文** | English（待补充）

![Python](https://img.shields.io/badge/Python-3.11-blue)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
![Dataset](https://img.shields.io/badge/Dataset-BraTS%202018-success)
![Backend](https://img.shields.io/badge/GPU-ROCm%20%7C%20CUDA-orange)

## 项目简介

本项目研究多模态 MRI 脑肿瘤分割中的**区域关系建模**。BraTS 要求同时分割三个相互嵌套的区域：全肿瘤（WT）、肿瘤核心（TC）和增强肿瘤（ET）。多数方法把它们当成三个独立的输出；本方法则把三个区域各自表示成一个"节点"，让节点之间通过动态学习的有向边交换信息，再把关系推理的结果写回特征图并指导解码。

模型以 T1、T1ce、T2、FLAIR 四个模态为输入，**整个脑体积一次前向**（不切 patch、不滑窗），输出三个可重叠的区域概率。代码完全基于标准 PyTorch 实现，同时支持 AMD ROCm 与 NVIDIA CUDA。

**当前进度（2026 年 11 月）**：BraTS2018 训练集 285 例的五折交叉验证主实验已完成，五折平均 Dice 为 **0.814**（WT 0.895 / TC 0.837 / ET 0.709）。第一档消融与基线实验（去掉关系推理、去掉层级一致性损失、双感受野主干、3D U-Net）正在进行。

## 亮点

- **区域关系推理**：WT/TC/ET 三个节点、六条有向边，边权由节点特征和几何信息（质心、体积、置信度）动态计算，多轮消息传递后写回最深层特征。
- **关系驱动的尺度自适应解码**：由关系推理结果决定每个区域在各个解码尺度上使用多少区域信息。
- **层级一致性约束**：在损失中软约束 ET ⊆ TC ⊆ WT 的嵌套关系。
- **整例训练的工程优化**：深度切片卷积、空洞卷积的等价交错执行、整网 bf16 混合精度与整脑外接框裁剪，在 RX 7900 XTX 上把每轮训练从 44 分钟降到约 4.4 分钟，显存峰值约减半。
- **严格可恢复**：checkpoint 保存模型、优化器、全部随机状态、病例顺序与日志位置，续跑前严格核对配置与代码版本；中断后**用同一条命令即可续跑**（CPU 测试中续跑轨迹与不中断逐位一致）。
- **消融只改配置**：每个消融实验与主实验只差一项设置，由测试自动核对。

## 方法概览

| 模块 | 作用 |
| --- | --- |
| 双感受野编码器 | 四级编码（32/64/128/256 通道），每级并联普通卷积与空洞卷积，并按两路特征及其差异学习融合权重 |
| 空间到节点 | 由最深层的粗预测得到三个区域的软掩码，把特征聚合成 WT/TC/ET 三个节点，并附带质心、体积、置信度 |
| 动态关系推理 | 三节点完全图上进行 K 轮消息传递，边权每轮重新计算，GRU 更新节点 |
| 深层回写 | 节点通过注意力写回最深层特征图：F' = F + γ·φ(G) |
| 尺度自适应解码器 | 在三个跳连尺度上用节点查询注入区域信息，尺度权重由关系推理结果决定 |
| 联合损失 | 最终输出与粗预测的 BCE + Dice，加上最终输出的层级一致性项 |

公式级的实现说明见 [METHOD_SPEC.md](./METHOD_SPEC.md)。

## 实验结果

BraTS2018 训练集 285 例，按患者随机分成五折；每折 228 例训练、57 例验证，训练至早停。下表为各折最佳轮的验证集 Dice 平均值 ± 标准差：

| 指标 | 平均 | WT | TC | ET |
| --- | ---: | ---: | ---: | ---: |
| 最佳轮 | 0.814 ± 0.015 | 0.895 ± 0.010 | 0.837 ± 0.026 | 0.709 ± 0.044 |
| 最后 5 轮平均 | 0.805 ± 0.015 | 0.891 | 0.822 | 0.702 |

验证集同时用于早停和选取最佳轮，因此"最佳轮"略偏乐观，"最后 5 轮平均"更保守。计分采用逐病例 Dice，真实与预测都为空的区域不计入，未做任何后处理。完整配置、逐折结果、困难病例分析和消融进展见 [EXPERIMENTS.md](./EXPERIMENTS.md)。

## 环境配置

本项目已在以下环境运行：

| 操作系统 | GPU | 后端 | 版本 | 状态 |
| --- | --- | --- | --- | --- |
| Ubuntu 24.04 | AMD Radeon RX 7900 XTX 24G | ROCm 7.2 | PyTorch 2.11.0+rocm7.2，Python 3.11 | 主实验环境 |
| Windows 11 | NVIDIA RTX 5060 8G | CUDA 12.8 | PyTorch 2.8.0+cu128 | 辅助环境，正式训练待验证 |

整例训练的显存峰值约 5–6 GiB（BraTS2018 最大病例，整网 bf16 混合精度）。

### 1. 克隆仓库

```bash
git clone https://github.com/jackygeng2001/OrganRelation3D.git
cd OrganRelation3D
```

### 2. 创建 conda 环境

```bash
conda create -n organrelation3d python=3.11 -y
conda activate organrelation3d
```

### 3. 安装 PyTorch

按设备选择**其中一种**：

- AMD ROCm：

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
```

- NVIDIA CUDA：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 4. 安装其余依赖并运行测试

```bash
pip install -r requirements.txt
python scripts/run_tests.py --suite all
```

测试全部在 CPU 上用合成数据运行，覆盖公式、梯度、混合精度、续训和消融变体；有任何跳过都算失败。

## 数据准备

本项目使用 **BraTS 2018** 训练集（285 例，含 HGG 与 LGG），需自行从官方渠道申请下载。

1. 把下载的压缩包放到 `datasets/BraTS2018/raw/brats2018.zip`。原始数据只读，程序不会解压或修改它。
2. 生成按患者划分的五折（固定种子，结果可复现）：

```bash
python scripts/brats_folds.py --cohort brats2018 --source datasets/BraTS2018/raw/brats2018.zip --seed 20260925 --output-dir configs/brats2018_folds
```

3. 预先生成预处理缓存（可选，约 16 GB，能显著加快训练）：

```bash
python scripts/brats_cache.py --config configs/experiments/main.json --cohort brats2018 --source-partition training --source datasets/BraTS2018/raw/brats2018.zip --data-cache cache/npy
```

4. 复制本机配置模板，并按需要修改路径和设备：

```bash
cp configs/machine.example.json configs/machine.local.json
```

本机配置记录数据、划分、缓存、输出目录、设备和启动环境变量，不提交到 git。NVIDIA 显卡请把其中的 `environment` 改为 `{}`；AMD 显卡保留 `PYTORCH_MIOPEN_SUGGEST_NHWC=1`。

## 训练与评估

### 运行一个实验的五折

```bash
python scripts/run_experiment.py configs/experiments/main.json
```

`configs/experiments/` 下每个文件是一个实验：`main.json` 为完整方法，`no_relation.json`、`no_hier.json`、`drf_backbone.json`、`unet.json` 为消融与基线。训练结果写入 `runs/<数据集>_<实验名>_fold<k>/`。

训练中断后，**再次执行同一条命令即可续跑**：已早停结束的折自动跳过，未完成的折从断点继续。只想跑部分折时加 `--folds 0 1`，只想查看将要执行的命令时加 `--dry-run`。

### 训练前检查显存

用最大病例做一次前向和反向，不更新参数：

```bash
python scripts/brats.py check --config configs/experiments/main.json --device cuda:0 --convolution-backend depth_sliced --cohort brats2018 --source-partition training --source datasets/BraTS2018/raw/brats2018.zip --data-cache cache/npy --case Brats18_TCIA13_653_1
```

### 查看结果

- 训练曲线（训练损失、验证 Dice 等）：

```bash
tensorboard --logdir runs
```

- 逐轮对比多个实验的验证 Dice 与训练速度：

```bash
python scripts/brats_compare_runs.py runs/<实验A>_fold0 runs/<实验B>_fold0
```

每个运行目录中的 `metrics.jsonl` 是完整的逐步日志，`validation/` 保存每次验证的逐病例结果，`best-dev.ckpt` 与 `last.ckpt` 分别为最佳轮与最新的 checkpoint。

### 检查关系推理是否真的在起作用

只用 CPU，读取同一折的若干 run（完整方法或 `no_relation`），在该折验证病例上测量回写占比、各轮边权和消息贡献，并在推理时逐项敲除（γ = 0、η = 0、边权置 0、跳过关系轮次），看 Dice 变化：

```bash
python scripts/brats_relation_probe.py probe --run E1=runs/<完整方法>_fold0 --run A3=runs/<no_relation>_fold0 --data-cache cache/npy --output <输出目录>/probe_fold0.jsonl --threads 8
python scripts/brats_relation_probe.py summarize <输出目录>/probe_fold0.jsonl
```

输出文件写在运行目录和数据目录之外；中断后再执行同一条命令会跳过已完成的病例。训练时与当前代码的 `src/` 不同（例如训练后代码有过重构）时，需加 `--allow-source-change`，差异会记在输出表头里。

## 相关文档

- [EXPERIMENTS.md](./EXPERIMENTS.md)：实验报告及日志，记录全部实验配置、进度与结果
- [METHOD_SPEC.md](./METHOD_SPEC.md)：方法实现规格，逐公式说明实现与工程等价变换
- [environments/](./environments)：各设备环境的详细说明
- [AGENTS.md](./AGENTS.md)：AI 协作开发规范
