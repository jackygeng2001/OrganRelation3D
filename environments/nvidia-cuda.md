# NVIDIA CUDA 辅助环境

## 配置

| 项目 | 版本 |
| --- | --- |
| 系统 | Windows 11 |
| GPU | NVIDIA RTX 5060 8GB |
| PyTorch | 2.8.0+cu128 |

**状态：尚未在本项目中验证，不能直接用于正式实验。** 计划只用于第一档消融中的 3D U-Net 基线（显存需求最低）。

## 投入使用前的验证步骤

在 PowerShell 中、仓库根目录下依次完成：

1. 检查环境（只在缺少时补装 `nibabel`、`tensorboard`、`tqdm`、`pillow`，不要动 PyTorch）：

   ```
   python -c "import torch, numpy, nibabel, tensorboard; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
   ```

2. 跑全套 CPU 测试：`python scripts\run_tests.py --suite all`
3. 生成五折划分或拷入 AMD 机器上的划分文件（`configs/brats2018_folds/*.local.json`），并生成 npy 缓存（见 README）。
4. 复制 `configs/machine.example.json` 为 `configs/machine.local.json`，把 `environment` 改为 `{}`，路径改成本机路径（Windows 路径中的反斜杠要写成 `\\` 或 `/`）。
5. 最大病例显存预检：

   ```
   python scripts\brats.py check --config configs\experiments\main.json --device cuda:0 --convolution-backend depth_sliced --cohort brats2018 --source-partition training --source datasets\BraTS2018\raw\brats2018.zip --data-cache cache\npy --case Brats18_TCIA13_653_1
   ```

6. 一致性检查：用主方法跑 fold0 的前 3 轮（加 `--stop-after 684`），与 AMD 上主实验 fold0 的前 3 轮对比训练损失和验证 Dice，同时实测每步耗时。

全部通过后，再用 `python scripts\run_experiment.py configs\experiments\unet.json` 运行。若 8GB 显存不够，按项目规范停下来报告，不缩小网络、不改成 patch 训练。
