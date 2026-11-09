# AMD ROCm 主实验环境

## 已验证配置

| 项目 | 版本 |
| --- | --- |
| 系统 | Ubuntu 24.04 |
| GPU | AMD Radeon RX 7900 XTX 24GB |
| ROCm | 7.2 |
| conda 环境 | `organrelation3d`，Python 3.11.16 |
| PyTorch | 2.11.0+rocm7.2 |
| 其余依赖 | NumPy 2.3.5、NiBabel 5.3.2、TensorBoard 2.20.0、tqdm 4.67.1、Pillow 12.3.0 |

这是全部正式实验所用的环境，不要升级或降级 PyTorch、ROCm、NumPy，以免续训时代码与环境身份对不上。

## 启动前必须的环境变量

模型以 channels_last_3d 布局运行。ROCm 下必须在 Python 启动**之前**设置：

```bash
export PYTORCH_MIOPEN_SUGGEST_NHWC=1 MIOPEN_ENABLE_LOGGING=0
```

缺少时训练入口会在读取数据前报错，不会自动设置或回退。用 `scripts/run_experiment.py` 时，这两个变量写在 `configs/machine.local.json` 的 `environment` 里，由入口显式传给每一折，并记录进 run 的身份信息。

## 性能与显存

- 整网 bf16 混合精度、深度切片卷积、空洞卷积交错执行：BraTS2018 每步约 1.07 秒，每轮（228 例训练 + 57 例验证）约 4.4 分钟。
- 最大病例（144×192×152，裁剪后）显存峰值：完整方法约 5.3 GiB（已分配），PyTorch 预留约 6.1 GiB。`nvidia-smi`/`rocm-smi` 显示的占用会更高，因为包含缓存分配器预留的部分。
- MIOpen 对空洞率 2 的 3×3 卷积会退回到极慢的 naive 内核，这正是交错执行要绕开的问题；不要关闭 `--convolution-backend depth_sliced`。

## TensorBoard

```bash
tensorboard --logdir runs --host 127.0.0.1 --port 6006 --load_fast=false
```

然后在本机浏览器打开 http://127.0.0.1:6006。
