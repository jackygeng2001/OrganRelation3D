# AMD 主实验环境

用户指定：Ubuntu 24.04.4 LTS、RX 7900 XTX 24GB、PyTorch 2.11.0 + ROCm 7.2。
本项目尚未验证该设备，不安装或修改现有环境。当前统计脚本仅使用标准库，Python >=3.10 可直接运行，无 GPU 依赖。
完整模型测试命令见下文，由用户 GitHub 同步后手动运行返回结果。使用已有 ROCm 环境，不能执行 CUDA 环境安装方案。

第一阶段骨干已提供配置和 `DeviceMemoryMonitor`，以及 scripts/check_backbone.py 合成诊断入口。
后续获准时可在现有环境手动使用 `--device cuda:0`（ROCm 使用相同 PyTorch device API）；
具体命令和指标限制见 [backbone_stage1.md](../docs/backbone_stage1.md)。当前只预留入口，未在 AMD 上运行。
不得安装 `requirements-backbone-cpu.txt` 覆盖此环境，微型骨干探针不代表完整模型显存验收。

## 完整模型单病例验证（待用户在 AMD 实测）

使用已配置好的 ROCm Python 环境，不执行 CPU/CUDA PyTorch 安装命令。完整数据模块需要 Python>=3.11 和 NumPy、SciPy、NiBabel；先检查已有依赖：

```bash
python -c "import torch,numpy,scipy,nibabel; print(torch.__version__,torch.version.hip); print(numpy.__version__,scipy.__version__,nibabel.__version__); print(torch.cuda.is_available())"
```

仅缺影像依赖时，由用户手动执行 `python -m pip install -r environments/requirements-fullscan.txt`；这个文件不包含 torch。若现有依赖不同，先记录实际版本和依赖冲突，不盲目覆盖 ROCm 环境。大型文件继续由用户下载。本轮开发端没有安装/修改该环境。

从 clone 的仓库根目录运行。先检查数据链路；A/B/C 全保留，通过 --candidate 切换，每次选择新的 JSON 路径：

```bash
python -B scripts/validate_full_scan.py --data-root /path/to/amos22 --case amos_0043 --candidate B --backend rocm --device cuda:0 --through preprocess --output reports/amos_0043_B_preprocess.json
python -B scripts/validate_full_scan.py --data-root /path/to/amos22 --case amos_0043 --candidate B --backend rocm --device cuda:0 --through transfer --output reports/amos_0043_B_transfer.json
```

正式容量尚未冻结。下面显式使用微型模型，只检查真实完整 CT 的算子兼容与工程链路，不能当成正式网络容量验收：

```bash
python -B scripts/validate_full_scan.py --data-root /path/to/amos22 --case amos_0043 --candidate B --backend rocm --device cuda:0 --model-config configs/segmentor_micro.json --allow-micro-model --through backward --output reports/amos_0043_B_micro_backward.json
python -B scripts/validate_full_scan.py --data-root /path/to/amos22 --case amos_0043 --candidate B --backend rocm --device cuda:0 --model-config configs/segmentor_micro.json --allow-micro-model --through step --optimizer adamw --lr 0.0001 --output reports/amos_0043_B_micro_step.json
```

AdamW/lr 是上述单步探针的显式示例，不是正式训练参数。SGD/AdamW 可选；weight_decay 默认 0，SGD momentum 默认 0，AdamW betas=(0.9,0.999)、optimizer epsilon=1e-8，foreach=False，AdamW fused=False，均记录到结果。这一 optimizer epsilon 与方法 epsilon=1e-6 是不同参数。只有 step 阶段建立 optimizer。第一步可能分配 AdamW 状态；不代表后续稳态/第二次前向的峰值。

正式容量决定后，用含完整 model 字段的配置 JSON 替换 --model-config，去掉微型开关。必须保留 model.epsilon=1e-6；K/Cr/da/Cg/各层宽度等全部显式指定，不允许从微型配置暗中推断。--through forward/loss/backward/step 都运行必要前置阶段。

入口固定 FP32/batch 1，保留检查，不使用 AMP 或 warmup。每个 stage 先同步并重置 allocator 峰值，再计时操作并同步；峰值包含先前仍存活的张量。独立列出模型/optimizer 初始化、梯度检查、参数有限性检查；这些检查不混入 backward/step 时间，但 forward 内既有 finite 检查仍在计时内。

JSON 包含病例、候选/实际网格、输入 shape、模型与 baseline 配置/哈希、参数量、每阶段时间/显存/OOM、环境和 Git 状态。mem_get_info 可用时记录整卡瞬时 free/total/used（包括其他进程）；不支持时为 null 并保存原因，不称整卡峰值。不调用外部显卡工具或伪造整卡统计。

退出码 0：执行到请求阶段通过；2：失败，查看 status/error/active_stage/stages。OOM 不重试、不换输入、不自动改候选，进程退出后先分析结果。每病例/候选建议独立进程，避免上一例 allocator/optimizer 状态影响比较。请返回 JSON；结果是工程可行性信息，不是 Dice 或模型性能。
