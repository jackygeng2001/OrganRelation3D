# Windows CPU 开发环境

元数据统计及其测试仅依赖 Python >=3.10 标准库，无 pip 安装步骤、不导入 torch。
本轮已使用 Codex 提供的 Python 3.12.14 Windows 运行时；本机可用 python 命令时按 README 执行即可。
骨干 CPU 环境另设 `.venv-backbone-cpu`，本轮已验证 PyTorch 2.8.0+cpu；不等于已验证完整研究方法。

真实 CT 重采样保真度使用独立 `.venv-resampling`（Python 3.11+）与
`requirements-resampling-cpu.txt` 中锁定的 NumPy、SciPy、NiBabel、Pillow、psutil。
这不包含 torch，不修改任何已有 GPU 环境。安装和运行命令见
[保真度协议](../docs/fidelity_protocol.md)。没有安装这些可选包时，新保真度测试会跳过；
原有标准库统计测试仍可运行。不能将有跳过的结果报告为全部保真度测试通过。

## 骨干环境

独立安装说明及用户手动下载命令见 [骨干第一阶段](../docs/backbone_stage1.md)。
`requirements-backbone-cpu.txt` 仅供 Windows CPU 开发使用，不用于 AMD/CUDA 环境。
测试记录见 [backbone_cpu_validation.md](../docs/backbone_cpu_validation.md)。
当前骨干环境仅配置 PyTorch 必需依赖，NumPy 桥接未配置；Torch 导入会给出可选 NumPy 提示，
纯 Tensor 前后向、数值梯度与参数检查已通过。已有重采样环境包含 NumPy/SciPy，保持不变。
数据到 Tensor 的联合流水线留到后续任务，不据本轮测试声称已验证。
