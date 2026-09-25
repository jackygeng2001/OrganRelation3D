# AMD 主实验环境

用户指定：Ubuntu 24.04.4 LTS、RX 7900 XTX 24GB、PyTorch 2.11.0 + ROCm 7.2。
本项目尚未验证该设备，不安装或修改现有环境。当前统计脚本仅使用标准库，Python >=3.10 可直接运行，无 GPU 依赖。
后续提供独立测试命令，由用户 GitHub 同步后手动运行返回结果。ROCm 安装/核验文档仅在对应任务获准后补齐，不能执行 CUDA 环境安装方案。

第一阶段骨干已提供配置和 `DeviceMemoryMonitor`，以及 scripts/check_backbone.py 合成诊断入口。
后续获准时可在现有环境手动使用 `--device cuda:0`（ROCm 使用相同 PyTorch device API）；
具体命令和指标限制见 [backbone_stage1.md](../docs/backbone_stage1.md)。当前只预留入口，未在 AMD 上运行。
不得安装 `requirements-backbone-cpu.txt` 覆盖此环境，微型骨干探针不代表完整模型显存验收。
