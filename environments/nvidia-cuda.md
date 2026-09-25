# NVIDIA 辅助环境

用户指定：Windows 11、RTX 5060 8GB、PyTorch 2.8.0 + CUDA 12.8。
仅用于重要集成节点 CUDA 兼容、独立模块和缩小输入测试，不承担正式训练。本项目尚未验证该设备，不安装或修改现有环境。
当前统计脚本只依赖 Python >=3.10 标准库，不需要 CUDA。不能用本环境安装方案覆盖 AMD ROCm 环境。

第一阶段骨干的 CPU 基线为 PyTorch 2.8.0+cpu；这仅证明 CPU 测试，不证明本卡已兼容。
后续重要集成节点可在既有 CUDA 环境使用微型配置与 scripts/check_backbone.py 的显式 device 入口。
不安装 CPU requirements 覆盖本环境，不因 8GB 容量改变正式方法。详见 [backbone_stage1.md](../docs/backbone_stage1.md)。
