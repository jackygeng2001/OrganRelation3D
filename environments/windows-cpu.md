# Windows CPU 开发环境

元数据统计及其测试仅依赖 Python >=3.10 标准库，无 pip 安装步骤、不导入 torch。
本轮已使用 Codex 提供的 Python 3.12.14 Windows 运行时；本机可用 python 命令时按 README 执行即可。
后续 CPU 模型环境另行配置，不代表当前已安装或验证 PyTorch。

真实 CT 重采样保真度使用独立 `.venv-resampling`（Python 3.11+）与
`requirements-resampling-cpu.txt` 中锁定的 NumPy、SciPy、NiBabel、Pillow、psutil。
这不包含 torch，不修改任何已有 GPU 环境。安装和运行命令见
[保真度协议](../docs/fidelity_protocol.md)。没有安装这些可选包时，新保真度测试会跳过；
原有标准库统计测试仍可运行。不能将有跳过的结果报告为全部保真度测试通过。
