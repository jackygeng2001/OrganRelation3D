# Windows CPU 开发环境

训练观测测试另需 `environments/requirements-monitoring.txt`（TensorBoard 2.20.0、tqdm 4.67.1、setuptools 80.9.0）。在独立 CPU 环境中用 `python -m pip install -r environments/requirements-monitoring.txt` 安装，不涉及 GPU/PyTorch 安装方案。本机本轮已补齐这些小型依赖，将 setuptools 从 84.0.0 调整到 80.9.0 以满足 TensorBoard CLI 的 pkg_resources 导入，并用真实 event 文件测试 purge、flush/close；影像依赖仍按下文已有 PYTHONPATH 方式组合。`--suite training` 同时运行训练状态和观测层测试，`--suite all` 回归全部测试。

元数据统计及其测试仅依赖 Python >=3.10 标准库，无 pip 安装步骤、不导入 torch。
本轮已使用 Codex 提供的 Python 3.12.14 Windows 运行时；本机可用 python 命令时按 README 执行即可。
张量 CPU 环境沿用 `.venv-backbone-cpu`，已验证 PyTorch 2.8.0+cpu 下完整 Segmentor 与 JointLoss 的合成公式和梯度；不等于真实 CT 或 GPU 验证。

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
完整数据到 Tensor 与微型 Segmentor/JointLoss/单步 optimizer 已通过下述进程级联合 CPU 测试；尚无独立合并环境的安装验收。

## 当前测试与包安装

从项目根目录分别运行：

```powershell
.\.venv-backbone-cpu\Scripts\python.exe -B scripts/run_tests.py --suite tensor
.\.venv-resampling\Scripts\python.exe -B scripts/run_tests.py --suite data
```

分别为 137 项与 53 项，入口拒绝将跳过当作通过。依赖齐全的单一 CPU 环境可运行 --suite all，但本轮不混装两套依赖。
已有 setuptools>=68 的开发环境可离线安装本项目：

```powershell
.\.venv-backbone-cpu\Scripts\python.exe -m pip install --no-deps --no-build-isolation --no-index -e .
```

本机实际验证：Python 3.12.14、pip 25.0.1、setuptools 84.0.0。只安装本地 OrganRelation3D 包，不下载依赖、不变更 torch 或任何 GPU 环境。构建工具缺失时不要去掉离线选项来隐式下载大依赖；由用户在独立开发环境中准备。

## 完整 NIfTI→Tensor 合成测试

fullscan 组同时需要 torch、NumPy、SciPy、NiBabel；全套历史数据测试还需要 Pillow/psutil。本轮不下载或安装包，使用两套既有同版本 CPython 3.12 Windows 环境：仅在测试进程将已有重采样 site-packages 加入 PYTHONPATH，再由 CPU PyTorch 解释器执行。实际运行的是 NumPy→Torch 的直接桥接，不是 mock。此做法只用于本机已有环境的离线验证，不是 AMD 安装方案。

```powershell
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path (Get-Location) '.venv-resampling\Lib\site-packages'
    .\.venv-backbone-cpu\Scripts\python.exe -B scripts/run_tests.py --suite fullscan
    # 具备所有既有依赖时，也可 --suite all，一次执行全部 216 项。
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
```

长期使用应在独立 CPU 环境安装 CPU torch 与影像依赖，不依赖上述本机路径。只需要 fullscan 时可使用 requirements-fullscan.txt；完整历史数据测试依赖见 requirements-resampling-cpu.txt。大 wheel 由用户手动下载，禁止因此改动 GPU 环境。
