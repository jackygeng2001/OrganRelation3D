# organ-relation

完整 CT 扫描范围的动态器官关系分割项目，暂名可修改。当前仅有方法规范、三平台开发规范及训练 CT 的 CPU 元数据统计；没有网络、训练、推理或 GPU 测试实现。

- [方法定义与待确认配置](METHOD_SPEC.md)
- [开发和设备约束](AGENTS.md)
- [元数据统计语义与局限](docs/metadata_audit.md)
- [Windows CPU](environments/windows-cpu.md)、[AMD ROCm](environments/amd-rocm.md)、[NVIDIA CUDA](environments/nvidia-cuda.md)

## 无需安装依赖

Python >=3.10，当前程序只使用标准库。直接从项目根目录运行，无需 pip install，不导入 torch，不改 GPU 环境。

```powershell
python -B -m unittest discover -s tests -v
python -B scripts/stat_training_ct.py --data-root "..\..\amos22" --output-dir "reports\ct_stats_run01"
```

数据路径由命令指定；上面的相对路径对应项目位于“项目/代码/organ-relation”、数据位于“项目/amos22”的布局。重命名项目无需修改源码。Linux 手动运行示例（替换数据挂载路径）：

```bash
python3 -B -m unittest discover -s tests -v
python3 -B scripts/stat_training_ct.py --data-root /path/to/amos22 --output-dir reports/ct_stats_run01
```

这不是 GPU 测试命令。若默认 python 不可用，可将命令开头替换为已存在 Python 解释器的绝对路径，不需要安装任何包。

默认配置 [ct_stats.json](configs/ct_stats.json) 只选择 `dataset.json` 的 training 中编号 1–499 的病例；不能使用 JSON 的单一 modality 字段将整个数据集视为 CT。测试集编号 500 的边界问题不在本脚本处理范围。

## 输出

每次指定一个新的输出目录，禁止放在原始数据目录内，禁止覆盖已有目录：

- `report.md`：汇总、风险与局限。
- `cases.csv`：每例尺寸、spacing、覆盖范围和配对几何结果，UTF-8 BOM 便于 Windows 审查。
- `anomalies.csv`：错误/警告逐条清单；无异常时只保留表头。
- `metadata.json`：结构化完整头摘要、单位换算、哈希、清单、配置、代码/环境/Git 来源。

退出码 0 表示无错误（仍可能有需审阅警告），2 表示异常或无效调用。不能仅凭退出码认定数据内容、重采样或模型测试已通过。所有生成报告默认被 Git 忽略，不会自动上传病例级信息。

## 项目布局

`src/organ_relation/` 当前只有元数据审查模块；`scripts/` 是免安装入口；`tests/` 是合成头和端到端统计测试；`configs/` 是统计配置；`docs/` 和 `environments/` 是方法外的说明。网络、训练、推理模块暂不创建，后续经明确任务逐阶段增加。

## 后续门槛

先审阅训练 CT 统计，再提出保留完整范围的候选重采样配置。保真度没有预设阈值；先报告实际几何损失和图像质量风险。完整网络建立后再提供 AMD 显存测试脚本，由用户同步到工作站执行。本轮不冻结输入尺寸、spacing 或评估协议。
