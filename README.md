# organ-relation

完整 CT 扫描范围的动态器官关系分割项目，暂名可修改。当前有方法规范、三平台开发规范、训练 CT 的 CPU 元数据统计和完整范围候选网格估算；没有实际重采样、网络、训练、推理或 GPU 测试实现。

- [方法定义与待确认配置](METHOD_SPEC.md)
- [开发和设备约束](AGENTS.md)
- [元数据统计语义与局限](docs/metadata_audit.md)
- [完整范围候选预处理研究（未冻结）](docs/preprocessing_candidates.md)
- [Windows CPU](environments/windows-cpu.md)、[AMD ROCm](environments/amd-rocm.md)、[NVIDIA CUDA](environments/nvidia-cuda.md)

## 元数据统计无需安装依赖

Python >=3.10，元数据统计程序只使用标准库。直接从项目根目录运行，无需 pip install，不导入 torch，不改 GPU 环境。真实重采样的隔离 CPU 环境另见下文。

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

`src/organ_relation/` 包含元数据、候选估算、CPU 保真度和可配置 3D U-Net 骨干模块；`scripts/` 是入口；`tests/` 包含数据、几何和骨干测试；`configs/` 是探索配置；`docs/` 和 `environments/` 是说明。关系模块、完整 Segmentor、训练与真实数据推理尚未实现。

## 后续门槛

先审阅训练 CT 统计，再提出保留完整范围的候选重采样配置。保真度没有预设阈值；先报告实际几何损失和图像质量风险。完整网络建立后再提供 AMD 显存测试脚本，由用户同步到工作站执行。本轮不冻结输入尺寸、spacing 或评估协议。

## 候选网格估算

已有统计 JSON 后，无需访问原始 CT，只计算全部病例的三组候选尺寸、体素量、16/32 倍数补齐敏感性：

```text
python -B scripts/estimate_preprocessing.py --metadata reports/ct_stats_20260922_final/metadata.json --output-dir reports/candidates_run01
```

候选参数见 configs/preprocessing_candidates.json，明确标为 exploratory_not_frozen。输出 report.md、cases.csv（200×3 行）和 estimates.json；不能将 estimated grid 或假设 padding 当作已经确认的预处理实现或显存验收。

## 少量真实 CT 保真度

10 例 × A/B/C 的隔离 CPU 协议、强度缩放、物理网格、往返指标和运行命令见 [fidelity_protocol.md](docs/fidelity_protocol.md)。配置 [fidelity_pilot.json](configs/fidelity_pilot.json)，入口 scripts/test_resampling_fidelity.py；无需且不安装 torch。未安装可选 CPU 包时，保真度测试跳过，不能称已完成全部测试。

可变完整输入与解码尺寸对齐的工程研究见 [variable_shape_unet_review.md](docs/variable_shape_unet_review.md)。这些研究均不冻结正式 spacing/尺寸，不新增节点 mask，也不能代替完整网络 AMD 显存测试。

本轮 10 例真实 CPU 运行、53 项测试及限制记录见 [fidelity_cpu_validation.md](docs/fidelity_cpu_validation.md)；病例级报告与复核图仅留在本地 reports/fidelity_20260923_run01/。

## 可变尺寸 3D U-Net 骨干

实现与接口见 [backbone_stage1.md](docs/backbone_stage1.md)，实测记录见 [backbone_cpu_validation.md](docs/backbone_cpu_validation.md)。Encoder 返回 F 与高到低分辨率 skips；Decoder 接受 F 或后续同形 Fprime，按实际 skip 尺寸解码，输出同范围 16 类 logits。没有显式输入补齐、裁剪或节点 mask。

configs/backbone_micro.json 只用于微型合成测试，不是正式结构或输入配置。骨干使用独立 `.venv-backbone-cpu` 与 CPU PyTorch；大文件下载由用户执行，不覆盖重采样、AMD 或 CUDA 环境。

```powershell
.\.venv-backbone-cpu\Scripts\python.exe -B -m unittest discover -s tests -p test_backbone.py -v
.\.venv-backbone-cpu\Scripts\python.exe -B scripts/check_backbone.py --device cpu --output reports/backbone_cpu_new.json
```

scripts/audit_backbone_shapes.py 仅从候选统计 JSON 推导各级 shape；scripts/check_backbone.py 只运行一次合成前后向，不实现训练或方法联合损失。两者均拒绝覆盖已有报告。GPU 显存接口已预留，但本轮未实测 GPU 或完整 CT 网络。
