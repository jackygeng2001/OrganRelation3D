# OrganRelation3D

完整 CT 扫描范围的三维腹部多器官分割科研项目，面向 AMOS22 CT。GitHub 仓库名称为 **OrganRelation3D**，Python package 为 **organ_relation**。论文方法名称尚未确定，仓库名称不代表论文方法命名。

已完成：200 例训练 CT 头信息统计、三候选网格估算、10 例真实 CT 的 CPU 重采样保真度探索，以及完整 Segmentor / 独立 JointLoss 的 CPU 公式、形状和梯度验证。现已提供完整扫描 NIfTI→Tensor Dataset 与单病例完整模型验证入口（可执行一次 optimizer step）。**尚无正式训练/验证/推理循环、scheduler 或 checkpoint；没有 GPU 实测或正式训练结果。**

方法唯一工程依据：[METHOD_SPEC.md](METHOD_SPEC.md)。开发与三台设备规范：[AGENTS.md](AGENTS.md)。完整扫描、15 个前景节点、单次前向和全局有向关系保持不变；真实标签只进入监督损失和评估。

## 代码导航

```text
configs/                     探索配置与 CPU 微型配置
src/organ_relation/
  models/                    骨干、关系模块、Segmentor、模型配置与诊断工具
  data/                      头信息统计、候选估算、CPU 保真度及完整扫描 Tensor 契约
  losses.py                  独立 JointLoss，不依赖模型类或数据工具
  provenance.py              仓库 Git 状态与源码/配置文件哈希
  __init__.py                包版本，不自动导入 torch 或影像依赖
scripts/                     命令行工具与严格的分组测试入口
tests/                       合成数据、独立公式参考、数值梯度与集成测试
docs/                        数据协议、结构说明及已有历史验证记录
environments/                CPU / ROCm / CUDA 分开的环境说明
reports/                     本地生成结果（Git 忽略，不随 clone 提供）
```

模型配置类与模型放在同一目录，但只依赖标准库；不导入 torch 也能做尺寸推导。包的三个 __init__.py 不做批量导出，使用明确的子模块 import。以下是当前 API；早期平铺模块的 import 已迁移，没有另建兼容层。

| 主要模块 | 职责 |
|---|---|
| models/backbone.py、backbone_config.py | Encoder/Decoder、可变尺寸及逐级 shape；UNetBackbone3D 仅为骨干直接连接 |
| models/coarse_head.py、space_to_node.py | 16 类粗概率和 15 个软器官节点 |
| models/dynamic_relation.py | 有向关系、消息求和、显式 FormulaGRU 和共享参数的 K 轮更新 |
| models/node_to_space.py、residual_fusion.py | 独立 sigmoid 匹配、节点内容回写、纯残差融合 |
| models/segmentor.py、segmentor_config.py | 严格连接已验收模块；输入只有 image |
| models/diagnostics.py | 参数/梯度清单和显式选择设备的显存接口 |
| losses.py | coarse/final 概率 CE 与每病例前景 Dice；仅监督接口接收 label |
| data/nifti_header.py、ct_stats.py | 有界头读取、配对几何与训练 CT 统计 |
| data/candidate_estimates.py | 已有元数据上的 A/B/C 网格和假设 padding 估算 |
| data/full_scan.py | 完整扫描预处理与逐例 Dataset；返回不带 batch 轴的单样本 |
| data/fidelity.py | 物理网格重采样、往返几何和 CT 灰度指标 |
| data/fidelity_pilot.py、fidelity_visuals.py | 小样本 CPU 探索运行与复核图；不是正式 Dataset 或模型评估器 |

今后有实际实现时再增加训练、推理、评估等模块；当前不创建空目录或虚假的入口。

## 环境和安装

核心源码要求 Python >=3.10；锁定的重采样环境使用 Python >=3.11，当前两套 CPU 环境实测 Python 3.12.14。按设备阅读环境说明，不共用 PyTorch 安装方案：

| 设备/用途 | 说明 | 当前状态 |
|---|---|---|
| Windows 笔记本 CPU | [windows-cpu.md](environments/windows-cpu.md) | PyTorch 2.8.0+cpu 张量测试；独立重采样环境的数据测试 |
| AMD RX 7900 XTX 24GB | [amd-rocm.md](environments/amd-rocm.md) | 用户指定 ROCm 主环境，尚未实测 |
| NVIDIA RTX 5060 8GB | [nvidia-cuda.md](environments/nvidia-cuda.md) | 用户指定辅助 CUDA 环境，尚未实测 |

PyTorch 与影像库由 environments/ 分别管理，pyproject.toml 故意不自动选择 CPU/CUDA/ROCm wheel。已有对应依赖及 setuptools>=68 时，在仓库根目录执行以下命令即可 editable install；命令离线且不安装/升级依赖：

```text
python -m pip install --no-deps --no-build-isolation --no-index -e .
python -c "import organ_relation; print(organ_relation.__version__)"
```

将 python 替换为目标环境解释器。若缺少构建工具，应在独立开发环境先补齐；不要为安装本包覆盖现有 GPU 环境。较大依赖由用户手动下载。本机在 .venv-backbone-cpu 中实际验证上述安装。

仓库脚本和既有测试也支持直接从 clone 运行：入口定位自身仓库的 src/，不依赖本地绝对路径或工作目录名称。不要求本地文件夹改名；现有 venv 内含绝对路径，迁移文件夹后应重建环境，不能假定 venv 可直接搬迁。

## CPU 验证

原有 **190 项**测试分为张量 137 项和数据 53 项；fullscan 组的 26 项测试需要同时具备 PyTorch 和影像依赖。两套环境有意分开，避免为单元测试混装 GPU 或影像依赖：

```powershell
.\.venv-backbone-cpu\Scripts\python.exe -B scripts/run_tests.py --suite tensor
.\.venv-resampling\Scripts\python.exe -B scripts/run_tests.py --suite data
```

只有标准库时可运行 `python -B scripts/run_tests.py --suite metadata`（43 项）。同时具备全部依赖的独立 CPU 环境可使用 --suite all。Linux 将解释器替换为相应 venv 的 bin/python。

该入口遇到失败、导入错误、零测试或任何跳过均返回非零；新增测试文件未归入组也会报错。直接 unittest discover 仍可用于单文件开发，但缺依赖时的 skip 不能视为完整验收。测试只用合成张量和临时合成 NIfTI，不需要 AMOS 或 GPU。

| 测试文件 | 数量 |
|---|---:|
| test_backbone / test_coarse_nodes | 22 / 24 |
| test_dynamic_relation / test_node_to_space | 27 / 23 |
| test_segmentor / test_joint_loss | 20 / 21 |
| test_metadata / test_candidate_estimates / test_fidelity | 32 / 11 / 10 |

## 模型与损失接口

```python
import json
import torch
from organ_relation.models.segmentor import Segmentor
from organ_relation.models.segmentor_config import SegmentorConfig

with open("configs/segmentor_micro.json", encoding="utf-8") as stream:
    config = json.load(stream)
torch.set_num_threads(config["probe"]["cpu_threads"])
torch.manual_seed(config["probe"]["seed"])
model = Segmentor(SegmentorConfig(**config["model"]))
image = torch.randn(*config["probe"]["input_shape_bcdhw"])
output = model(image)
print(output.coarse_logits.shape, output.final_logits.shape)
# [1,16,5,5,5] 和 [1,16,17,18,19]；仅 CPU 微型示例。
```

数据流为 `image → Encoder → F,skips → CoarseHead(F) → SpaceToNode(F,P) → DynamicRelation → zK → NodeToSpace(原F,zK) → G → ResidualFusion(原F,G) → Fprime → Decoder(Fprime,原skips)`。普通前向只返回 coarse/final logits；显式 forward_with_diagnostics(image) 才返回中间量和各轮记录。诊断保留计算图，不应长期持有。

```python
from organ_relation.losses import JointLoss

# 当前 baseline：实际入口从 configs/baseline.json 读取这些值。
criterion = JointLoss(epsilon=model.config.epsilon, lambda_c=0.5, align_corners=False)
loss = criterion(output.coarse_logits, output.final_logits, label)
```

label 为同输入网格 [B,D,H,W] 的 int64 类别索引 0–15。粗 logits 先插值再 softmax，GT 不插值；CE 使用 log(S+epsilon) 且包含背景，Dice 按病例计算全部 15 个前景类，再组合分支并 batch 平均。详细公式唯一维护在 METHOD_SPEC。S=1 时 CE 可略为负值，这是冻结公式的结果。

当前完整模型/loss 支持已验证的 CPU FP32/FP64、关闭 autocast；完整扫描验证入口统一 FP32。finite/autocast 检查保留；GPU 同步开销、完整双分支内存和 GPU scatter 归约的数值/复现性留待 AMD 实测。

## 工具入口与配置

| 入口 | 用途及输入 |
|---|---|
| scripts/stat_training_ct.py | --data-root、--output-dir；标准库，只读取 NIfTI 头 |
| scripts/estimate_preprocessing.py | --metadata、--output-dir；不读取 CT 体素 |
| scripts/test_resampling_fidelity.py | --data-root、--metadata、--output-dir；CPU 小样本保真度 |
| scripts/summarize_fidelity.py | --report-dir；只汇总已有探索结果 |
| scripts/audit_backbone_shapes.py | --estimates、--output；纯尺寸推导 |
| scripts/check_backbone.py | --device cpu、--output；微型骨干一次前后向，非完整模型 benchmark |

各入口支持 --help。例如从仓库根目录运行：

```text
python -B scripts/stat_training_ct.py --data-root /path/to/amos22 --output-dir reports/ct_stats_run01
python -B scripts/estimate_preprocessing.py --metadata reports/ct_stats_run01/metadata.json --output-dir reports/candidates_run01
```

Windows 同样用 --data-root 指定实际位置。原始数据只读，运行输出放在数据目录之外的新目录。头统计无错误不代表体素内容或重采样质量已通过。探索重采样和汇总命令见 [fidelity_protocol.md](docs/fidelity_protocol.md)。

ct_stats.json 定义训练清单选择及空间容差；preprocessing_candidates.json 为未冻结的 A/B/C 与假设 padding 对照；fidelity_pilot.json 为已记录的小样本探索协议。backbone_micro.json / segmentor_micro.json 只用于 CPU 合成测试。baseline.json 明确记录 epsilon=1e-6、lambda_c=0.5、coarse align_corners=False；模型容量为 null。正式 spacing、模型容量及训练预处理仍待实验决策。

## 复现、数据保护与后续开发

state_dict 保存已注册参数；重建模型还须保存完整 SegmentorConfig，包括 K、epsilon、步幅、归一化和插值等无参数设置。loss 的 epsilon/lambda_c/align_corners 不在 state_dict 中，必须另存；模型与 loss 使用一致的 epsilon。应保存 state_dict 与配置，不依赖整个 Python 模型对象的 pickle 跨源码重构恢复。

正式实验另需记录 Git commit/dirty、依赖和 GPU 后端、随机种子及恢复所需 RNG 状态、数据划分/哈希、预处理及空间逆变换、评估协议；未来可恢复训练还需 optimizer 等状态。当前没有实现 checkpoint 或实验管理系统。

数据、影像、权重、缓存、venv、预测、日志和 reports/ 默认忽略。不要用 git add -f 上传这些内容；也不要把患者 PNG/CSV 移到源码或 docs 绕过输出目录。已有 docs/*validation.md 是当时的历史记录，日期、测试数量及“未实现”描述只针对当时阶段；当前状态以本 README 和 METHOD_SPEC 为准。后续不为每个小阶段另建报告文件。

完整范围的数据到 Tensor 契约和单病例 Segmentor+JointLoss 验证入口现已实现并完成 CPU 合成测试。下一步确认模型容量后，由用户在 AMD 工作站手动运行并返回结果。正式预处理、评估及实验配置须单独确认；不能把探索工具直接当作生产 Dataset，也不能根据微型骨干内存断言 AMD 24GB 可训练。NVIDIA 仅承担后续兼容验证。

## 完整扫描验证入口

数据模块复用已验证的底层物理网格/重采样函数，不运行 fidelity 探索脚本。单样本 image 为 float32 [1,D,H,W]，label 为 int64 [D,H,W]，D/H/W 对应 S/A/R。DataLoader(batch_size=1) 默认得到模型所需 [1,1,D,H,W] / [1,D,H,W]；单病例 probe 则在搬运阶段自行 unsqueeze(0)，Dataset 不负责 batching。完整边界与中心保持，保存原空间双向映射，无 GT ROI 或全局 padding。当前明确使用缩放后的 HU，无额外截断/归一化。A/B/C 只是待比较候选，正式训练预处理尚未冻结。

在现有 AMD PyTorch 环境准备影像依赖的说明和完整命令见 [amd-rocm.md](environments/amd-rocm.md)。以下从仓库根目录执行；替换实际数据路径：

```bash
python -B scripts/validate_full_scan.py --data-root /path/to/amos22 --case amos_0043 --candidate B --backend rocm --device cuda:0 --through preprocess --output reports/amos_0043_B_preprocess.json
```

可用 --through 值：preprocess、transfer、forward、loss、backward、step；运行必要前置阶段后停止。forward 及以后必须传 --model-config；没有容量默认值。采用现有微型配置时必须显式 --allow-micro-model，结果只能解释为管线/算子兼容性：

```bash
python -B scripts/validate_full_scan.py --data-root /path/to/amos22 --case amos_0043 --candidate B --backend rocm --device cuda:0 --model-config configs/segmentor_micro.json --allow-micro-model --through step --optimizer adamw --lr 0.0001 --output reports/amos_0043_B_micro_step.json
```

这里 AdamW/lr 仅为一次工程探针的示例，不冻结正式优化器。实际选项（包括 weight_decay、betas、optimizer epsilon、foreach/fused）记录到 JSON；SGD 的状态显存不能代替 AdamW。选定正式容量后换为对应模型配置并去掉 --allow-micro-model。更换 --candidate A/B/C 即切换 spacing，每次用新输出文件，建议每病例/候选独立进程；不构成 epoch 循环。

每个报告包含共同的病例/config/shape/dtype/Git/环境上下文及分阶段时间、allocator 显存峰值、OOM/失败位置和梯度检查。可用时记录 mem_get_info 的整卡瞬时快照；不可用为 null，不把它称为整卡峰值。时间为首轮同步 wall time，包含 finite 检查，未做预热；一次 optimizer step 不代表稳态训练峰值。

正式容量尚未选定时可先运行预处理/搬运或显式微型兼容探针，不能宣称正式模型已通过 24GB 验证。本轮 Windows 只测试合成 NIfTI；--backend cpu 必须同时 --device cpu --cpu-synthetic，并在头读取阶段限制原/目标体素数≤262144，超出直接拒绝，不缩小扫描。
