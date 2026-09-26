# OrganRelation3D

完整 CT 扫描范围的三维腹部多器官分割科研项目，面向 AMOS22 CT。GitHub 仓库名称为 **OrganRelation3D**，Python package 为 **organ_relation**。论文方法名称尚未确定，仓库名称不代表论文方法命名。

已完成：200 例训练 CT 头信息统计、三候选网格估算、10 例真实 CT 的 CPU 重采样保真度探索，以及完整 Segmentor / 独立 JointLoss 的 CPU 公式、形状和梯度验证。现已提供完整扫描 Dataset、资源探针及可恢复训练/内部验证入口。用户已完成 AMD FP32 capacity backward 基线和真实 10-step smoke；**新增 Console/TensorBoard 观测层仅完成 CPU 验收，尚未开展正式实验。**

方法唯一工程依据：[METHOD_SPEC.md](METHOD_SPEC.md)。开发与三台设备规范：[AGENTS.md](AGENTS.md)。完整扫描、15 个前景节点、单次前向和全局有向关系保持不变；真实标签只进入监督损失和评估。

## 代码导航

```text
configs/                     探索配置与 CPU 微型配置
src/organ_relation/
  models/                    骨干、关系模块、Segmentor、模型配置与诊断工具
  data/                      头信息统计、候选估算、CPU 保真度及完整扫描 Tensor 契约
  training/                  统一训练循环、完整 checkpoint/resume、标量日志及 ETA
  evaluation/progress.py     独立的病例级恢复 ledger
  metrics.py                 内部逐病例/逐器官 hard Dice
  losses.py                  独立 JointLoss，不依赖模型类或数据工具
  provenance.py              仓库 Git 状态与源码/配置文件哈希
  __init__.py                包版本，不自动导入 torch 或影像依赖
scripts/                     命令行工具与严格的分组测试入口
tests/                       合成数据、独立公式参考、数值梯度与集成测试
docs/                        数据协议、结构说明及已有历史验证记录
environments/                CPU / ROCm / CUDA 分开的环境说明
reports/                     本地生成结果（Git 忽略，不随 clone 提供）
```

模型配置类与模型放在同一目录，但只依赖标准库；不导入 torch 也能做尺寸推导。各包 __init__.py 不做批量导出，使用明确的子模块 import。以下是当前 API；早期平铺模块的 import 已迁移，没有另建兼容层。

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

完整官方 inference/export 后续实现；当前不创建虚假的提交入口。

## 环境和安装

核心源码要求 Python >=3.10；锁定的重采样环境使用 Python >=3.11，当前两套 CPU 环境实测 Python 3.12.14。按设备阅读环境说明，不共用 PyTorch 安装方案：

| 设备/用途 | 说明 | 当前状态 |
|---|---|---|
| Windows 笔记本 CPU | [windows-cpu.md](environments/windows-cpu.md) | PyTorch 2.8.0+cpu 张量测试；独立重采样环境的数据测试 |
| AMD RX 7900 XTX 24GB | [amd-rocm.md](environments/amd-rocm.md) | 已完成资源探针；新增训练系统待验收 |
| NVIDIA RTX 5060 8GB | [nvidia-cuda.md](environments/nvidia-cuda.md) | 用户指定辅助 CUDA 环境，尚未实测 |

PyTorch 与影像库由 environments/ 分别管理，pyproject.toml 故意不自动选择 CPU/CUDA/ROCm wheel。已有对应依赖及 setuptools>=68 时，在仓库根目录执行以下命令即可 editable install；命令离线且不安装/升级依赖：

```text
python -m pip install --no-deps --no-build-isolation --no-index -e .
python -c "import organ_relation; print(organ_relation.__version__)"
```

将 python 替换为目标环境解释器。若缺少构建工具，应在独立开发环境先补齐；不要为安装本包覆盖现有 GPU 环境。较大依赖由用户手动下载。本机在 .venv-backbone-cpu 中实际验证上述安装。

仓库脚本和既有测试也支持直接从 clone 运行：入口定位自身仓库的 src/，不依赖本地绝对路径或工作目录名称。不要求本地文件夹改名；现有 venv 内含绝对路径，迁移文件夹后应重建环境，不能假定 venv 可直接搬迁。

## CPU 验证

原有 **190 项**测试分为张量 137 项和数据 53 项；fullscan 与 training 组需要同时具备 PyTorch 和影像依赖。两套环境有意分开，避免为单元测试混装 GPU 或影像依赖：

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

正式实验另需记录 Git commit/dirty、依赖和 GPU 后端、随机种子及恢复所需 RNG 状态、数据划分/哈希、预处理及空间逆变换、评估协议；当前 training/state.py 已保存 optimizer/RNG/病例游标等完整恢复状态，见下方训练入口。

数据、影像、权重、缓存、venv、预测、日志和 reports/ 默认忽略。不要用 git add -f 上传这些内容；也不要把患者 PNG/CSV 移到源码或 docs 绕过输出目录。已有 docs/*validation.md 是当时的历史记录，日期、测试数量及“未实现”描述只针对当时阶段；当前状态以本 README 和 METHOD_SPEC 为准。后续不为每个小阶段另建报告文件。

完整范围的数据到 Tensor 契约和单病例 Segmentor+JointLoss 验证入口现已实现并完成 CPU 合成测试。下一步由用户在 AMD 工作站运行 3S-P1/B 的训练系统 smoke 并返回结果。正式预处理、评估及实验配置须单独确认；不能把探索工具直接当作生产 Dataset，也不能根据微型骨干内存断言 AMD 24GB 可训练。NVIDIA 仅承担后续兼容验证。

## 完整扫描验证入口

下述 validate_full_scan.py 是资源探针；统一训练入口见文末，不将单次 backward 与连续 AdamW 训练混为一谈。

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

默认 memory format 保持原 contiguous 路径。可显式传入 `--memory-format channels_last_3d`，仅转换 image / model，不改变 label、模型公式或科学配置；ROCm 要求启动进程前设置 `PYTORCH_MIOPEN_SUGGEST_NHWC=1`，否则提前报错。CPU 合成小例也支持该选项。请求布局、实际输入/卷积权重 stride 及相关 backend 环境会记录到 JSON。使用范围、已测结果和无逐算子 profiling 的 AMD 命令见 [AMD 环境说明](environments/amd-rocm.md#显式-channels-last-3d-执行选项)。

定位 forward 显存时显式增加 `--profile-forward-memory`（要求 `--through forward` 或以后）。该选项仅在正常 forward 外临时安装 module hooks 和 ATen dispatch 观察器，不调用保留中间张量的 diagnostics，不改变 autograd、dtype 或模型配置。JSON 的 `forward_memory` 保存模块/算子输入输出 shape、stride、逻辑字节数、同步时间、前后 allocated/reserved 和各作用域峰值；逻辑字节数不能把共享 storage 的 view 重复相加。嵌套区间峰值会合并回父模块及外层 forward，CPU 显存字段为 null。OOM 尽量在算子异常尚未退出模型栈时记录活动模块、算子、错误原文和 traceback；失败后不同步、不清缓存、不重试。后端卷积内部 workspace/隐式复制不单独可见，失败申请也不计入已分配峰值，不能仅凭卷积 OOM 就断言是 workspace。逐算子同步会影响时间及分配器复用，结果是诊断运行而非正常性能基准。观察器使用 PyTorch `TorchDispatchMode`，当前环境测试之外的版本仍需兼容验证。

正式容量尚未选定时可先运行预处理/搬运或显式微型兼容探针，不能宣称正式模型已通过 24GB 验证。本轮 Windows 只测试合成 NIfTI；--backend cpu 必须同时 --device cpu --cpu-synthetic，并在头读取阶段限制原/目标体素数≤262144，超出直接拒绝，不缩小扫描。

## 可恢复训练 v1

`scripts/train.py` 共用于 smoke、单/双病例 overfit、pilot 与后续正式训练。`configs/train_sanity.json` 显式采用 **3S-P1 / B / FP32 / batch=1 / workers=0 / channels_last_3d**，AdamW lr=3e-4、betas=[0.9,0.999]、eps=1e-8、weight_decay=0、foreach=false、fused=false。它只用于训练系统验收，不冻结正式 backbone、spacing 或训练参数；模型、损失和预处理公式均未改变。

已有 AMD 环境中，从仓库根目录运行（不安装/切换 GPU 环境）：

```bash
git pull --ff-only origin main
AMOS_ROOT="/absolute/path/to/amos22"

# 首次生成一次固定 160/40；已存在则拒绝覆盖，后续不重跑此命令。
python -B scripts/train.py --config configs/train_sanity.json \
  --data-root "$AMOS_ROOT" --prepare-split runs/splits/development_160_40.json

# 10-step smoke：使用固定 train 列表首例，第 5、10 步同病例 hard Dice 监测。
PYTORCH_MIOPEN_SUGGEST_NHWC=1 python -B scripts/train.py \
  --config configs/train_sanity.json --data-root "$AMOS_ROOT" \
  --split runs/splits/development_160_40.json --run-dir runs/smoke_3sp1_B_01

# 中断后使用同一配置、同一源代码和原 run 目录恢复。
PYTORCH_MIOPEN_SUGGEST_NHWC=1 python -B scripts/train.py \
  --config configs/train_sanity.json --data-root "$AMOS_ROOT" \
  --split runs/splits/development_160_40.json --run-dir runs/smoke_3sp1_B_01 \
  --resume runs/smoke_3sp1_B_01/last.ckpt
```

准备 split 只读官方 training 清单/头信息选择，不加载 CT 体素；真实入口核对 200 例及 160/40。首次 split 的 seed 默认 20260925，可在首次 `--prepare-split` 时显式覆盖，随后只读取保存的 artifact。病例文件路径保持 root-relative；manifest、split 与实际 subset 都记录并哈希。raw data 只读，输出不能位于数据目录内。自动选择的病例 ID 会显示在启动日志中。

各阶段修改配置创建新 run：`data.train_cases` 或 `--cases` 显式选择 development train 子集；`train_limit=null` 表示不限数量，2 表示固定列表前两例。`training.max_steps/max_epochs` 任一可为 null，同时给出取先达到的上限。`validation_role=train_monitor` 用于 overfit 的训练病例监测；pilot 改为 `internal_dev`，不把训练 Dice 称为 held-out 验证。最终训练设 `data.role=final_train`、`train_limit=null`，必须使用完整 200 例，从头开始；不能 resume development run 冒充最终重训。官方 validation/test 不会被本入口加载。

`checkpoint_every`、`diagnostics_every`、`validation_every` 独立配置，后两者可为 0。smoke 三者分别为 1/1/5；长训练自行配置较低频率。诊断保存各有参数模块的有限梯度、范数和参数更新量，SpaceToNode 不伪造参数统计。CPU 入口仅用于合成小例，要求 CPU config + `--cpu-synthetic` 并限制原/目标体素数，不在 Windows 尝试真实 full-scan 网络。

恢复和输出约定：

- `last.ckpt` 在完整 optimizer step + 标量日志提交后临时写入、flush/fsync、原子替换，内含校验和。保存模型、optimizer、未来 scheduler 槽位（当前 null）、step/已完成 epoch、当前 epoch 实际病例顺序/下一位置、Python/NumPy/CPU/ROCm RNG、loader/sampler generator、完整配置、Git/dirty/源码指纹、manifest/split/subset 哈希、日志提交位置及 pending validation。仅加载本项目可信 checkpoint。
- `--resume` 严格检查运行身份（含软件环境与数据头哈希/文件大小/mtime），要求原 run 的 `run.json` 和 `metrics.jsonl`。不自动兼容改过配置/源码的运行。源码指纹和 Git commit 均校验；跨机器搬运数据改变 mtime 也会拒绝，不提供静默绕过。未全量哈希 CT payload，不把头指纹称作体素内容校验。
- 日志是标量 JSONL；恢复时保留 checkpoint 提交前缀，明确丢弃后续尾部，重跑未保存步骤。GPU/进程失败不保存半个 step。首个成功 checkpoint 前中断需新 run；保留已有 run，不自动清理。`--stop-after N` 可用于人为分段验收：它限制本次进程执行步数，不修改配置中的总步数。恢复时不要修改 `max_steps`。
- 每步显示病例/shape、epoch/step、LR/loss、最近耗时、rolling20、epoch/train ETA 和 GPU allocator 峰值。少于 5 个样本显示 warming up；resume 重置计时窗口。训练 ETA 排除独立 validation 与 checkpoint 写盘时间，因此预计时刻是训练工作量估计。各阶段耗时波动不代表模型质量。
- 验证前先保存 pending 状态；`validation/step_*/` 的 CaseLedger 逐病例原子保存 JSON 结果并核对身份、checksum、文件存在性后才跳过。中断后继续未完成病例，再汇总并清除 pending；验证拥有独立 ETA，不消耗训练 RNG。ledger 基础设施支持无标签结果，完整预测 NIfTI 导出/文件校验接口留待官方 inference 实现。
- 内部 hard Dice 在重采样网格逐病例/器官计算，背景排除，两者皆空为 null；有任一前景时正常计算，GT 空而预测非空为 0 并记录 FP 体素。汇总先病例平均，另有每器官有效病例数。它不是 JointLoss soft Dice，也不是 AMOS 官方指标；不提供近似 NSD/HD95。

200 official training CT 的固定 160/40 用于 development，完整 200 用于冻结后重训。官方 validation 需 Linux 核实后用于最终 held-out 评价；hidden test 无标签，后续恢复预测到官方原空间并提交 DSC/NSD，不以排行榜反复调参。原始 affine/shape 与双向映射已保存，但降采样信息不能无损恢复；本版没有官方 export/submission。

新增 CPU 接受测试运行 `python -B scripts/run_tests.py --suite training`。覆盖连续训练与中途/epoch 边界恢复的精确 CPU 轨迹、优化器/RNG/病例顺序、损坏与不匹配拒绝、日志回滚、pending 验证、ETA、split 隔离以及临时 NIfTI→完整 Segmentor→JointLoss→AdamW→保存/恢复。无 AMD 逐位一致承诺；其验收需检查状态/顺序、数值连续性与真实运行显存。

本阶段 Windows CPU：新增训练组 23 项通过；全量 252 项通过，零跳过。仅使用小张量及临时合成 NIfTI，没有运行真实 CT 训练或 GPU。

## Console 与 TensorBoard 观测

已收到用户的 AMD 10-step smoke 通过结果；本轮仅为训练入口增加独立观测层，新增代码只在 CPU 测试。`training/console.py` 提供固定两列启动/摘要、TTY 下的 tqdm、独立 monitor/validation 进度条和简短恢复块。单/双病例用 run 级进度，多病例用 epoch 级进度；non-TTY 禁用动态条，仅打印启动与阶段摘要。`Last Loss` 是最近一次训练 loss，不冒充 epoch 均值；`Monitor Dice` 来自训练病例监测，不叫 Val Dice。正常诊断不展开模块字典；完整标量仍在 JSONL。ETA 算法与计时范围不变。

`training/tensorboard.py` 默认向 `<run-dir>/tensorboard/` 写 scalar。可用 `--no-tensorboard`、`--quiet-console` 独立关闭观测；这些 CLI 开关不进入数值配置或 checkpoint 身份。完整有效训练配置仍在 run.json。依赖见 `environments/requirements-monitoring.txt`，不包含 torch/TensorFlow，不覆盖 GPU 后端：

```bash
python -m pip install -r environments/requirements-monitoring.txt
tensorboard --logdir runs --host 127.0.0.1 --port 6006 --load_fast=false
```

浏览器打开 http://127.0.0.1:6006；使用标准事件加载器处理 restart/purge 标记。默认只监听本机；无需上传日志或建立远程开发环境。

观测依赖固定 TensorBoard 2.20.0、tqdm 4.67.1、setuptools 80.9.0；最后一项用于兼容 TensorBoard CLI 的 pkg_resources 导入。只升级/补齐观测依赖，不安装 TensorFlow 或替换 PyTorch。

scalar 包括 `Train/{Total_Loss,Coarse_CE,Coarse_DiceLoss,Final_CE,Final_DiceLoss,SoftDice_Mean}`、`Optimizer/LR`、`System/{Step_Time,GPU_Peak_Allocated_GiB,GPU_Peak_Reserved_GiB}`。诊断触发时写 `GradNorm/{Encoder,CoarseHead,Relation,NodeToSpace,Fusion,Decoder}`；仅 monitor/validation 完成时写 `Monitor/HardDice_Mean`、`Val/HardDice_Mean` 和 `MonitorDice/<organ>`、`ValDice/<organ>`。15 个器官按 METHOD_SPEC 标签顺序命名；null Dice 和 CPU 不可用显存不伪造为 0。无 volume、histogram、embedding 或图片。

恢复仍先校验 checkpoint 并回滚 JSONL。checkpoint 提交在 K 时，writer 使用 **purge_step=K，再仅重放已提交 JSONL 中 K 的记录**，然后从 K+1 继续。相比只 purge K+1，这还能清除“训练 K 已提交、验证 K event 已写但验证 checkpoint 未提交”的残留；已提交验证会重放一次，未提交验证由原 pending/ledger 机制完成。event 文件物理保留，TensorBoard 有效曲线不重复。若此前停用/故障造成历史缺点，JSONL 仍完整，不将缺失曲线当作训练失败或自动补造数据。

writer 在 checkpoint 成功后 flush；正常结束、`--stop-after` 和异常退出均执行 flush/close。写入/flush/close 失败仅输出一次明确的 observer 警告并停用 TensorBoard；已成功的 checkpoint 不失效，训练继续。观测不持有模型张量、计算图或更改 RNG；不计入训练 step throughput。不得同时启动两个进程写同一 run。

同一秒快速重启时，默认 event 文件名中的 PID/未补零计数器可能排错。observer 只在启动阶段最多等待约一秒，让新文件时间戳严格晚于旧文件，保证 purge 后读；明显时钟回退则停用 observer 并告警，不无限等待或改变训练状态。

进入单病例 overfit 的第一个 100-step 区间时使用新 run，沿用已有 split（不重新生成）。`train_single_case_overfit.json` 的总步数为 100，保存、monitor 和诊断周期均为 25；沿用 3S-P1/B、AdamW、FP32 与完整 CT，不代表已选定最终训练配置：

```bash
git pull --ff-only origin main
AMOS_ROOT="/absolute/path/to/amos22"
env -u MIOPEN_DEBUG_CONV_GEMM -u MIOPEN_LOG_LEVEL -u MIOPEN_ENABLE_LOGGING_CMD \
  PYTORCH_MIOPEN_SUGGEST_NHWC=1 MIOPEN_ENABLE_LOGGING=0 \
  python -B scripts/train.py --config configs/train_single_case_overfit.json \
    --data-root "$AMOS_ROOT" --split runs/splits/development_160_40.json \
    --run-dir runs/overfit_1case_3sp1_B_100
```

默认仍为固定 train 列表第一例；若此前 smoke 显式选了其他 train 病例，同样追加 `--cases <该病例ID>`。不要直接 resume 旧源码/10-step 配置的 smoke checkpoint；本轮未放宽严格来源与配置校验。新 overfit 中断后，同一命令追加 `--resume runs/overfit_1case_3sp1_B_100/last.ckpt`，保留相同 config/病例/环境；最后一次保存后未提交的步会重跑。只有到 100 步的数值趋势经审查后，再确定后续运行长度。

### 显式延长同一训练轨迹

审查后需要从 100 延长到 200 steps 时，继续使用原始 100-step config，显式增加 `--resume ... --extend-to 200`。禁止直接编辑 config 的 max_steps 来恢复。普通 resume 的完整 identity 检查不变；extension 也要求 model/loss/preprocessing/optimizer/lr/data/split/seed/runtime/environment 及原始 training 配置全部匹配。

```bash
env -u MIOPEN_DEBUG_CONV_GEMM -u MIOPEN_LOG_LEVEL -u MIOPEN_ENABLE_LOGGING_CMD \
  PYTORCH_MIOPEN_SUGGEST_NHWC=1 MIOPEN_ENABLE_LOGGING=0 \
  python -B scripts/train.py --config configs/train_single_case_overfit.json \
    --data-root "$AMOS_ROOT" --split runs/splits/development_160_40.json \
    --cases amos_0109 --run-dir runs/overfit_1case_3sp1_B_100 \
    --resume runs/overfit_1case_3sp1_B_100/last.ckpt --extend-to 200
```

- `--extend-to` 仅用于 resume；目标必须大于已完成 global_step、当前有效 total 和原始 max_steps。不接受降低或重复设置同一个上限；max_steps=null 的仅 epoch 运行不能用此接口扩展。
- max_epochs 保持不变，目标超过其对应的步数上限时明确拒绝，不截短目标或绕过 epoch 限制。
- `run_id` 与 `run.json` 保留，model/optimizer/RNG/sampler/order/cursor 全部从 checkpoint 恢复。metrics.jsonl 保留已提交前缀，TensorBoard 使用原目录和既有 purge/replay 规则；ETA 在新 total 下重新 warm up。
- checkpoint 的 `horizon` 保存 `original_total_steps`、有效 `total_steps` 和追加的 `extensions` 历史，包括扩展发生的 global_step、前后 total、不变的 max_epochs、UTC 时间、父 checkpoint SHA-256、升级前后 provenance。`origin_identity` 保留最初运行身份；validation ledger 始终使用该原始身份，支持 pending validation 恢复。
- 扩展先在已有完整 optimizer 边界原子保存，再开始下一步。保存成功后的中断恢复使用原 config 和普通 `--resume`，**去掉 `--extend-to 200`**；checkpoint 已记住 200。只有再次提高上限时才再次传入 `--extend-to`。
- 针对已有 `88ea4fc9fd8bc2aa3ede2c3fe87b343c386e6941` 的旧 checkpoint，显式 extension 支持一次受控源码升级：校验该版本完整源码哈希清单，仅允许 `scripts/train.py`、`training/engine.py`、`training/state.py` 及新增 `training/extension.py` 的变化；模型、loss、data 及其他源码必须逐文件一致，目标 checkout 必须干净。不是通用忽略源码开关。原 provenance 保存在 origin/history，新的执行 provenance 成为后续普通 resume 的严格匹配目标；其他旧版本不自动迁移。

该接口不启动新 run、不重新初始化训练轨迹，不引入 optimizer reset、scheduler 或新的科学配置。CPU 验收包含真实 100→200 的逐值轨迹对照及 TensorBoard step 去重；AMD 非确定性算子不要求逐位重现。

本轮新增 21 项观测测试；Windows CPU 全量 273 项通过，零跳过。包括真实 TensorBoard event 的 scalar/purge、同一步 pending validation、写入故障隔离、TTY/non-TTY、启用/关闭的模型/optimizer/RNG/轨迹严格对照、启动中断清理和 TensorBoard CLI 导入。未运行 GPU。


## Backbone-only overfit diagnostic

`configs/train_backbone_only_overfit.json` 是隔离的诊断对照，不是论文方法或 Segmentor 的替代。`mode=backbone_only_final` 使用现有 Encoder3D → Decoder3D，只有 final logits；缺省 `organ_relation_joint` 继续使用完整 Segmentor + JointLoss。两者共用 `scripts/train.py`、AdamW、恢复/延长、ETA、console、TensorBoard 和病例 ledger。

公平初始化：先按相同 seed 和完整 SegmentorConfig 在 CPU 构建 reference Segmentor，再仅保留其 encoder/decoder；额外模块不保留、不前向、不优化。这样 encoder/decoder state 及构建后的 PyTorch RNG 与完整模型逐值一致。`run.json` 保存 reference 全配置和初始化策略；其中 relation 配置仅用于复现初始化序列，不表示诊断模型包含关系模块。

`SegmentationLoss(epsilon=1e-6)` 与 JointLoss 共用同一个分支算子：16 类概率 CE `-log(S_true+epsilon)` 加 15 前景类 soft Dice loss，逐病例再 batch 平均。诊断不计算 coarse，不乘辅助权重；JSONL 明确记录 mode，TensorBoard 不写 Coarse scalar。所有已有完整模型公式与预处理保持不变。

独立新 run 固定 amos_0109 / B / 3S-P1 / FP32 / batch=1 / channels_last_3d / seed=20260925，200 steps，每 25 steps 保存、诊断和 train_monitor。AdamW lr=3e-4、betas=[0.9,0.999]、eps=1e-8、weight_decay=0，无 scheduler。沿用此前同一 split 和数据路径：

```bash
git pull --ff-only origin main
AMOS_ROOT="/absolute/path/to/amos22"
env -u MIOPEN_DEBUG_CONV_GEMM -u MIOPEN_LOG_LEVEL -u MIOPEN_ENABLE_LOGGING_CMD \
  PYTORCH_MIOPEN_SUGGEST_NHWC=1 MIOPEN_ENABLE_LOGGING=0 \
  python -B scripts/train.py --config configs/train_backbone_only_overfit.json \
    --data-root "$AMOS_ROOT" --split runs/splits/development_160_40.json \
    --run-dir runs/backbone_only_amos0109_3sp1_B_200
```

中断后同一命令加 `--resume runs/backbone_only_amos0109_3sp1_B_200/last.ckpt`；需要经审查延长时沿用 `--extend-to`。这是新对照，不能从完整模型 checkpoint 恢复。已有完整模型 run 的严格源码校验不放宽；继续旧 run 须使用其对应源码版本。

每次 monitor 的病例 ledger 保存原有 15 类 hard Dice/计数，并新增 `diagnostic`；JSONL 的 `diagnostic_cases` 按 case ID 保存相同标量：`final_soft_dice`（15 类均值）、`predicted_foreground_voxels`、`foreground_true_positive_voxels`（预测类别与 GT 完全相同且 GT>0）、`gt_foreground_voxels`、GT 前景位置上的 `gt_foreground_true_class_mean_probability` 与 `gt_foreground_background_mean_probability`。GT 无前景时两种概率均值记 null。TP 不把预测成另一前景器官算作正确。TensorBoard 另外记录 `Monitor/FinalSoftDice_Mean`（病例等权平均），其他附加诊断查 JSONL/ledger；不长期保留概率体积。

本对照只回答普通骨干在相同条件下能否拟合单病例；不据此改动关系方法、损失、强度方案或优化器，也不自动开展第二个对照。

用户指定的首个 CE 单变量对照使用 `configs/train_backbone_only_overfit_balanced_ce.json`；它与上述配置仅相差新增 `ce_reduction_mode=foreground_background_balanced`。运行时替换 `--config`，并使用新的 `--run-dir runs/backbone_only_amos0109_3sp1_B_balanced_ce_200`，不要传旧 checkpoint 的 `--resume`。其余病例、初始化、200 steps、每 25 步监测/保存/诊断、预处理与 AdamW 全部相同。缺省仍为 `voxel_mean`；缺省 JointLoss 保持原公式；完整模型独立 balanced 对照见下文。

balanced CE 在每个病例内按背景/全部前景分别平均，再各取 0.5；不是 15 器官均权 CE。无前景或无背景时明确报错，不暗设空组策略。`run.json`/checkpoint 身份与每条训练/monitor 日志记录 `ce_reduction_mode`，改变模式不能 resume 或通过 `--extend-to` 绕过校验。新 balanced run 自身中断后仍可正常恢复。

每 25 步 monitor 的 `diagnostic_cases[case_id]`（及病例 ledger）额外记录 `CE_bg_mean`、`CE_fg_mean`、`balanced_ce`；与原 hard/soft Dice、前景计数、TP、概率诊断一起保留。训练行的 `final.ce` 是当前模式实际使用的 CE；voxel-mean monitor 中的 `balanced_ce` 只是对照统计，不参与该模式训练。空组观测值记 null，console 不展开这些字段。


完整 OrganRelation3D 的 balanced-CE 单变量对照使用 `configs/train_single_case_overfit_balanced_ce.json`，运行命令沿用 `scripts/train.py`，显式传 `--cases amos_0109` 和新目录 `--run-dir runs/full_amos0109_3sp1_B_balanced_ce_200`，首次不传 `--resume`。新配置与 `train_single_case_overfit.json` 只差 CE mode 和 max_steps 从 100 到 200，保留病例选择接口、B/3S-P1、相同 seed、AdamW、FP32 和每 25 步保存/monitor/诊断。

JointLoss 的 coarse/final 都使用选定 CE reduction，各自 Dice 不变；coarse 仍先插值 logits 到 GT 网格再 softmax。总损失为 final + 0.5×coarse。每次 balanced joint monitor 在 JSONL 的 `ce_diagnostic_cases[case_id].coarse/final` 中记录 `CE_bg_mean / CE_fg_mean / balanced_ce`，病例 ledger 对应字段为 `ce_branches`；双分支顺序计算统计，不同时持有两份完整概率体积。普通训练行仍记录各分支实际 CE/Dice/segmentation，console 布局不变。checkpoint 的 CE mode、配置及来源严格校验保留，禁止用旧 voxel-mean checkpoint 开始新对照。
