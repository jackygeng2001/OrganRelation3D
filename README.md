# organ-relation

完整 CT 扫描范围的动态器官关系分割项目，暂名可修改。已实现 CPU 元数据统计、候选网格估算、少量真实重采样保真度测试、可变尺寸 U-Net 骨干、CoarseHead、SpaceToNode 及动态有向关系推理与显式 GRU。尚未实现空间回写、残差融合、联合损失、完整 Segmentor 或正式训练。

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

`src/organ_relation/` 包含数据工具、骨干、粗头、节点构建和动态关系推理模块；`scripts/` 是入口；`tests/` 包含数据、几何、网络模块及公式/梯度测试；`configs/` 是探索配置；`docs/` 和 `environments/` 是说明。空间回写、残差融合、完整 Segmentor、训练与真实数据推理尚未实现。

## 后续门槛

数据统计、三候选估算、10 例保真度与骨干已通过阶段验收。后续按冻结公式逐模块完成 CPU 数值与梯度验证；完整网络建立后再提供 AMD 显存测试脚本，由用户同步到工作站执行。正式输入尺寸、spacing、保真度阈值及评估协议仍未冻结。

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

## 粗预测与器官节点

`CoarseHead(in_channels, bias=...)` 返回 `CoarsePrediction(logits, probabilities)`：两个张量均为 `[B,16,Df,Hf,Wf]`，概率由 16 类 softmax 得到。`SpaceToNode(epsilon=...)` 接收 F 与 P，返回具名字段：

| 字段 | 形状 | 含义 |
|---|---|---|
| z0 | `[B,15,C]` | `sum(P_i F)/(M_i+epsilon)`，只加权一次 |
| mass | `[B,15,1]` | `sum(P_i)` |
| centroid | `[B,15,3]` | `sum(P_i u)/(M_i+epsilon)`，坐标轴 D、H、W |
| size | `[B,15,1]` | `M_i/N` |
| confidence | `[B,15,1]` | `sum(P_i^2)/(M_i+epsilon)` |

节点索引 0–14 对应前景类别 1–15；背景只参与 softmax。所有输出保留梯度，属性未来保持初值不表示 detach。接口不接受标签，不屏蔽零/小质量节点，不裁剪或补齐网格。语义使用 `bmm`，质心使用分轴求和，避免 `[B,15,C,N]` 张量和完整坐标网格。

```python
from organ_relation.coarse_head import CoarseHead
from organ_relation.space_to_node import SpaceToNode

# F 来自已验收 Encoder；以下数值仅示例，不是正式配置。
head = CoarseHead(F.shape[1], bias=True).to(device=F.device, dtype=F.dtype)
node_builder = SpaceToNode(epsilon=1e-6)
coarse = head(F)
nodes = node_builder(F, coarse.probabilities)
```

SpaceToNode 当前要求 F/P 同设备、同 dtype 的 FP32/FP64，且关闭 autocast；不会隐式转换或修正 P。调用方保证 P 是合法概率。FP16/BF16 与 AMP 归约需后续专门验证。epsilon 没有默认值，重建模块时须从配置显式传入，并随实验记录；参数为空的 state_dict 不携带这个配置值。CoarseHead 的 bias 同样须显式指定。

CPU 复现命令（`src` 由测试入口加入路径）：

```powershell
.\.venv-backbone-cpu\Scripts\python.exe -B -m unittest discover -s tests -p test_coarse_nodes.py -v
.\.venv-backbone-cpu\Scripts\python.exe -B -m unittest discover -s tests -p test_backbone.py -v
.\.venv-resampling\Scripts\python.exe -B -m unittest discover -s tests -p test_metadata.py -q
.\.venv-resampling\Scripts\python.exe -B -m unittest discover -s tests -p test_candidate_estimates.py -q
.\.venv-resampling\Scripts\python.exe -B -m unittest discover -s tests -p test_fidelity.py -q
```

两套既有 CPU 环境分别执行张量与数据测试；不需要改动 GPU 环境。模块测试中的标量探针只用于检查自动微分，不是方法的联合训练损失。

## 动态有向关系与显式 GRU

`DynamicRelation(channels=C, relation_channels=Cr, rounds=K)` 接收 `z0:[B,15,C]`、`centroid:[B,15,3]`、`size/confidence:[B,15,1]`，默认返回 `zK:[B,15,C]`。这些构造参数必须显式配置，测试不冻结正式 C、Cr、K。

```python
from organ_relation.dynamic_relation import DynamicRelation

# C、Cr、K 从调用方配置读取；nodes 来自 SpaceToNode。
relation = DynamicRelation(C, relation_channels=Cr, rounds=K).to(
    device=nodes.z0.device, dtype=nodes.z0.dtype)
zK = relation(nodes.z0, nodes.centroid, nodes.size, nodes.confidence)
```

关系描述严格按 `[z_i;z_j;c_j-c_i;s_i;s_j;q_i;q_j]` 排列。`alpha[b,i,j]` 为 i→j，逐边 sigmoid 后将对角线置零；没有入边归一化、均值、对称化或器官屏蔽。消息通过 `alpha.transpose(1,2) @ W_m(z)` 聚合，W_m 无偏置。GRU 的六个矩阵、三个偏置显式实现：`U_z(rho*z)` 在 reset 后做线性映射，`u` 是候选写入比例。每轮先完成全部边和消息，再同步产生新 z；下一轮重新计算边，固定属性不修改、不 detach。

`return_diagnostics=True` 返回 `RelationResult(zK, rounds)`；每个 `RelationRound(alpha, messages, z)` 的形状依次为 `[B,15,15]`、`[B,15,C]`、`[B,15,C]`。索引 t 记录从 z^(t) 到 z^(t+1) 的过程。诊断张量保留计算图，调用方应只读使用，长期保留会延长计算图生命周期；默认不额外保存诊断历史。

参数总数为 `Cr*(2C+9)+1+7C²+3C`，与 K 无关。K 轮、全部节点和边复用同一套参数。重建时应从记录的配置恢复 C、Cr、K；state_dict 不包含 K。参数按标准 nn.Linear 初始化，正式实验仍须记录初始化约定及随机种子。

当前仅验证 Windows CPU 的 FP32/FP64、关闭 autocast；非法 shape、混合设备/dtype 和 NaN/Inf 输入会被拒绝。有限性检查在 GPU 上可能带来同步开销，需后续实测。未运行真实 CT、GPU 或训练。

```powershell
.\.venv-backbone-cpu\Scripts\python.exe -B -m unittest discover -s tests -p test_dynamic_relation.py -v
```

该测试覆盖独立循环参考、边方向、GRU reset 顺序、同步更新、每轮重算边权、参数共享及输入/参数梯度；全部既有回归命令见上文。
