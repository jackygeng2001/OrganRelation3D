# 完整扫描范围动态器官关系分割规范

状态：算法公式与全局上下文已确认；实验配置待确认。2026-09-22 建立。

## 依据

主文档为项目原始资料《腹部多器官分割_方法章节.docx》第 3.1–3.5 节、式 (1)–(23)；《细化方法.docx》补充 GRU 展开公式。此处是工程转录，原文件不随项目重命名而改变。原 AGNN 的二维帧节点、5 轮更新、BCE+L1、空间两两注意力不是本方法定义。

## 已确认的上下文和类别

正式输入是完整 CT 扫描覆盖范围，一次前向构建覆盖该扫描的 15 个节点；训练与推理均使用完整范围。允许保留扫描范围的方向统一、重采样、强度归一化和必要尺寸处理，必须记录变换及逆变换。禁止局部 patch 独立建图/滑窗替代。没有出现的器官仍保留节点，不以真实标签判断存在性。

类别 0 背景；1 脾；2 右肾；3 左肾；4 胆囊；5 食管；6 肝；7 胃；8 主动脉；9 下腔静脉；10 胰腺；11 右肾上腺；12 左肾上腺；13 十二指肠；14 膀胱；15 前列腺或子宫（单一联合类别）。内部节点索引 0–14 对应标签 1–15。

## 形状和顺序

批次 B，输入 I:[B,1,D,H,W]。最深层 F:[B,C,Df,Hf,Wf]，N=Df*Hf*Wf。C 同时是节点/消息维度；K>0 为轮数，Cr 为关系隐藏维度，da 为匹配维度，Cg 为内容维度，epsilon>0、lambda_c>0。

固定顺序：Encoder → 16 类粗头 → Space-to-Node → K 轮 Node-to-Node → Node-to-Space → 残差融合 → 带多尺度 skip 的 Decoder → 16 类最终输出。

### Encoder 和粗头

- F = Encoder(I)；其余尺度 E1,...,E(l-1) 保留作 skip。
- Lc = Conv1x1x1(F)，P = softmax_class(Lc)，均为 [B,16,Df,Hf,Wf]。
- 背景参与 softmax，只有前景构建节点；用低分辨率 P 建图。

### Space-to-Node

以下公式省略 batch，i=1,...,15，x 属于特征网格 Omega。

```text
Hi(x) = Pi(x) F(x)
Mi = sum_x Pi(x)
zi(0) = sum_x Pi(x) F(x) / (Mi + epsilon)
ua(x) = xa / max(na - 1, 1), (n1,n2,n3)=(Df,Hf,Wf)
ci = sum_x Pi(x) u(x) / (Mi + epsilon)
si = Mi / N
qi = sum_x Pi(x)^2 / (Mi + epsilon)
vi = [zi(0); ci; si; qi]
```

z0:[B,15,C]；c:[B,15,3]；s,q,M:[B,15,1]；v:[B,15,C+5]。本模块无新增可学习参数。Hi 可不物化，不能把池化误写成 P²F。坐标是当前特征网格归一化索引，不是毫米坐标或原图 crop offset；单轴长度 1 时坐标为 0。qi 不是预测正确率。不二值化、不硬屏蔽节点。

### Node-to-Node 和 GRU

15 节点、无自环有向全连接、210 条有效边。i 发送、j 接收；alpha[b,i,j] 沿 i 聚合。

```text
rij(t) = [zi(t); zj(t); cj-ci; si; sj; qi; qj]
alpha_ij(t) = sigmoid(w2^T ReLU(W1 rij(t) + b1) + b2)
mij(t) = alpha_ij(t) Wm zi(t), i != j
mj(t) = sum_(i != j) mij(t)
uj(t) = sigmoid(Wu mj(t) + Uu zj(t) + bu)
rhoj(t) = sigmoid(Wrho mj(t) + Urho zj(t) + brho)
candidate_j = tanh(Wz mj(t) + Uz(rhoj(t) * zj(t)) + bz)
zj(t+1) = (1-uj(t))*zj(t) + uj(t)*candidate_j
zi' = zi(K)
```

rij:[B,15,15,2C+7]；alpha:[B,15,15]；m,z:[B,15,C]。W1:[Cr,2C+7]，w2,b1:[Cr]，b2 标量；Wm:[C,C]。GRU 六个矩阵均 [C,C]、三个偏置均 [C]。

各边独立 sigmoid，不做入边 softmax、均值或强制对称。消息映射按 Wm 矩阵不另加偏置。重置门作用在线性变换之前；更新门是候选写入比例。不能不经公式核验直接替换库 GRU。

每轮读取同一旧状态，计算全部关系和消息，再同步更新。参数在边、节点和轮次间共享。每轮重算关系，仅更新 z；P,c,s,q 保持初值但保留梯度。禁止 detach、外层粗概率重算或截断轮次反传。

### Node-to-Space、融合和输出

```text
Ai(x) = sigmoid((WQ zi')^T (WK F(x)) / sqrt(da) + beta_i)
G(x) = sum_i Ai(x) WV zi'
Fprime = F + phi(G)
Lf = Decoder(Fprime; E1,...,E(l-1))
Pf = softmax_class(Lf)
Yhat(x) = argmax_(c=0,...,15) Pf_c(x)
```

WQ,WK:[da,C]，WV:[Cg,C]，beta:[15]；这些矩阵投影无额外偏置。A:[B,15,N]，G:[B,Cg,Df,Hf,Wf]。phi 是 Cg→C 的 1x1x1 卷积；Fprime 与 F 同形；Lf,Pf:[B,16,D,H,W]。

键来自原 F，查询和内容来自更新节点。Ai 是独立 sigmoid 写入强度而非分割概率。不额外乘原 P，不替换成 Ai*F。不得新增融合系数、替换为拼接或添加监督分支。

### 训练目标

```text
Pc_up = softmax_class(trilinear_upsample(Lc, size=(D,H,W)))
CE(S,T) = -1/|Omega_I| * sum_x sum_(c=0..15) Tc(x) log(Sc(x)+epsilon)
Dice_c(S,T) = (2 sum_x Sc(x)Tc(x)+epsilon) /
              (sum_x (Sc(x)+Tc(x))+epsilon)
DiceLoss(S,T) = 1 - (1/15) sum_(c=1..15) Dice_c(S,T)
SegLoss = CE + DiceLoss
Loss = mean_b [SegLoss(Pf_b,T_b) + lambda_c SegLoss(Pc_up_b,T_b)]
```

粗 logits 先插值再 softmax，不下采样标签替代。病例内计算后 batch 平均，不跨 batch Dice、不跳过缺失器官、不加类别权重或其他损失。log(S+epsilon) 不能静默替换为截断概率或普通 CE。当前统一 epsilon=1e-6，同一符号使用一致约定。

真实标签只用于监督与评估，不输入模型、节点或关系，也不用于存在性判断及裁剪采样。最终损失必须经回写、图、语义和属性路径反传至粗头和编码器；固定属性不表示 detach。推理保留同一核心前向，不计算损失。

## 接口契约

| 模块 | 输入 | 输出 |
|---|---|---|
| Preprocessor | 原影像，可选标签 | 输入影像、可选监督标签、逆变换元数据 |
| Encoder3D | image | F, skips |
| CoarseHead | F | CoarsePrediction(logits, probabilities)，均 [B,16,Df,Hf,Wf] |
| SpaceToNode | F, coarse_probabilities | OrganNodes(z0, mass, centroid, size, confidence) |
| DynamicRelation | z0, centroid, size, confidence | zK:[B,15,C]；可选 RelationResult(zK, rounds) |
| NodeToSpace | 原始 F, zK | G:[B,Cg,Df,Hf,Wf]；可选 NodeToSpaceResult(G,Q,K,V,A) |
| ResidualFusion | 原始 F,G | Fprime=F+phi(G)，与 F 同形 |
| Decoder3D | Fprime, skips | final_logits |
| Segmentor.forward | 仅 image | SegmentorOutput(coarse_logits, final_logits) |
| JointLoss | coarse_logits, final_logits, label | JointLossResult(total, per_case, coarse, final) |
| Evaluator | prediction, label, geometry | 逐病例逐器官指标 |

当前粗头和节点的工程接口分别见 `src/organ_relation/models/coarse_head.py`、`src/organ_relation/models/space_to_node.py`。`OrganNodes` 的节点轴始终对应类别 1–15；centroid 最后一轴依次是特征网格 D、H、W。mass/centroid/size/confidence 使用具名字段，不与 z0 的语义通道混排；全部保留梯度。粗头 bias 与节点 epsilon 必须显式配置；当前 epsilon=1e-6，粗头 bias 仍属于待选模型配置。当前节点数值实现限定 FP32/FP64 且关闭 autocast；混合精度归约策略尚未验证。这些是接口与数值支持范围说明，不改变上述公式。

`src/organ_relation/models/dynamic_relation.py` 实现 `DynamicRelation(channels=C, relation_channels=Cr, rounds=K)` 和显式 `FormulaGRU`。C、Cr、K 须显式给出；默认前向仅返回 zK，`return_diagnostics=True` 返回每轮 `RelationRound(alpha, messages, z)`，分别对应该轮旧状态计算的边权、接收消息及同步更新后的状态，均保留梯度。属性以独立具名参数传入，质量 mass 不进入关系描述；不接受标签或外部边 mask。当前限定 FP32/FP64、关闭 autocast，拒绝非有限输入；不改变公式。参数初始化采用 PyTorch Linear 默认实现，正式初始化及随机种子应随实验配置记录；K 不存于 state_dict，重建时须使用原配置。

`src/organ_relation/models/node_to_space.py` 实现 `NodeToSpace(channels=C, attention_channels=da, content_channels=Cg, beta_init=0.0)`。三个维度显式给出；默认返回 G，`return_diagnostics=True` 返回具名 G/Q/K/V/A，均保留梯度。Q:[B,15,da]、K:[B,N,da]、V:[B,15,Cg]、A:[B,15,N]，空间展开按 D/H/W（W 最快）。三个投影采用无偏置 Linear；beta 为独立参数向量 [15]，默认零初始化是可配置工程初值，不是冻结的正式实验配置。前向只接收原始 F 和 zK，无法从 shape 推断张量来源，调用方须保证来源正确。当前限定 FP32/FP64、关闭 autocast，保留有限性检查；AMD 性能阶段须审查其潜在 GPU 同步开销。本实现不含残差融合。

`ResidualFusion(channels=C, content_channels=Cg, bias=...)` 仅含 Cg→C 的 1x1x1 Conv3d 与 F 相加，无额外尺度、gate、归一化或激活。`Segmentor(SegmentorConfig)` 按上述冻结顺序调用各模块，C 从骨干最深层宽度派生；phi bias 与粗头 bias 分别显式配置。`forward(image)` 仅返回原特征网格 coarse logits 与输入网格 final logits，不额外上采样粗头或执行最终 softmax。`forward_with_diagnostics(image)` 是独立的显式诊断入口，返回具名中间结果并保留梯度；普通前向不创建关系/回写诊断结果或模块缓存。配置见 `src/organ_relation/models/segmentor_config.py`；`configs/segmentor_micro.json` 只用于 CPU 合成测试，不冻结任何正式实验配置。损失独立于 Segmentor，训练入口仅调用已验收的模型与损失。

`src/organ_relation/losses.py` 实现 `JointLoss(epsilon=..., lambda_c=..., align_corners=...)`，三项均无默认值。组合形式已冻结：每分支 CE 与 Dice 系数均为 1，最终分支系数为 1，粗分支系数为 lambda_c；当前 baseline 明确采用 lambda_c=0.5、epsilon=1e-6、粗分支插值 align_corners=False，不从微型配置或骨干插值设置推断。lambda_c 未来可通过实验调整，但本轮 GPU 可行性验证统一为 0.5。label 必须为与 final 网格同形的 `[B,D,H,W]` int64 类别索引 0–15。结果 total 为 batch 均值标量，per_case 为 `[B]`；每分支返回 ce/dice_loss/segmentation:[B] 与 dice_per_class:[B,15]，保留梯度但不返回完整概率体积。gather/scatter_add 等价计算真类概率、逐类交集和计数，不生成稠密 one-hot GT；无概率截断、标准 CE 替代或空类屏蔽。严格 log(S+epsilon) 在 S=1 时可产生负 CE，属于原公式结果，不另作截零。调用方须与模型共用 epsilon；当前仅验证 FP32/FP64 CPU，保留 finite/autocast 检查。

## 当前 baseline 与完整扫描数据契约

2026-09-25 用户确认：epsilon=1e-6（节点与 loss 一致）；lambda_c=0.5（第一版 baseline 辅助权重，未来允许实验调整）；粗 logits 三线性插值 align_corners=False（当前实现约定）。配置见 configs/baseline.json。三个构造参数继续显式传入，算子公式不变；骨干内部插值参数仍是独立模型配置。

正式模型容量没有冻结：channels、层数/逐轴 stride、归一化、bias、K、Cr、da、Cg 均不能由 segmentor_micro.json 推断。baseline.json 的 model 留为 null；完整 GPU 前向必须另外传入完整模型配置，不自动选宽度或轮数。

数据入口 src/organ_relation/data/full_scan.py 的 FullScanPreprocessor / FullScanDataset 每次返回一整例单样本：image=[1,D,H,W] float32，label=[D,H,W] int64，均在 CPU。channel 属于样本，batch 不属于 Dataset；DataLoader(batch_size=1) 默认得到 image=[1,1,D,H,W]、label=[1,D,H,W]。单病例 validate_full_scan.py 不使用 DataLoader，在搬运阶段显式 unsqueeze(0) 后进入模型与 loss。样本 metadata 不记录 batch_size，probe 配置仍记录 batch_size=1。此次接口调整不改变体素、几何或重采样定义。无 crop、patch、滑窗、GT ROI、全局 padding 或有效域 mask，标签只用于配对检查与监督。

当前可行性预处理协议复用已验收的底层几何：只支持 NIfTI-1、毫米单位、轴对齐网格；图像和标签各自 scaling / affine 分别处理，同一目标 RAS 网格。保留完整体素单元边界、ceil 尺寸、中心保持，最近边界复制；CT 降采样方向采用既有 Gaussian 预滤波和三线性插值，标签最近邻。A/B/C=(1.5,1.5,3)/(2,2,3)/(2,2,5) mm 全部保留。配置中的上述预处理为本轮可行性协议，不冻结正式训练预处理或最终 spacing。

目标数组 R,A,S 转置成 Tensor 的 D=S,H=A,W=R，记录原图像/标签 affine、目标 affine、Tensor affine 和双向索引映射。逆映射可供后续原空间恢复，不表示降采样信息可逆。本轮不实现预测恢复或正式评估。

当前强度模式显式为 scaled_hu：只遵循每文件 metadata scaling，不统一减 1024，不额外窗截断/归一化。正式训练强度方案仍待决定，不将显示窗 [-160,240] HU 作为模型预处理。

scripts/validate_full_scan.py 为单病例分阶段工程验证：预处理→搬运→模型初始化→完整前向→JointLoss→反向→可选一次 optimizer step；支持中途停止，固定 FP32/batch 1，无 AMP、epoch、scheduler 或 checkpoint。optimizer 类型与 lr 必须由调用方明确给出，其他工程探针选项完整记录；不构成正式优化器配置。OOM 停止并保留结果，不缩小输入或自动换候选。CPU 模式仅用于有体素上限的合成小例。

## 尚待确认，不能当作已冻结配置

- 完整输入的 target spacing、尺寸约束、padding 是否进入节点统计/损失；保留完整范围的具体重采样网格与逆变换协议。
- 强度截断/归一化参数、空间增强及左右方向处理。仅训练集可以拟合统计参数。
- 具体 U-Net 尺度、层宽、归一化、上下采样、卷积偏置；K、Cr、da、Cg；优化器与正式训练配置。
- 后续 lambda_c 消融实验设置；第一版 baseline 的 lambda_c=0.5 已确定，不属于当前待决项。
- 官方 validation 本地目录、病例数与来源的实际核实，以及 test 清单与官方导出格式核实。
- 最终 checkpoint 选择、原空间概率/标签恢复顺序、官方 NSD 的严格定义/容差及指标对齐；内部 Dice 规则见下文，不冒充官方评价。
- 小器官保真度没有预设合格阈值；先提交候选实际几何损失、图像质量风险和显存对照，再共同决定。

工程可决定等价向量化、分块、配置组织、诊断开关、设备/路径处理。影响结果的配置必须单独记录并在正式实验前确认，不能因 8GB 辅助卡改变正式方法。

## 分阶段验收

1. 已完成元数据读取 CPU 测试及 200 例训练 CT 头信息统计、三候选网格估算，以及其中 10 例的真实 CPU 重采样保真度试验。候选分析见 docs/preprocessing_candidates.md，探索协议和实测验收记录见 docs/fidelity_protocol.md、docs/fidelity_cpu_validation.md。没有冻结保真度阈值、正式 spacing/尺寸/边界或有效域；不代表全部 200 例体素检查或 GPU 验证。
   数据阶段已获用户验收；第一阶段 Encoder/Decoder 骨干与微型 CPU 测试已实现，结构接口见 docs/backbone_stage1.md，实测见 docs/backbone_cpu_validation.md。仅骨干直接连接不代表已实现下述图模块或联合损失；正式层数、宽度、归一化、步幅、插值约定仍未冻结。
2. CPU 公式：手算软池化/属性，方向/无自环/求和，独立 sigmoid，指定 GRU，同步动态更新、回写和损失。
3. CPU 梯度：平滑点数值梯度与参考实现比较；最终损失经过语义/固定属性路径；标签变动不得改变前向输出。完整小网络集成与优化器覆盖测试。
4. AMD 主环境：正式结构的前向/双损失/反向/优化器状态显存、真实数据处理、短测及恢复；之后才允许正式训练。
5. 重要集成节点做 NVIDIA CUDA 模块和缩小输入兼容测试，不替代 AMD 验收，不要求跨平台逐位一致；容差预先声明。

AMD 测试代码/命令由开发端提供，用户 GitHub 同步后手动执行返回结果。当前不建立远程环境。

## 训练系统 v1 与数据角色（不改变上述方法公式）

- 200 official training CT 为 development pool；显式生成一次 160 train / 40 internal-dev 病例列表，保存 manifest、seed 与 SHA-256。架构、3/4-stage、容量、spacing、超参数和消融在此选择，不创建自定义 test。全部配置冻结后使用完整 200 例从头训练最终模型。
- Linux 核实 official imagesVa/labelsVa 来源后，才可用于冻结模型的 held-out evaluation；不加入训练，不反复调参。official test 是 image-only，最终计划提交 AMOS CT Regular Evaluation (Test)，报告官方 overall 与逐器官 DSC/NSD。官方导出格式、NSD 实现与评估对齐后续验证；不按 leaderboard 反复调参。
- configs/train_sanity.json 是系统验收配置：3S-P1、B、FP32、batch=1、workers=0、channels_last_3d。AdamW lr=3e-4、betas=[0.9,0.999]、eps=1e-8、weight_decay=0、foreach=false、fused=false，无 scheduler。不是正式容量、spacing 或最终训练参数。沿用当前 scaled_hu，不新增预处理。
- scripts/train.py 统一用于 smoke→单病例 overfit→双病例 overfit→恢复验收→short pilot→后续正式训练；运行长度、病例、验证/诊断/checkpoint 频率显式配置。当前新增系统仅做 Windows CPU 合成验收；既有 AMD capacity backward 结果不等于 AdamW 稳态训练已通过。
- checkpoint 在完成 optimizer step 并提交标量日志后原子替换，保存 model/optimizer、计数、实际病例顺序/游标、全部 RNG/generator、配置、来源和 split/subset 哈希、日志提交位置及 pending validation。普通 resume 严格匹配这些运行身份；失败半步不提交。诊断/保存频率可调，不强制正式训练每步扫描或保存。恢复算法不要求 AMD 非确定性算子逐位一致。
- 内部 validation 在当前重采样网格计算逐病例、逐前景器官 hard Dice，与 JointLoss soft Dice 分开。GT/pred 任一非空正常计算；两者均空记 null 并排除均值；GT 空而预测非空为 0，另记 FP 体素。先病例内有效前景平均，再病例平均；另报逐器官均值和有效病例数。不实现近似 NSD/HD95，不代替官方原空间指标。
- 训练 ETA 采用最近 20 个完整 step 的均值，少于 5 个显示 warming up；验证/推理使用独立计时器，resume 重新预热。训练 ETA 明确仅估算训练计算与取数，排除验证/checkpoint 写盘，不宣称是包含所有 I/O 的精确完成时刻。
- CaseLedger 提供安全结果写入、身份/完整性核验与按病例恢复；内部验证已接入。official validation/test 角色可在 split 数据结构表达，但本版训练 CLI 不加载它们。无标签预处理入口、原空间 NIfTI 导出与完整 challenge pipeline 尚未实现。双向物理映射可支持后续恢复到原网格，不代表降采样信息可无损恢复。
