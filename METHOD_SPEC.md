# 完整扫描范围动态器官关系分割规范

状态：算法公式与全局上下文已确认；实验配置待确认。2026-09-22 建立。

## 当前 full-development A/C 实验协议（2026-09-27 用户确认）

当前唯一新增模型机制（2026-09-27 用户授权）：正式 C-gated 使用 `F'=F+gamma*phi(G)`，单个全局可学习 scalar `gamma_init=0.1`，不限制符号，不使用 clamp/sigmoid/softplus/正则/warmup/schedule。必须先计算 phi(G)，再乘 gamma（包括缩放 phi bias）。配置为 `train_monai_relation_gated_A_160_40.json`，其 relation 中显式设置 `learnable_relation_scale=true, relation_scale_init=0.1`；模型配置与旧 C 只差这两个字段，训练协议更新见下文。纯 A 不加 gamma。旧配置/default 为 learnable=false、effective gamma=1，保留 `F+phi(G)` 的原计算、参数键与初始化，不修改旧 B-spacing C 配置。以下历史无缩放公式仍对应 legacy 模式；本变体不改变 c/s/q、GRU、attention、K、loss/coarse weight 或其他结构。

Gated 观测：同次前向以整张 F、phi(G)、gamma*phi(G) 的全 tensor L2 norm 计算两个比值，分母为 `||F||_2+1e-6`，使用 detached FP64 norm reduction；不改变前向/梯度/RNG，不保留激活历史。每个训练 step JSONL 的 `relation_scale` 记录更新前 gamma、`writeback_to_feature_norm`（未缩放）及 `scaled_writeback_to_feature_norm`（实际扰动），另记录更新后 `gamma_after_step`；验证逐病例保存在 ledger 与 `relation_scale_cases`。当前 TensorBoard 每 epoch 额外记录 `Relation/Gamma`（epoch 末更新后的值）和 `Relation/WritebackToFeatureNorm`（该 epoch 同次训练前向实际扰动比的病例均值）；baseline 不生成这两条曲线。gamma 随 model/optimizer 保存恢复，relation 配置纳入严格 identity，不允许 legacy checkpoint 静默迁移到 gated 模式。

本节更新当前实验配置，保留下面的原始方法转录与历史 diagnostic 记录。A 为已有 MONAI reference；C 为同一个 MONAI backbone 加已有粗头、节点构建、动态关系/指定 GRU、节点回写与残差融合。骨干保持 channels=[8,16,32,64,128]、strides=[2,2,2,2]、num_res_units=2、InstanceNorm/PReLU；C 保持 Cr=8、K=2、da=4、Cg=6，不重新搜索关系参数。

- 使用已有 `runs/splits/development_160_40.json`，其 JSON 内 `split_hash=7d308eca4f7324f0e899c7416a45a03f8dfbec5e867e5ed6018353e96541468c`。启动同时验证内部 manifest/partition 哈希、train=160、internal-dev=40、与实际 official training manifest 一致；不生成新 split，不从 validation 抽样。文件 SHA256 单独记录，不能代替 split_hash。
- 当前 A-spacing 比较使用完整扫描、[1.5,1.5,3] mm、原 full-scan 几何/scaling；MONAI 输入层 clip HU [-1000,1000]、除以 1000 到 [-1,1]，按已验收实现对高端做最小 16 整除 padding，值 -1。一次整例前向；final logits 去除新增边界后，对原重采样网格 GT 计算 loss/metric。C 的节点域和 coarse logits 到 GT 网格的插值沿用既有 MONAI adapter；不修改几何、GT、不加有效域 mask。
- A 与 C 的 final loss 使用同一个已实现 MONAI DiceCELoss：16 类标准 CE + 15 前景 Dice，smooth_nr=smooth_dr=1e-5、per-case、lambda_ce=lambda_dice=1。C coarse logits 先三线性插值到完整 GT 网格（align_corners=False），再用同一 MONAI loss，total=final+0.5*coarse。本比较不调用历史自定义概率 CE；原 JointLoss 及所有 diagnostic 配置保持原行为。
- 统一 seed=20260925；model/Python/NumPy/PyTorch RNG 同种子，sampler 独立 generator 同种子，loader generator=seed+1。记录并恢复全部 RNG；不依据 seed 重建 split。A/C 骨干初始化保持已有逐 tensor 对齐行为。
- FP32、batch=1、workers=0、ROCm/channels_last_3d；AdamW lr=3e-4、weight_decay=0，其他参数沿用已有配置。固定 LR，无 scheduler/AMP/activation checkpointing/gradient accumulation。
- 当前 A/C-gated 均为 `development_160_40_v2`：max_epochs=500（最多 80,000 steps）、min_epochs=100。每 epoch 汇总同次训练前向 total/final/coarse loss 和 final hard Dice。独立 checkpoint_every_steps=5，在完整 optimizer step 边界原子保存 last.ckpt；保留 checkpoint_every=1（epoch），epoch 末即使不是 5 的倍数也保存，验证前后及受控停止原有保存语义不变。每 5 epochs 单次整例前向验证全部 40 例，无后处理；记录 final/coarse loss、hard/soft Dice 及逐器官结果。Hard Dice 保留 both-empty=null 的病例/器官协议，soft Dice 保留 MONAI 定义。
- Early stopping 只看完整 dev 的 final mean-case foreground hard Dice，区分 raw best 与 patience reference。raw best 在任何严格新高时更新并保存 best-dev，平局保留较早 checkpoint；reference 首次验证初始化，此后仅当 current >= reference + 1e-4 时更新并清零计数，否则计数加一且 reference 不变。连续小幅上涨可累计达到阈值。连续 5 次（25 epochs）未达到该阈值且 epoch>=100 时停止；100 之前累计计数但禁止停止，最早可在 100 停止。gamma 不参与判断。
- checkpoint 包含 early-stopping best_metric/best_epoch（raw best）、patience_reference_metric、no_improvement_count/stopped、完整 validation history、epoch 聚合进度和原有全部恢复状态；resume 重放历史校验状态，已停止的 run 不会恢复后继续训练。v2 不允许用 controlled extension 绕过 500 上限或 early stop。旧 v1 的 300→400→500 延长机制仅保留供历史配置；跨协议/source/config/split 不兼容继续拒绝。
- 正式训练 `loss_observation=monitor_only`：可微路径只调用官方 DiceCELoss 标量，coarse 只上采样一次；不计算额外 per-class DiceLoss、detached CE/softmax 或概率诊断。final hard Dice 在 backward 后用现有 logits.argmax 统计；train soft Dice/coarse hard Dice 不伪造或重复分配全尺寸概率，其完整统计留到 validation。旧 loss.forward 诊断 API 与历史 B 配置保留。
- 正式 console_every_steps=5：每 5 步显示 epoch/global step/epoch 内位置、当前与最近最多 5 步 total loss 均值、当前 final/coarse loss、同次前向 gamma/实际扰动比、LR、GPU allocated 峰值、原 rolling step time/ETA；近期 console loss 窗口恢复后重新累积。逐 step JSONL 不删减，console 摘要不另写机器日志。TensorBoard 每 epoch 记录 loss/final hard Dice/gamma/扰动比，每 5 epochs 记录完整 validation loss、hard/soft Dice 和逐类 hard Dice；逐类 soft Dice、CE 分解及概率诊断保留 validation JSONL/ledger。No-update preflight 在 amos_0097 测 forward/loss/backward，仍保留两分支必要 autograd，不使用 empty_cache 掩盖生命周期；实际显存须 AMD 复测，OOM 即停止。
- 最大病例 preflight 通过后可用同一正式配置 `--stop-after 5 --run-dir runs/preflight_gated_A_5steps`：按冻结 160 train 的正常 seed/order 运行前 5 个病例，真实执行 optimizer 更新并清空梯度；独立于正式 run。报告 step 1、step 2–5 均值/中位数及各步 loss/gamma/扰动比/显存/ETA。同步计时包含读取、计算及轻量指标，不含随后日志/checkpoint 写盘；只是不同 shape 下的粗略 throughput，不是科研结果。

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

独立诊断对照（用户授权，非上述 JointLoss baseline 的替换）：`SegmentationLoss` 与 `JointLoss` 均可显式选择 `ce_reduction_mode=foreground_background_balanced`。保持 `l(x)=-log(S_true(x)+epsilon)`，逐病例计算 `0.5*mean(l|label=0)+0.5*mean(l|label>0)`，再加原有 15 类 DiceLoss，最后 batch 平均；不进一步均衡前景器官。缺省 `voxel_mean` 与所有历史配置仍用原公式。若某病例无背景或无前景，该组均值未定义，当前对照明确拒绝，不补零或另设权重。配置 `train_backbone_only_overfit_balanced_ce.json` 仅增加 CE reduction 选项，从 step 0 独立运行；mode 纳入 checkpoint 身份，不允许从 voxel-mean 运行迁移或延长为 balanced 运行。

完整模型的独立 balanced-CE 对照：coarse logits 仍先三线性上采样到 GT 网格，再 softmax；final 与 coarse 各自使用同一 CE reduction 和各自原 DiceLoss，`total=mean_batch(final.segmentation+0.5*coarse.segmentation)`。不下采样 GT，不改 lambda_c=0.5。`train_single_case_overfit_balanced_ce.json` 与原完整 overfit 配置只差 CE mode 和 max_steps=200；病例通过 `--cases amos_0109` 指定，从新 run 的 step 0 开始。

用户授权的组权重对照：两种 loss 的 `foreground_background_balanced` 模式可配置 `ce_background_weight=w_bg` 与 `ce_foreground_weight=w_fg`，使用 `w_bg*CE_bg_mean+w_fg*CE_fg_mean`；两者有限且严格大于 0、和为 1（仅浮点求和校验容差 1e-12，不自动归一化）。两项缺省均为 0.5，保持历史 50:50 的计算顺序与结果；voxel_mean 不使用组权重。70:30 backbone-only 对照仅显式设置 0.7/0.3，不改 Dice、初始化、lambda_c 或其他训练条件。新 run identity 保存 resolved 权重；旧身份缺失字段只在比较时按 0.5/0.5 解释，不修改原 run.json/origin identity/哈希，不放宽来源或其他配置校验。

用户授权的 hierarchical foreground CE 诊断：`foreground_ce_reduction` 缺省为 `voxel_mean`；显式 `class_macro_mean` 仅与 `ce_reduction_mode=foreground_background_balanced` 配合。每病例先计算实际存在的前景类各自的 `CE_c=mean(l|y=c)`，再等权平均得到 `CE_fg_macro`，CE 为 `w_bg*CE_bg+w_fg*CE_fg_macro`。缺失类不进入 CE macro，不补零；GT 仅在 loss/metrics 使用，不改变图节点或模型前向。Dice 仍为全部 15 类逐病例均值。JointLoss 两分支使用相同 foreground reduction，coarse 仍先插值 logits 到完整 GT 网格。整例无前景或无背景时沿用明确拒绝规则。首个配置 `train_backbone_only_overfit_fg_class_macro.json` 只比原 50:50 backbone balanced 配置增加 foreground reduction，使用默认 0.5/0.5，从新 run 的 step 0 开始。旧身份缺失该字段仅在比较时解释为 voxel_mean；原身份/哈希与严格来源校验保留。

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
