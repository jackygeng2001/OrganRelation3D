# 脑肿瘤动态区域关系分割：当前工程规范

2026-10-31 起以用户提供的《脑肿瘤分割方法_修订版.docx》为方法依据。仓库与包名保留 OrganRelation3D / organ_relation；不提前命名论文方法。早期腹部 CT（AMOS，15 节点）阶段的代码、配置与文档已于 2026-11-09 从工作树删除（git 历史中保留），不用于本方案。

方法文档 SHA256：`76cf95d033b0026539cf38c1d3f07240223928dde854d86c48ee2ae676258a0b`。
2026-11-05 用户提供《脑肿瘤分割方法_层级一致性修订版.docx》（SHA256：`b687a888cf65a38d3d42f00519c54f2a46c4314c5e72ac28b223a26f14093691`）。逐段比对，相对上一版只改动损失：新增仅作用于最终预测的层级一致性损失（新式 32）、总损失改为新式 33，并补充参考文献 [14][15]；网络结构与式 1–31 不变。
实验参考 PDF《Deep Hierarchy-Aware Segmentation: A Novel Framework for MRIs Brain Tumor Segmentation》SHA256：`43dd46d0bff87a399884fa3773fc8d153b88b7e1a2d0dfbd5a87fa573fb88065`。参考论文不是新方法的结构来源；若用户另有第二篇实验参考，须再核实。

## 输入、区域与上下文

- 完整配准 MRI，顺序 T1、T1ce、T2、FLAIR；image `[B,4,D,H,W]`，FP32。无 patch、滑窗或标签引导空间处理。
- 可选整脑外接框（2026-11-05 用户批准，显式 `preprocessing.spatial_crop={kind:nonzero_bbox,margin,multiple_of}`，默认 null 为原网格）：框只由四模态非零体素的并集决定，再按 margin 外扩、按 multiple_of 居中补足并限制在原体积内；所有非零体素在框内，故逐模态 z-score 统计不变。若任何标注肿瘤体素落在框外则加载直接报错（不静默丢弃）。每例仍是整脑一次前向；节点 c/s 相对当前输入框，BCE 按框内体素平均。框外等价于预测背景，在 GT 全在框内时框内 Dice 等于贴回原网格的 Dice。`scripts/brats_crop_audit.py` 只读核验全部病例。
- 原标注统一到 0/1/2/3：背景、坏死/非增强核心、瘤周水肿、增强肿瘤。三重叠区域顺序 WT={1,2,3}、TC={1,3}、ET={3}。
- coarse/final 均为三个独立 sigmoid 通道。背景不单列通道/节点。始终三个节点，GT 只进入损失与指标。
- 推理每通道独立以 `P>=tau` 二值化；不做 argmax，不额外强制 ET⊆TC⊆WT。监测可以报告层级违反数。
- 奇数尺寸直接支持：每层 stride-2 卷积的实际形状为 ceil(size/2)，解码插值到实际 skip 尺寸，最终严格恢复输入网格。无需为了整除引入 padding。

## 编码器（式 1–5）

四层通道固定 32/64/128/256。所有编码/解码卷积单元为 Conv3d→GN(8 groups)→LeakyReLU(0.01)。

对 FP32/FP64 的 channels_last_3d 特征，GN 使用等价布局保留实现：无拷贝拆分通道为 `[B,8,C/8,D,H,W]`，在每病例每组的通道及**完整空间**上计算总体方差（correction=0）与均值，保持原 epsilon/逐通道 affine；将小尺寸 scale/shift 合并后用 `addcmul` 输出原布局。统计量正常参与 autograd，不 detach，不按空间块分别归一化。前向不创建完整的 NCDHW 拷贝或中心化中间体。反向在 PyTorch autograd Function 中实现解析梯度：先按完整空间归约每通道 sum(dy)/sum(dy*x)，再按原组累加带 gamma 的统计量，生成 dx/dgamma/dbeta；元素乘积和 dx 写入分段执行（FP32 每份临时张量至多约16 MiB），不改变全局统计、不重新计算网络前向、不做 activation checkpointing。均值和逆标准差作为辅助 autograd 输出，保留二阶导数；CPU gradcheck/gradgradcheck 及原生 GN 梯度对照验证。普通连续布局走原生 GroupNorm；参数名、初始化及组数不变。此实现只用于当前脑肿瘤模型，旧模型不变。

1. E0：两次 3³、stride=1、padding=1 卷积，4→32→32。
2. 每层 l=1,2,3 先 3³ stride=2 padding=1 下采样，得到 X_l。
3. 对同一 X_l 独立计算 local 3³ dilation=1 pad=1 与 context 3³ dilation=2 pad=2，两路都有 GN/激活。
4. 拼接 `[local;context;abs(local-context)]`，1³ 卷积 3C_l→C_l/4→2，中间 LeakyReLU；逐位置沿两分支 softmax。
5. `V=w_local*local+w_context*context`；`E_l=X_l+Pi_l(V)`。Pi 为通道不变的 1³ 卷积，不额外加 GN/激活。

F=E3，C=256；原始 E2/E1/E0 进入关系解码器。

显式运行选项 `--convolution-backend depth_sliced` 只改变卷积执行：对原 3D kernel 的每个深度切面执行 NCHW Conv2d，并将其加到相应的完整 3D 输出位置；按原 stride/padding/dilation 计算输入索引，越界贡献为零。每批最多 4 个输出深度平面，反向分别计算并累加入输入及原 3D 权重梯度；只保存原输入/权重，不重算网络前向。1³ 卷积使用等价逐位置线性映射。跨切片感受野不变，GN/关系/后续层始终接收完整体积，不是 patch segmentation 或改成 2D 模型。浮点求和次序不同，以输出、一阶/二阶梯度容差验证。参数对象、键、初始化、dtype 及模型配置不变；默认 native 原路径保留，执行选项记入 runtime identity，不能在普通 resume 时切换。平面内空洞率大于 1、stride 1、'same' padding 的切面卷积（双感受野 context 分支）按位置对空洞率取余拆成 dilation_h×dilation_w 张交错子图，各做一次同权重的普通 Conv2d 后放回原位：每个输出仍是同一组输入与权重的同一组乘加，只是求和次序不同（FP64 下与 F.conv3d 差 1e-14 量级）。原因是 MIOpen 对空洞 Conv2d 只有 naive 内核（7900 XTX 上每次调用慢 18–35 倍）；BraTS2018 一例同权重 FP32 训练步 3.10→1.05 s，梯度相对差 1e-5。runtime identity 记 `convolution_dilation=interleaved_v1`。

`depth_sliced` 的解码融合进一步避免完整 `cat(decoded, enhanced)` 存储：在每个有界平面批次内按相同通道顺序拼接后调用同一 Conv2d，保存两个原始输入而非整幅拼接副本。反向将卷积输入梯度按原通道范围分别累加；原 3D 卷积权重、感受野、GN/激活和完整输出均不变，不重算网络前向。默认 native 路径保持原整幅拼接。

## 节点构建与关系（式 6–16）

`Lc=Conv1x1x1(F)`，`P=sigmoid(Lc)`。P 只用于节点构建，不重新乘到空间回写。

```
M_i = sum_x P_i(x)
z_i = sum_x P_i(x) F(x) / (M_i+epsilon)
c_i = sum_x P_i(x) u(x) / (M_i+epsilon)
s_i = M_i/N
q_i = sum_x P_i(x)^2 / (M_i+epsilon)
u_a(x)=x_a/max(n_a-1,1), axes D,H,W
```

张量 alpha 的轴是 `[batch,sender,receiver]`，无自环。关系图由显式 `model.graph` 决定：

- **嵌套双向链**（`graph=nested_chain`，2026-11-10 老师批准，当前方法）：节点 [WT,TC,ET] 由外到内，边集 E = E_in ∪ E_out，WT 与 ET 之间无边。向内边 E_in = {WT→TC, TC→ET} 表示包含关系，用外层区域约束内层的范围；向外边 E_out = {ET→TC, TC→WT} 用内层区域帮助外层确定位置。两类边各有一套边门控 MLP（W1,b1,w2,b2）与消息矩阵 Wm，记为上标 τ(i,j) ∈ {in,out}；GRU 两类共用。
- **完全图**（省略 graph，历史实验 E1/A3）：六条有向边 i≠j，只有一套边门控与 Wm。

```
r_ij = [z_i;z_j;c_j-c_i;s_i;s_j;q_i;q_j]   # 2C+7
alpha_ij = sigmoid(w2_τ^T ReLU(W1_τ r_ij+b1_τ)+b2_τ)   for (i,j) ∈ E；不在 E 中的 alpha_ij = 0
m_j = sum_((i,j)∈E) alpha_ij Wm_τ(i,j) z_i            # 接收端求和，不平均
z_new_j = GRU(m_j,z_j)
```

实现：每类边的 MLP 对 3×3 全部节点对打分后，用该类的固定结构掩码置 0（与无自环掩码同类，与标签无关），两类掩码互不重叠，相加即 `[B,3,3]` 的 alpha；聚合时再按掩码拆回两类，各乘自己的 Wm 后相加。被掩去的位置值恰为 0、对参数无梯度。两个掩码是非持久 buffer（随模型移动设备，不写入 state_dict 与 checkpoint，不消耗随机数）。参数名 `inward_hidden/inward_output/W_inward` 与 `outward_*`，构建顺序为向内、向外、GRU；完全图的参数名、构建顺序、初始化与计算不变（CPU 训练场景新旧代码日志与 checkpoint 逐值一致）。在 Cr=64 下双向链比完全图多一套边门控与 Wm（+98,881），完整模型 8,486,001 参数。链上 WT 与 ET 的信息只能经 TC 中转，K=2 是二者能互相影响的最少轮数。尺度描述中每个区域的两个入边门控按发送者升序排列，链上不存在的入边（WT 自 ET、ET 自 WT）恒为 0。

每轮从同一旧 z 同步计算全部消息后更新，所有轮次共享参数。c/s/q/P 保持不变但不 detach。最后输出 z(K) 与实际末轮 `alpha(K-1)`；不额外基于 z(K) 重算边权。

修订文档只写标准 GRU，未展开。当前工程沿用已测试的 reset-before-linear 版本（仍需在实验定义中明确）：

```
u = sigmoid(Wu m + Uu z + bu)
rho = sigmoid(Wrho m + Urho z + brho)
z_tilde = tanh(Wz m + Uz(rho*z) + bz)
z_new = (1-u)*z + u*z_tilde
```

W 各有一个偏置，U 无偏置，Wm 无偏置。

## 关系推理 v2（2026-11-11，用户已同意；效果待正式实验检验）

依据：EXPERIMENTS.md 5.5。三项都由显式配置开启；省略时上一节的模型、参数初始化、损失与 identity 完全不变。

**归一化与残差读出**（`model.relation_conditioning = layernorm_residual`）。式 12–16 的形式不变，只是作用在归一化后的状态上：

```
h_i(0) = LayerNorm(z_i(0))                      # 逐节点，256 维，可学习仿射
r_ij(t) = [h_i(t); h_j(t); c_j-c_i; s_i; s_j; q_i; q_j]
alpha_ij(t), m_j(t), h_j(t+1) = GRU(m_j(t), h_j(t))   # 同式 13–15，以 h 代 z
z_i' = z_i(0) + W_o h_i(K)                      # W_o ∈ R^{C×C}，无偏置，零初始化
```

初始时 z' = z(0)，与不做关系推理的模型相同；送入解码器的仍是 z'、(c, s, q) 与末轮边权 alpha(K-1)。新增的 LayerNorm 与 W_o 排在原有参数之后构建，原有参数的初始化不变。

**包含读出**（`model.containment`）。只定义在两条相邻嵌套的向内边 (i→j) ∈ {WT→TC, TC→ET} 上，取末轮之后的状态：

```
[u_ij, e_ij] = W2^c ReLU(W1^c [h_i(K); h_j(K); c_j-c_i; s_i; s_j; q_i; q_j] + b1^c) + b2^c
```

u_ij 是包含比例 |T_j|/|T_i| 的 logit，e_ij 是"内层区域 j 存在"的 logit（偏置初值 4，即起点近似都存在）。W1^c ∈ R^{Cr×(2C+7)}，与边门控 MLP 同形但参数独立。`containment = supervised` 只输出 u（仅受监督）；`prior_shift` 另把内层 logit 整体平移 10·a_j·clamp(u_ij, ±6)（a_j 零初始化，平移处 u 取 detach；实测无益，保留作对照）；`gate` 输出 u 与 e 并做存在性门控。

**存在性门控**（`containment = gate`）。记解码器输出为 L，

```
pi_TC = sigmoid(e_WT→TC),   pi_ET = pi_TC * sigmoid(e_TC→ET)
P_WT = sigmoid(L_WT),   P_TC = pi_TC * sigmoid(L_TC),   P_ET = pi_ET * sigmoid(L_ET)
```

模型输出 logit(P)（用 logsigmoid 与 logaddexp 计算，任何有限输入都得到有限值），损失、阈值与评估都作用在它上面；WT 通道不变。pi → 1 时退回原输出，pi → 0 时整块区域被关掉。

**包含监督**（`loss.lambda_contain`，需同时给出 `lambda_hier`）。真值只进入损失：

```
rho_ij = |T_j| / |T_i|,   y_ij = 1[|T_j| > 0]                 # 真值体素数；|T_i| = 0 的边不计
L_contain = mean_valid_edges [ BCE(u_ij, rho_ij) + BCE(e_ij, y_ij) ]   # 没有 e 时只有第一项
L = L_seg(final) + lambda_c L_seg(coarse) + lambda_hier L_hier + lambda_contain L_contain
```

**额外的节点输入**（`model.node_inputs`，需要 layernorm_residual）。都以同一组粗概率 P 为权重（P 三线性插值到相应网格），只加到关系状态上：h_i(0) = LayerNorm(z_i(0)) + x_i；残差读出仍以 z(0) 为底。

```
multiscale: z_i^(l) = sum_x P_i(x) E_l(x) / (sum_x P_i(x) + eps),  l = 2, 1     # 1/4、1/2 分辨率，128、64 维
            x_i += W_l LayerNorm_l(z_i^(l))                                    # W_l 无偏置，映射到 C 维
intensity:  mu_i,m = sum_x P_i(x) I_m(x) / (sum_x P_i(x) + eps),  m = T1, T1ce, T2, FLAIR
            sd_i,m = sqrt(max(E_i[I_m^2] - mu_i,m^2, 0) + eps)
            x_i += MLP([mu_i; sd_i])                                           # 8 → 64 → C，LeakyReLU(0.01)
```

池化在 FP32 中累加；粗概率不 detach。最深层网格上 TC 与 ET 的掩码几乎重合，较细的尺度和各模态的原始对比度是深层池化向量里没有的信息。

**逐体素嵌套输出**（`model.output_hierarchy = nested`）。记解码器输出为 L：

```
P_WT = sigmoid(L_WT),   P_TC = P_WT * sigmoid(L_TC),   P_ET = P_TC * sigmoid(L_ET)
```

即 L_TC、L_ET 分别是“在 WT 内属于 TC”、“在 TC 内属于 ET”的条件 logit（先外后内的级联，单网络形式）。模型输出 logit(P)，必然满足 P_ET ≤ P_TC ≤ P_WT，层级一致性项恒为 0。与存在性门控同时启用时先做嵌套分解，再乘存在概率。

**协议 P2**（`training.scheduler = {"kind": "cosine", "min_lr": 1e-6}`，`max_epochs = 80`，不设 `early_stopping`）。第 t 步（从 1 起）的学习率为 min_lr + (lr − min_lr)·0.5·(1 + cos(π (t−1)/T))，T 为总步数；它只是 (t, T) 的函数，没有状态，续训逐步相同；有调度的 run 不能用 `--extend-to` 延长。

参数量（Cr = 64，嵌套双向链）：归一化 + 残差读出 +66,048；包含读出 +33,410（只有比例时 +33,345）；multiscale +49,536；intensity +17,216。R3 8,552,049，R3A 8,585,459，`p2_r4_full` 8,652,211。

## 深层回写（式 17–19）

```
Q=WQ z(K), K=WK F, V=WV z(K)
A_i(x)=sigmoid(Q_i^T K_x/sqrt(d_a)+beta_i)
G(x)=sum_i A_i(x)V_i
F'=F+gamma*phi(G)
```

Q/K/V 无投影偏置，beta 为区域偏置。gamma 是无符号约束的全局可学习标量，乘在 phi(G) 之后，不改为 phi(gamma G)。

## 关系尺度自适应解码器（式 20–26）

按 E2、E1、E0 解码。每尺度独立学习 Q_l:256→32、K_l:C_l→32、V_l:256→32 和三个 beta_l；原始 E_l 为 key，z(K) 为 query/value。空间注意力独立 sigmoid。

每接收区域 j 的尺度描述：

```
xi_j=[z(K)_j;c_j;s_j;q_j;alpha_i1,j(K-1);alpha_i2,j(K-1);onehot(j)]
```

共 266 维。发送方槽按编号升序排除自己，依次是 WT←[TC,ET]、TC←[WT,ET]、ET←[WT,TC]。共享 MLP 266→64→3，中间 LeakyReLU(0.01)，按 **尺度维** softmax。三个分量依次对应 E2、E1、E0；各区域独立归一化。

```
G_l(x) = sum_i w_i,l A_i,l(x) V_l,i
E_enhanced_l = E_l + eta_l Phi_l(G_l)
```

Phi_l 为 **无偏置** 1³ 卷积 32→C_l；三个独立 eta 初值 **1e-3**，不限制符号。残差相加后不再归一化。

解码器仅调整线性计算的结合顺序：`Q_l (W_K,l E_l) = (Q_l W_K,l) E_l`，避免创建整幅 32 通道 key；`eta_l Phi_l(sum_i w_i,l A_i,l V_i) = sum_i w_i,l A_i,l [eta_l Phi_l(V_i)]`，先在三个节点上执行原有无偏置 Phi_l 和 eta_l，再回写。注意力的 sqrt(32)、beta、sigmoid 和尺度权重位置不变，均值/归约定义不变。输出保持 channels_last_3d，不保存整幅未投影 G_l；原 NodeToSpace 默认接口及深层回写保持不变。输出及所有输入/参数梯度与原展开公式作容差对照，含 eta=0 和负值。

从 J3=F' 开始，逐级三线性插值到 E_l 的真实尺寸，1³ Conv→GN8→LeakyReLU 降至 C_l，与 E_enhanced_l 拼接；两次 3³ stride=1 pad=1 卷积单元完成 2C_l→C_l→C_l。最后 1³ 32→3 输出 Lf。

工程执行使用数学等价的顺序：仅将逐点 1³ 仿射 Conv 移至三线性插值之前，GN8/LeakyReLU 仍在插值之后。该投影与逐通道线性插值可交换（含 bias），使插值输出通道减半；浮点计算次序变化以输出与梯度容差验证，不保证逐位一致。参数、state_dict 键和配置不变，严格 resume 的源码身份校验不放宽。拼接后释放 enhanced 的局部引用，autograd 所需张量仍正常保存；没有 detach、checkpointing 或精度变化。卷积单元的 LeakyReLU(0.01) 仅原地复用 GN 的私有输出（GN 反向不依赖此输出），不覆盖原始 feature/skip/输入，保留标准 autograd 与正负区间导数。

## 联合监督（式 27–33，含层级一致性修订）

GT 不下采样。先把 coarse **logits** 三线性插值到完整原始 GT 网格，再 sigmoid。两分支各自计算：

```
BCE = -mean_(region,voxel)[T*log(S+eps)+(1-T)*log(1-S+eps)]
Dice_i = (2*sum(S_i*T_i)+eps)/(sum(S_i)+sum(T_i)+eps)
Lseg = BCE + 1-mean_i(Dice_i)
L = mean_batch(Lseg_final + lambda_c*Lseg_coarse)
```

三区域全部保留，病例内归约后取 batch mean。该概率 BCE 不能静默换成 BCEWithLogitsLoss、MONAI DiceCELoss、balanced CE 或类别加权。

层级一致性修订（新式 32–33）仅作用于最终概率 P^f（WT/TC/ET 依次为通道 1/2/3），coarse 仍只用 Lseg：

```
Lhier = (1/(2|Omega_I|)) * sum_x [ReLU(P_TC(x)-P_WT(x)) + ReLU(P_ET(x)-P_TC(x))]
L = mean_batch(Lseg_final + lambda_c*Lseg_coarse + lambda_hier*Lhier_final)
```

Omega_I 为输入体素网格（使用 spatial_crop 时即框内体素，与 BCE 的平均范围一致）。软约束只在训练中惩罚违例，推理仍独立阈值化、不强制包含。`loss.lambda_hier` 缺省（null）时目标函数与返回结构逐位等同旧式 27–32，历史 run 的严格 identity 不变；配置为数值（>=0）时启用式 33，日志与 TensorBoard 另记未乘权重的 Lhier（`hierarchy`）。文档规定 lambda_hier>=0 但未给数值，`development/year_fold_bbox_fp32.json` 的 1.0 是用户 2026-11-05 同意提交的开发取值（未调参）。

## 明示的开发配置与尚待冻结项

`configs/development/implementation_check.json` 是实现检查配置，**不是正式实验协议**。文档固定的编码器和解码器尺寸已完整实现；不是缩通道的 micro 替身。

尚未由修订文档给定的 deep Cr/K/da/Cg 暂沿用旧工程值 8/2/4/6，epsilon=1e-6，lambda_c=.5，gamma_init=.1；conv/coarse/deep fusion bias=true、align_corners=false、beta_init=0、tau=.5 均明确记录。标准层初始化保留 PyTorch 默认。模型参数量在这些显式设置下为 8,330,296。整脑外接框协议的配置（`configs/development/year_fold_bbox_fp32.json`、`configs/development/year_fold_bbox_conv_bf16.json`、`configs/development/complete_graph/*.json` 与 `configs/experiments/*.json`）改用 2026-11-06 用户批准的 Cr/K/da/Cg=64/2/32/32：修订文档只规定 K 为正整数、未给 Cr/da/Cg 数值；边门控 MLP 输入为 2C+7=519 维，Cr=64=C/4；三节点完全图一轮即可两两直接交互，K=2 为交互一次再修正一次；深层回写 da=Cg=32 与文档给定的解码器 d=Cg=32 对齐。参数量 8,387,120（+0.7%；嵌套双向链 8,486,001），新增计算只在节点与最深层网格。epsilon、lambda_c、gamma_init、lambda_hier=1.0、beta_init、tau 不变。其余配置保持 8/2/4/6 以维持历史 run 可追溯。

MRI 在原配准网格按各文件 scaling 读取；逐模态非零体素 z-score（std 下限 1e-8），原零值保持零；默认不空间裁剪，裁剪仅按上述 spatial_crop 显式开启。该强度选择、seed=20260925、AdamW 3e-4/weight_decay=0、batch=1、channels_last_3d 是开发选择，不声称来自论文或已是最优配置。没有 checkpointing 或 scheduler；默认没有 augmentation 和 AMP。整脑外接框协议配置显式开启 2026-11-06 用户批准的轻量训练增强 `training.augmentation=flip_intensity_v1`（`data/augmentation.py`）：仅训练病例，三个空间轴各以 0.5 概率翻转（图像与标签同步），每个模态的非零体素乘 U(0.9,1.1)、加 U(-0.1,0.1) 个 z-score 单位，零值保持零；dev/验证/推理从不增强。随机数由 (seed, global step) 派生，不消耗其他随机流，续训逐步复现，每步参数写入训练日志。增强作用于输入与标签，标签不进入模型或图像变换。不使用旋转、弹性形变或噪声。

可选卷积计算精度（2026-11-05 用户批准，显式 `training.convolution_compute_dtype`，缺省为 FP32 且计算与 identity 逐位不变）：只在 depth_sliced 的 3×3×3 卷积内部把每批输入平面与权重舍入为 bfloat16 做乘加，部分和、输出、输入/权重梯度均累加回 FP32；参数、优化器、1×1 投影、GroupNorm、激活、节点构建与关系推理、深层回写、解码器注意力与尺度权重、损失全部保持 FP32，方法公式不变。训练入口只接受 null/bfloat16：float16 在真实体积上每体素梯度下溢（BraTS2018 一例实测 27.6% 梯度为 0，全梯度相对误差 67%），需要未实现的损失缩放。同例 bf16 相对 FP32：损失 2.61243/2.61230，logits 相对 L2 误差 0.65%，全参数梯度 0.54%（余弦约 1.0）。7900 XTX 实测训练步 3.47 s 对 3.64 s（约快 5%）。对照臂为 `configs/development/year_fold_bbox_conv_bf16.json`；所有对比实验必须统一精度。

可选整网混合精度（2026-11-06 用户批准，显式 `training.mixed_precision=bfloat16`，与上一项二选一，仅 depth_sliced；缺省 null 时计算与 identity 不变）：编码器/解码器体数据以 bfloat16 存储，3×3×3 卷积、1×1 投影、激活、残差相加与解码器两次体积乘积用 bf16 计算；卷积每个深度块内各核平面的部分和在 FP32 中累加、每个输出只舍入一次，输入梯度按块在 FP32 中求和后写回，权重/偏置梯度累加为 FP32。保持 FP32 的是：参数与 AdamW、GroupNorm 的统计量与归一化算术（分块 FP32 var_mean 按 Chan 公式合并，不复制整卷）、粗预测头、Space-to-Node、关系轮次、深层回写与融合（最深层网格仅为体积的 1/512）、尺度权重、解码器打分偏置与 sigmoid 门控、交给损失的 logits 与损失本身。bf16 指数范围同 FP32，不需要损失缩放。BraTS2018 一例（136×160×136）同权重实测，相对 FP32：损失 2.61244/2.61230，最终 logits 相对 L2 误差 0.84%，全参数梯度 0.70%（余弦 0.99998；仅卷积 bf16 为 0.54%），峰值显存 3.82 对 7.10 GiB。正式训练入口同 30 例 30 步（npy 缓存、交错空洞执行）中位步时：FP32 1.56 s、仅卷积 bf16 1.21 s（1.29×）、整网 1.10 s（1.42×），峰值显存 10.10/10.10/5.43 GiB，30 步损失 1.4574/1.4578/1.4573；整网模式中断续训通过。采用交错空洞执行之前三者为 4.18/3.82/3.77 s：MIOpen 对空洞率 2 的 3×3 卷积回退的 naive 内核占主导，bf16 同样慢。对照臂为 `configs/development/complete_graph/main.json`。

第一档消融（2026-11-09 用户批准；显式 `model.variant`，缺省即完整文档方法，其构建顺序、初始化、参数与 identity 逐位不变，`to_dict` 不写出 variant）：`no_relation` 照常由粗掩码构建三节点（含 c/s/q）并用于深层回写与解码器，但不运行关系轮次，z(K):=z(0)，尺度描述中的入边门控因无边而固定为 0，参数 7,894,255；`drf_backbone` 保留双感受野编码器，解码器换为普通 U-Net 解码（三线性放大后与 skip 拼接、两个单元、1×1 头），无节点、回写、区域查询与尺度权重，参数 7,780,444；`unet` 在此基础上把每级 DRF 块换成一个普通 3×3×3 单元（stride-2 单元 + 单元，经典两卷积层级），无空洞卷积，参数 5,305,350。各变体都保留最深层 1×1 粗预测头：完整方法中它提供节点掩码；两个 plain 变体中只作辅助深监督、不接入其他模块。因此所有消融使用完全相同的目标（式 27–33，λc=0.5、λ_hier=1），只改结构；第四臂 `lambda_hier=0` 只去掉式 (33) 层级项（仍计算并记录）。配置为 `configs/development/complete_graph/{no_relation,drf_backbone,unet,no_hier}.json`，与主臂 `configs/development/complete_graph/main.json` 只差上述一项（测试逐项比对）；2026-11-10 起这一组由下述累加消融取代，其中只有 E1（main）与 A3（no_relation）实际运行。plain 变体中未用到的节点/关系字段保留原值仅为 identity 对齐。这些是开发对照，不是冻结的论文协议；同一组对比必须同精度、同折、同早停，并报告参数量。

累加消融（2026-11-10 老师批准，参照 DHAS，只在 BraTS 2018 五折上做，不跑基线）：`configs/experiments/` 下 R1 `r1_drf_backbone`（variant=drf_backbone，λ_hier=0，7,780,444 参数）→ R2 `r2_nodes`（variant=no_relation，λ_hier=0，7,894,255）→ R3 `r3_relation`（省略 variant，graph=nested_chain，λ_hier=0，8,486,001）→ R4 `r4_full`（λ_hier=1，即完整方法）；可选 `r4_k1`、`r4_k3`（R4 的 K=1、3）。相邻两行只差一项，R4 与 E1 的配置只差 graph（测试逐项比对）；所有新配置都写 graph=nested_chain，无关系模块的变体不使用它。λ_hier=0 时层级项仍计算并记录。相邻两行做配对 t 检验（检验单位待定，见 EXPERIMENTS.md）。

## 数据与实验协议边界

当前下载归档是 BraTS2023 GLI（1251 scans/1133 subject IDs），BraTS2024 GLI 主集（1350/613）与追加集（271/118）；后两者不是第三个年份，患者数不能直接相加视为独立人群。2024 为术后数据，含 RC 标签4；**不能把它当作旧 BraTS 的 ET=4 转成3**。当前三区域训练入口拒绝 2024；inspect 可列出其元数据。旧 2018–2021 术前编码只有显式声明对应 cohort 才允许 4→3。

当前输入可直接只读 ZIP 或已解压目录，不自动下载/解压。manifest 记录文件名与 ZIP size/CRC32 或目录 size/mtime；这不是解压影像的 SHA256。每个训练 run 保存完整 manifest/划分及哈希；划分须包含患者 ID，禁止同一患者不同时间点跨 train/dev。

所给 DHAS 论文实验分别使用 BraTS2018/2019/2020，各自五折训练与官方 validation 评估，并非三年混成一个数据集；其 128³ patch、batch=4、两阶段训练和后处理不能直接移植为本项目既定协议。当前实现内部逐病例三区域 DSC 监测（both-empty=null），不冒充官方 BraTS 评估；HD95、官方空肿瘤约定、正式 split/CV、后处理、训练周期须后续明确。

训练复用现有 Trainer 的原子 checkpoint、optimizer/RNG/病例顺序、ETA、JSONL、TensorBoard 与严格 resume。旧诊断入口保持 step cadence；独立年份折配置支持 epoch cadence。`development/year_fold_fp32.json` 的500轮上限、最少20轮、20轮patience、min_delta=.001是2026-11-02用户同意准备的首轮开发协议，不是最终论文选择。每轮全量internal dev，以final三区域mean-case hard Dice早停；原始最高分保存best-dev，早停计数和历史严格恢复。2018–2020支持只读nii/nii.gz及显式training子目录选择，五折由固定seed的患者随机分配产生，无分层，不混合年份。CPU 合成测试不能证明 24GB 上完整 MRI 可训练；下一步需用户授权真实 AMD preflight，OOM 停止，不自动改方法。

训练数据预读（2026-11-09 用户批准；`scripts/brats.py` 训练入口默认开启，runtime identity 记 `data_prefetch=next_case_thread_v1`，续跑不能悄悄切换）：GPU 计算当前病例时，后台单线程提前读出本轮的下一例；在 CUDA/ROCm 上放入锁页内存，并以非阻塞方式拷入显存。读取器仍按原样每步创建一次，读取器随机数对每个已提交样本恰好消耗一次；送入模型的数值、增强、损失、优化与 checkpoint 均与不预读时逐位相同。checkpoint 不含预读状态，续跑后的第一步照常现读。每轮第一例（顺序在该步才抽取）和验证第一例照常现读。只允许对 BraTSDataset 或显式声明 `prefetch_safe` 的数据集开启：读取必须是纯函数、不消耗随机数。只用标准线程与 PyTorch 锁页内存，CPU/ROCm/CUDA 与 Linux/Windows 通用。CPU 训练场景中新旧代码（预读开/关、一口气跑完/中途续跑）日志与 checkpoint 逐值一致。模型内部的有限值检查会在前向中同步 CPU 与 GPU，因此只预读数据、不去掉每步首尾的同步。
