# 脑肿瘤动态区域关系分割：当前工程规范

2026-10-31 起以用户提供的《脑肿瘤分割方法_修订版.docx》为方法依据。仓库与包名保留 OrganRelation3D / organ_relation；不提前命名论文方法。历史腹部 CT 规范移至 [legacy_amos_method.md](docs/legacy_amos_method.md)，其 15 节点、MONAI、AMOS split 与训练配置不用于新方案。

方法文档 SHA256：`76cf95d033b0026539cf38c1d3f07240223928dde854d86c48ee2ae676258a0b`。
实验参考 PDF《Deep Hierarchy-Aware Segmentation: A Novel Framework for MRIs Brain Tumor Segmentation》SHA256：`43dd46d0bff87a399884fa3773fc8d153b88b7e1a2d0dfbd5a87fa573fb88065`。参考论文不是新方法的结构来源；若用户另有第二篇实验参考，须再核实。

## 输入、区域与上下文

- 完整配准 MRI，顺序 T1、T1ce、T2、FLAIR；image `[B,4,D,H,W]`，FP32。无 crop、patch、滑窗或标签引导空间处理。
- 原标注统一到 0/1/2/3：背景、坏死/非增强核心、瘤周水肿、增强肿瘤。三重叠区域顺序 WT={1,2,3}、TC={1,3}、ET={3}。
- coarse/final 均为三个独立 sigmoid 通道。背景不单列通道/节点。始终三个节点，GT 只进入损失与指标。
- 推理每通道独立以 `P>=tau` 二值化；不做 argmax，不额外强制 ET⊆TC⊆WT。监测可以报告层级违反数。
- 奇数尺寸直接支持：每层 stride-2 卷积的实际形状为 ceil(size/2)，解码插值到实际 skip 尺寸，最终严格恢复输入网格。无需为了整除引入 padding。

## 编码器（式 1–5）

四层通道固定 32/64/128/256。所有编码/解码卷积单元为 Conv3d→GN(8 groups)→LeakyReLU(0.01)。

1. E0：两次 3³、stride=1、padding=1 卷积，4→32→32。
2. 每层 l=1,2,3 先 3³ stride=2 padding=1 下采样，得到 X_l。
3. 对同一 X_l 独立计算 local 3³ dilation=1 pad=1 与 context 3³ dilation=2 pad=2，两路都有 GN/激活。
4. 拼接 `[local;context;abs(local-context)]`，1³ 卷积 3C_l→C_l/4→2，中间 LeakyReLU；逐位置沿两分支 softmax。
5. `V=w_local*local+w_context*context`；`E_l=X_l+Pi_l(V)`。Pi 为通道不变的 1³ 卷积，不额外加 GN/激活。

F=E3，C=256；原始 E2/E1/E0 进入关系解码器。

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

三节点、六条有向边、无自环；张量 alpha 的轴是 `[batch,sender,receiver]`。

```
r_ij = [z_i;z_j;c_j-c_i;s_i;s_j;q_i;q_j]   # 2C+7
alpha_ij = sigmoid(w2^T ReLU(W1 r_ij+b1)+b2)
m_j = sum_(i!=j) alpha_ij Wm z_i
z_new_j = GRU(m_j,z_j)
```

每轮从同一旧 z 同步计算全部消息后更新，所有轮次共享参数。c/s/q/P 保持不变但不 detach。最后输出 z(K) 与实际末轮 `alpha(K-1)`；不额外基于 z(K) 重算边权。

修订文档只写标准 GRU，未展开。当前工程沿用已测试的 reset-before-linear 版本（仍需在实验定义中明确）：

```
u = sigmoid(Wu m + Uu z + bu)
rho = sigmoid(Wrho m + Urho z + brho)
z_tilde = tanh(Wz m + Uz(rho*z) + bz)
z_new = (1-u)*z + u*z_tilde
```

W 各有一个偏置，U 无偏置，Wm 无偏置。

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

从 J3=F' 开始，逐级三线性插值到 E_l 的真实尺寸，1³ Conv→GN8→LeakyReLU 降至 C_l，与 E_enhanced_l 拼接；两次 3³ stride=1 pad=1 卷积单元完成 2C_l→C_l→C_l。最后 1³ 32→3 输出 Lf。

工程执行使用数学等价的顺序：仅将逐点 1³ 仿射 Conv 移至三线性插值之前，GN8/LeakyReLU 仍在插值之后。该投影与逐通道线性插值可交换（含 bias），使插值输出通道减半；浮点计算次序变化以输出与梯度容差验证，不保证逐位一致。参数、state_dict 键和配置不变，严格 resume 的源码身份校验不放宽。拼接后释放 enhanced 的局部引用，autograd 所需张量仍正常保存；没有 detach、原地覆盖、checkpointing 或精度变化。

## 联合监督（式 27–32）

GT 不下采样。先把 coarse **logits** 三线性插值到完整原始 GT 网格，再 sigmoid。两分支各自计算：

```
BCE = -mean_(region,voxel)[T*log(S+eps)+(1-T)*log(1-S+eps)]
Dice_i = (2*sum(S_i*T_i)+eps)/(sum(S_i)+sum(T_i)+eps)
Lseg = BCE + 1-mean_i(Dice_i)
L = mean_batch(Lseg_final + lambda_c*Lseg_coarse)
```

三区域全部保留，病例内归约后取 batch mean。该概率 BCE 不能静默换成 BCEWithLogitsLoss、MONAI DiceCELoss、balanced CE 或类别加权。

## 明示的开发配置与尚待冻结项

`configs/brats_regions.json` 是实现检查配置，**不是正式实验协议**。文档固定的编码器和解码器尺寸已完整实现；不是缩通道的 micro 替身。

尚未由修订文档给定的 deep Cr/K/da/Cg 暂沿用旧工程值 8/2/4/6，epsilon=1e-6，lambda_c=.5，gamma_init=.1；conv/coarse/deep fusion bias=true、align_corners=false、beta_init=0、tau=.5 均明确记录。标准层初始化保留 PyTorch 默认。模型参数量在这些显式设置下为 8,330,296。

MRI 在原配准网格按各文件 scaling 读取；逐模态非零体素 z-score（std 下限 1e-8），原零值保持零，不空间裁剪。该强度选择、seed=20260925、AdamW 3e-4/weight_decay=0、batch=1、channels_last_3d 是开发选择，不声称来自论文或已是最优配置。没有 augmentation、AMP、checkpointing 或 scheduler。

## 数据与实验协议边界

当前下载归档是 BraTS2023 GLI（1251 scans/1133 subject IDs），BraTS2024 GLI 主集（1350/613）与追加集（271/118）；后两者不是第三个年份，患者数不能直接相加视为独立人群。2024 为术后数据，含 RC 标签4；**不能把它当作旧 BraTS 的 ET=4 转成3**。当前三区域训练入口拒绝 2024；inspect 可列出其元数据。旧 2018–2021 术前编码只有显式声明对应 cohort 才允许 4→3。

当前输入可直接只读 ZIP 或已解压目录，不自动下载/解压。manifest 记录文件名与 ZIP size/CRC32 或目录 size/mtime；这不是解压影像的 SHA256。每个训练 run 保存完整 manifest/划分及哈希，不复用 AMOS split；划分须包含患者 ID，禁止同一患者不同时间点跨 train/dev。

所给 DHAS 论文实验分别使用 BraTS2018/2019/2020，各自五折训练与官方 validation 评估，并非三年混成一个数据集；其 128³ patch、batch=4、两阶段训练和后处理不能直接移植为本项目既定协议。当前仅实现内部逐病例三区域 DSC 监测（both-empty=null），不冒充官方 BraTS 评估；HD95、官方空肿瘤约定、正式 split/CV、后处理、训练周期须后续明确。

训练复用现有 Trainer 的原子 checkpoint、optimizer/RNG/病例顺序、ETA、JSONL、TensorBoard 与严格 resume。当前新入口为 step cadence；不挪用历史 AMOS 的 160/40 或 500-epoch 协议。CPU 合成测试不能证明 24GB 上完整 MRI 可训练；下一步需用户授权真实 AMD preflight，OOM 停止，不自动改方法。
