# 项目开发规范

本目录为独立项目；GitHub 仓库/发行名称为 OrganRelation3D，Python package 保持 organ_relation。论文方法名尚未确定，不虚构或提前绑定；本地目录名不影响包名。代码项目均放在项目文件夹的“代码”目录内。所有任务首先读取本文件和 METHOD_SPEC.md。

## 方法与任务边界

- 方法主要依据《腹部多器官分割_方法章节.docx》，《细化方法.docx》补充公式；工程转录见 METHOD_SPEC.md。原 AGNN 仅作参考，不能修改本方法。
- 正式输入保留完整 CT 扫描范围；一次前向构建 15 个节点并做全局关系推理。训练与推理使用一致的完整扫描上下文。禁止用局部 patch 独立建图或滑窗替代。
- 背景参与 16 类预测但不建节点；15 个节点始终保留。真实标签仅用于监督损失和评估，不用于节点、关系、存在性判断、屏蔽或标签引导采样/裁剪。
- 各轮只更新语义状态；粗概率、质心、大小和置信度在一次前向内保持不变但不 detach。同步更新，各轮重算关系，参数共享。
- 严格保留指定 GRU、独立 sigmoid 边权、入边求和、节点内容回写、残差融合和联合损失，不以现成图网络替代。
- 输入分辨率、尺寸、padding 语义、预处理数值、骨干具体配置及评估协议尚未冻结。不得把草案/测试配置当作正式方案。
- 显存不足先定位瓶颈、优化标准 PyTorch 实现，再提出备选方案；不自行改变方法。
- 原始数据只读。不得安装或改动 GPU 环境、训练、发布或扩展任务范围，除非当前任务明确授权。

## 三台设备

1. Windows 核显笔记本：主开发、文档、Git、CPU 合成张量/公式/形状/边界/梯度测试。
2. AMD 主实验工作站：Ubuntu 24.04.4 LTS；RX 7900 XTX 24GB；PyTorch 2.11.0 + ROCm 7.2。负责完整模型 GPU 测试、显存、真实数据、小样本过拟合、正式训练和评估。模型和正式配置以此为目标。
3. NVIDIA 辅助工作站：Windows 11；RTX 5060 8GB；PyTorch 2.8.0 + CUDA 12.8。仅 CUDA 兼容、独立模块和缩小输入测试，不承担正式训练，不得据其容量改变正式模型。

上述 GPU 版本是用户指定环境，不表示本项目已经验证。AMD 验证由开发端提供代码和命令，用户通过 GitHub 同步、手动运行并返回结果；不配置远程开发环境。

## 跨平台实现

- 核心兼容 CPU、PyTorch ROCm 和 CUDA，优先标准 PyTorch 算子，禁止引入 AMD 不支持的 NVIDIA 专属扩展。
- 入口统一管理 device；内部不得硬编码设备编号、GPU 或绝对数据路径。临时张量继承输入设备和适当 dtype。
- 不同设备可使用不同 PyTorch 版本，使用版本特有 API 时必须有兼容处理和测试。
- 保留当前模块的 finite/autocast 等运行时正确性检查。后续 AMD 性能测试须审查这些检查（尤其 finite 归约后的主机判断）可能造成的 GPU 同步开销；不得在公式验证阶段为性能擅自删除。
- 环境说明分别维护，不能用辅助 CUDA 安装方案覆盖主 ROCm 环境。
- 需要下载较大文件时，先暂停对应下载并提供明确命令，由用户手动完成；继续不依赖该下载的工作。不得自行启动新的大型依赖或数据下载。
- pathlib 等跨平台路径工具；数据、缓存、权重和输出由入口配置。当前元数据统计仅依赖 Python 标准库。

## Git 与复现

- GitHub 用于阶段同步源代码、配置、方法文档、测试。未经明确发布任务不上传远程。
- 后续新提交使用 Conventional Commits，按职责选择 feat/fix/refactor/test/docs/chore/perf；不为统一格式重写既有历史。
- 不提交 AMOS 原始数据、患者影像、标签体积、权重、缓存、大型输出。病例级本地报告默认忽略；需要分享时先确认范围。
- 正式实验记录 Git commit、完整配置、数据划分及哈希、环境、随机种子、评估结果。
- 数据阶段、完整 Segmentor 与独立 JointLoss 已通过阶段验收。工程整理不得改变已验收的公式、数值行为、配置值或测试断言。当前 baseline 为 epsilon=1e-6、lambda_c=0.5、coarse align_corners=False，继续显式传参。已实现统一可恢复训练入口；本轮仅授权 CPU 合成测试及提供 AMD 命令，未授权运行 GPU 或正式训练。无 augmentation、scheduler、AMP、activation checkpointing 或 gradient accumulation。
- 源码按职责归入 models/ 与 data/，单个损失保留 losses.py，复现工具为 provenance.py。模型不依赖数据工具或 loss；各包 __init__.py 保持轻量。无实际代码时不建立空目录，不增加泛化框架。
- 原有 190 项回归由 scripts/run_tests.py 按 tensor/data 两组执行；新增 fullscan 组验证真实文件格式的合成 NIfTI 与完整小网络链路，任何跳过不算验收通过。依赖由 environments/ 分设备管理；本包 editable install 不自动选择或替换 PyTorch 后端。
- smoke、单/双病例 overfit、pilot、正式训练共用 training/；batch=1、num_workers=0。完整 optimizer step 边界原子保存全部恢复状态；严格校验配置、split/manifest、来源指纹，不静默兼容不同实验。病例级评估 ledger 与训练 checkpoint 分离。标量日志提交位置随 checkpoint 保存，恢复时截断未提交尾部。
- 固定 160/40 development split 用于选择模型；冻结后完整 200 official training CT 从头训练。官方 validation 必须先核实来源，不进入训练或反复调参；hidden test 仅影像，计划官方 DSC/NSD 评价。阶段验收配置 3S-P1/B 不是最终模型或 spacing。
- 训练/验证计时窗口独立，ETA 先 warm up；诊断和 checkpoint 频率必须可配置。CPU 恢复测试严格对照轨迹；未冻结 GPU 确定性时不得要求 AMD 逐位相同。
- Console/TensorBoard 仅为 observer，不能改变训练、RNG、采样、ETA、指标及 checkpoint 语义。JSONL/checkpoint 是事实来源；TensorBoard 恢复须清除未提交事件（包括同一步未提交的验证），写入失败明确停用 observer 而不误报训练失败。只记录 scalar，不默认写体积、histogram 或 embedding。

## 验收和交付

- 用户唯一授权的当前模型变体：正式 C-gated 显式启用单个可学习 relation residual scale，gamma_init=0.1，`F + gamma * phi(G)`；不得写成 `phi(gamma*G)`。旧配置无缩放路径和 B-spacing C 保留，A 无 gamma；当前正式配置用 `train_monai_relation_gated_A_160_40.json`。不限制 gamma 符号、不增加 schedule/正则。其他关系、固定属性、loss、K 和训练协议不变。诊断仅观察同次前向，JSONL 保留完整 scalar，TensorBoard 额外仅 Gamma 与实际扰动范数比。AMD 最大病例 preflight 由用户执行，通过后才由用户启动长训练。

- 当前 full-development A/C-gated 协议以 `train_monai_reference_A_160_40.json` / `train_monai_relation_gated_A_160_40.json` 为准：固定已有 160/40 split，严格校验 JSON 内 split_hash `7d308eca4f7324f0e899c7416a45a03f8dfbec5e867e5ed6018353e96541468c`，不得重新生成或用文件 checksum 替代。两者同 seed=20260925、A spacing、相同骨干/最终 MONAI DiceCELoss/AdamW 固定 LR；最多 500 epochs，最少 100。每 5 optimizer steps 原子保存 last，epoch 末仍补存；每 epoch 汇总，每 5 epochs 全量 40-case dev；final mean-case hard Dice 使用独立 patience reference，仅 current >= reference+1e-4 时更新 reference 并清零计数；连续小幅上涨可累计达到阈值。连续 5 次未达到阈值且 epoch>=100 时 early stop，gamma 不参与。raw best/best-dev 在任何严格新高时更新，与 reference 分离；两者均保存恢复。checkpoint/resume 必须保留并校验完整 early-stopping state/history，不能用 extension 绕过停止。历史 v1/single-case 配置及原 JointLoss 保留。
- 当前正式训练/preflight 只用 MONAI 标量 optimization objective，昂贵 per-class soft Dice、CE 分解和概率诊断留到 monitor/validation；不为每步 JSONL 重复创建全尺寸 16-class observer 张量。TensorBoard 按 epoch 汇总（gamma 为 epoch 末更新后值，实际扰动比为病例均值）。正式 console 另每 5 steps 输出当前/最近 loss、同次前向 gamma/实际扰动比、病例位置、LR、显存与 ETA；不增加 TensorBoard 指标或冗余 JSONL summary。只授权 CPU 测试；AMD 最大病例 preflight、正常病例顺序的独立 5-step 短训练及正式训练由用户手动执行，OOM 不自动改变模型或引入内存技术。

- CPU 公式与梯度测试先行；阶段同步后在 AMD 验证 GPU、显存、梯度与真实数据。
- 重要集成节点做 NVIDIA CUDA 兼容性测试，不要求每次修改三机全测。
- 核心模块、完整集成和 AMD 短程运行通过后，才启动正式实验。
- 尚无小器官保真度阈值；先报告候选的几何损失、图像质量风险和显存，由用户共同决定正式输入。
- 每轮结束报告修改文件、测试设备、实际通过的测试、未验证事项及 Git 版本。
- 关键实现、公式对应、测试结果、风险和待决事项直接在对话中完整汇报，方便复制审阅。只维护确有长期价值的规范、README、配置、测试和核心复现记录；不为每个小阶段新增 validation/review/report 文档。

- FullScanDataset 返回单样本 image=[1,D,H,W]、label=[D,H,W]，不负责 batching；DataLoader(batch_size=1) 默认拼批，单病例 probe 则显式增加 batch 轴，不添加输入 padding。当前 cardinal-mm 几何、每文件 scaling 和完整覆盖重采样已验收为 full-scan feasibility v1。正式模型容量仍未冻结；GPU 验证不得静默使用微型模型。正式强度方案与最终 spacing 仍待选择。
