# 项目开发规范

所有任务先读本文件、METHOD_SPEC.md 和 EXPERIMENTS.md。方法依据为《脑肿瘤分割方法_修订版.docx》及 2026-11-05 的《层级一致性修订版》（仅新增层级一致性损失）。早期腹部 CT 阶段的代码、配置与文档已于 2026-11-09 从工作树删除，只在 git 历史中保留，不再恢复或参考。GitHub 仓库/发行名 OrganRelation3D，Python 包 organ_relation，不虚构论文方法名。

## 语言与文档约定

- 代码注释与 docstring 用中文；已有的英文注释不必翻译，改动时再用中文重写。
- 文件名、目录名、提交信息（Conventional Commits）、配置键、命令行参数用英文；命令行输出保持英文。
- README.md 面向第一次接触本项目的人，将来做中英双语，目前只有中文；不写局限、不放项目树。
- EXPERIMENTS.md 是唯一的实验报告及日志（中文，给人和 AI 看）：实验配置、进度、结果与决定都写在这里，每有新结果就更新；项目结档时删除。不另建阶段性 report/review markdown。
- METHOD_SPEC.md 记录逐公式实现与工程等价变换；AGENTS.md 记录长期规范。

## 方法与范围

- 完整四模态 MRI，顺序 T1/T1ce/T2/FLAIR。一次整例（整脑）前向，无 patch/sliding-window 或标签引导采样。原始数据只读。唯一允许的空间裁剪是 2026-11-05 用户批准的显式 `preprocessing.spatial_crop`：仅由四模态非零体素决定的整脑外接框，标签只用于拒绝会丢失肿瘤体素的裁剪。
- WT/TC/ET 三个重叠区域，独立 sigmoid 与阈值；三个固定节点，无 self-loop。关系图为 2026-11-10 老师批准的嵌套双向链 WT⇄TC⇄ET：四条有向边，WT 与 ET 之间无边；向内边（WT→TC、TC→ET）表示包含关系，用外层约束内层范围，向外边（ET→TC、TC→WT）用内层帮助外层定位；两类边的门控 MLP 与消息矩阵分开设参，GRU 共用。配置显式写 `model.graph=nested_chain`；省略 graph 即历史的三节点完全图（六条有向边，E1/A3 所用），identity 不变。GT 只用于监督与评估，绝不进入模型或存在性判断。
- 关系推理 v2 与协议 P2（2026-11-11 用户同意，见 METHOD_SPEC 与 EXPERIMENTS.md 5.5–5.7）：`model.relation_conditioning`、`model.containment`、`model.node_inputs`、`model.output_hierarchy`、`loss.lambda_contain`、`training.scheduler` 都是显式配置，省略即历史行为、identity 不变；正式配置在 `configs/relation_v2/`，队列在 `configs/queues/`。真值只进入包含监督的损失，不进入模型。
- 严格按 METHOD_SPEC 实现双感受野编码器、关系公式、深层回写、关系尺度自适应解码器、BCE+Dice 联合损失；层级一致性损失仅加在最终预测上，权重由 `loss.lambda_hier` 显式给出。c/s/q 固定但不 detach；各轮共享参数、同步更新、重算边权。解码使用 z(K) 与缓存的 alpha(K-1)，不再重算一轮关系。
- 消融只通过显式 `model.variant`（no_relation/drf_backbone/unet，2026-11-09 用户批准）或配置项实现；省略 variant 与 graph 时构建与 identity 和历史 run 完全一致。当前消融（2026-11-10 老师批准，参照 DHAS）只在 BraTS 2018 五折上做、不跑基线，按 R1 双感受野编码器 + 普通解码器 → R2 加区域节点、不做关系推理 → R3 加嵌套关系推理 → R4 加层级一致性损失（完整方法）逐项累加，相邻两行只差一项（由测试核对），相邻两行做配对 t 检验；时间允许再跑 K=1、K=3 两组。
- 方法文档未确定的实验选择必须显式配置并标为待冻结，不声称正式最优。`configs/development/implementation_check.json` 仅用于实现检查。
- 数据集（2026-11-10 老师批准）：BraTS 2018（消融）、2023 GLI（验证集结果提交到官方持续评测）、2024 GLI。2023/2024 的训练与提交协议在 EXPERIMENTS.md 中确定前，只做元数据与缓存检查。2024 GLI 术后 RC 标签不能当作旧 BraTS 的 ET=4；不同年份/任务不自动混合；同一患者多时间点不得跨 train/dev。
- 修改 AGENTS 或 METHOD_SPEC 不能替代用户对科学变更的授权。文档缺失定义时先说明，沿用值必须可审计。

## 设备与环境

- AMD 主实验环境：Ubuntu 24.04，RX 7900 XTX 24GB，conda 环境 organrelation3d（Python 3.11 / PyTorch 2.11.0+ROCm7.2）。Windows 11 + RTX 5060 8GB（PyTorch 2.8 + CUDA 12.8）为辅助环境，正式训练前须先完成验证（见 EXPERIMENTS.md）。不得自行升级/降级核心环境。
- GPU 预检、短训练与正式实验须用户明确授权；CPU 合成测试、只读数据检查可直接进行。
- RX 7900 XTX 上同时跑两个训练（2026-11-11 实测总吞吐约 1.45–1.48 倍，合计保留显存约 17 GiB，见 EXPERIMENTS.md 5.7）：成批实验用 `scripts/run_queue.py <队列文件>` 两个并行执行，中断后同一条命令续跑；三个并行放不下。RTX 5060 一次一个。
- 只用标准 PyTorch 算子，兼容 CPU/ROCm/CUDA，不引入 NVIDIA 专属扩展。设备由入口管理，内部临时张量继承输入 device/dtype。
- 保留有限值、autocast 与形状检查；不能用 CPU 小输入通过来宣称整例显存可行。
- channels_last_3d 为显式运行设置；ROCm 启动前须有 `PYTORCH_MIOPEN_SUGGEST_NHWC=1`，只能由本机配置或命令显式给出，不偷偷设置或回退。
- 显存不足先定位并报告，不自动缩通道/spacing、开 AMP、activation checkpointing 或改 patch。允许的降精度只有两种，须在配置中显式开启且二选一：`training.convolution_compute_dtype=bfloat16`（2026-11-05 批准）与 `training.mixed_precision=bfloat16`（2026-11-06 批准，见 METHOD_SPEC）；同一组对比实验必须统一精度。大型依赖/数据下载由用户手动执行。

## 工程与复现

- 源码按 models/data/training 职责放置，损失在 losses.py；复用现有 Trainer，不增 registry/manager/service 框架。各 `__init__.py` 保持轻量。
- 配置：`configs/experiments/` 为正式五折实验（主方法与消融），`configs/development/` 为开发与历史配置（旧全连接图阶段的正式配置在 `configs/development/complete_graph/`）；本机路径与设备写在不进 git 的 `configs/machine.local.json`（模板 `configs/machine.example.json`）。五折统一用 `scripts/run_experiment.py <配置>` 运行，中断后同一命令续跑。
- 原子 checkpoint 保存完整 model/optimizer/RNG/sampler/order/日志提交位置；续跑严格验证 config/source/data/split/environment，不能通过宽松 resume 偷换模型或科学条件。实验进行期间，不得改动该实验所用目录中的代码。
- `src/organ_relation/data/brats.py` 的 SHA256 是 npy 缓存键的一部分，改动它会使全部缓存失效；确需修改时先告知用户。
- JSONL/checkpoint 为事实来源；TensorBoard/console 只是观察者，失败不改变训练轨迹，续跑时去除未提交的日志尾部。日志字段沿用现有格式，保证新旧 run 可直接对比。
- 数据集返回 image=[4,D,H,W]、label=[D,H,W]；入口显式 batch=1。不原地写数据/标签，不把 MRI 强度按 HU 处理。
- 改动训练相关代码时，用 CPU 训练场景对比新旧代码的日志与 checkpoint；`scripts/run_tests.py --suite all` 不允许跳过冒充通过。测试公用工具在 `tests/support.py`。

## Git 与交付

- 默认分支为 `dev`；`main` 是早期状态的历史分支。新工作在功能分支或独立 worktree 中进行；合并到 `dev` 或改动 `main` 需用户明确同意。
- 提交使用 Conventional Commits，按职责选 feat/fix/refactor/test/docs/chore/perf。不 rebase/amend/squash 旧历史，不 force push。提交前审 diff 并跑测试，推送后确认本地/远端 HEAD 一致、工作区干净。
- 不提交原始 MRI、标签、权重、checkpoint、runs、缓存、本机配置或临时输出。不要修改/删除历史 runs。
- 结果在对话中报告：改动、实际设备/测试、未验证项、Git SHA/推送情况；同时更新 EXPERIMENTS.md。
