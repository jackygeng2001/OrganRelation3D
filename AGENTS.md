# 项目开发规范

所有任务先读本文件和 METHOD_SPEC.md。2026-10-31 用户明确重启研究方向：当前依据《脑肿瘤分割方法_修订版.docx》及 2026-11-05 的《层级一致性修订版》（仅新增层级一致性损失），旧腹部 CT 方法保留为历史代码与回归，不再支配新方法。GitHub 仓库/发行名 OrganRelation3D，Python 包 organ_relation，不虚构论文方法名。沿用当前 src/configs/scripts/tests/environments 骨架，不再建立嵌套“代码”目录。

## 方法与范围

- 当前完整四模态 MRI，顺序 T1/T1ce/T2/FLAIR。一次整例（整脑）前向，无 patch/sliding-window 或标签引导采样。原始数据只读。唯一允许的空间裁剪是 2026-11-05 用户批准的显式 `preprocessing.spatial_crop`：仅由四模态非零体素决定的整脑外接框，标签只用于拒绝会丢失肿瘤体素的裁剪；默认 null 保持原网格。
- WT/TC/ET 三个重叠区域，独立 sigmoid 与阈值；三个固定节点、六条有向边、无 self-loop。GT 只用于监督与评估，绝不进入模型或存在性判断。
- 严格按 METHOD_SPEC 的双感受野编码器、关系公式、深层回写、关系尺度自适应解码器、BCE+Dice 联合损失实现；层级一致性损失仅加在最终预测上、权重由配置 lambda_hier 显式给出。c/s/q 固定但不 detach；各轮共享参数、同步更新、重算边权。解码使用 z(K) 与缓存 alpha(K-1)，不再重算一轮关系。
- 新方法无需 MONAI backbone；旧 AMOS/MONAI 配置与接口保持可追溯。不得把旧 15 节点、16 类 CE、CT HU、160/40 split 套到 MRI。
- 方法文档未确定的实验选择必须显式配置并标为待冻结，不声称正式最优。当前 brats_regions.json 仅实现检查，训练周期与 split 必须显式提供。
- 2024 GLI 术后 RC 不能当作旧 BraTS ET=4；未确定标签/实验协议前只允许元数据检查。不同年份/任务不自动混合；同一患者多时间点不得跨 train/dev。
- 修改 AGENTS 或 METHOD_SPEC 不能替代用户对科学变更的授权。遇到文档缺失定义先说明，沿用值必须可审计。

## 设备与环境

- AMD 主实验环境：Ubuntu 24.04，RX 7900 XTX 24GB，现有 organrelation3d / PyTorch 2.11.0+ROCm7.2。Windows CPU 开发与 NVIDIA RTX5060 8GB/PyTorch2.8+CUDA12.8 为辅助兼容环境。
- 本轮仅代码修改、CPU 合成测试与只读数据/文献检查。GPU preflight、短训练、正式实验须用户明确授权；不得自行升级/降级核心环境。
- 标准 PyTorch 算子，兼容 CPU/ROCm/CUDA，不引入 NVIDIA 专属扩展。设备由入口管理，内部临时张量继承输入 device/dtype。
- 保留有限值、autocast 与形状检查；性能同步问题待 AMD profile 后处理，不为速度擅自删除。不能用 CPU 小输入通过来宣称 full-volume 显存可行。
- channels_last_3d 为显式运行设置；ROCm 启动前检查 PYTORCH_MIOPEN_SUGGEST_NHWC=1，不偷偷设置或回退。不保证每个中间算子都保持该布局。
- 显存不足先定位并报告，不自动缩 channels/spacing、AMP、activation checkpointing、patch。允许的降精度只有两种，均须在配置中显式开启且二选一：2026-11-05 用户批准的 `training.convolution_compute_dtype=bfloat16`（仅 depth_sliced 3×3×3 卷积乘加），与 2026-11-06 用户批准的 `training.mixed_precision=bfloat16`（整网 bf16 体数据，敏感部分 FP32，见 METHOD_SPEC）；同一组对比实验必须统一精度。大型依赖/数据下载由用户手动执行。

## 工程与复现

- 源码按 models/data/training 职责放置，损失仍在 losses.py；复用现有 Trainer，不增 registry/manager/service 框架。各 __init__.py 保持轻量。
- 默认旧接口与数值行为保持；旧实验仅在严格匹配的历史版本恢复。历史工程 migration 白名单保持原审计约束，不因新方法而扩宽。
- 原子 checkpoint 保存完整 model/optimizer/RNG/sampler/order/日志提交位置；严格验证 config/source/data/split/environment。不能通过宽松 resume 偷换模型或科学条件。
- JSONL/checkpoint 为事实来源；TensorBoard/console 仅 observer，失败不改变训练轨迹，恢复去除未提交尾部。三区域日志不能误标为15器官或 CE。
- 数据集返回 image=[4,D,H,W]、label=[D,H,W]；入口显式 batch=1。不原地写数据/标签，不把 MRI 强度按 HU 处理。
- CPU 公式、梯度、全链路、恢复测试先行；scripts/run_tests.py --suite all 不允许跳过冒充通过。依赖在 environments 分设备管理，不让 editable install 自动切换 PyTorch 后端。

## Git 与交付

- 新 commit 使用 Conventional Commits，按职责选 feat/fix/refactor/test/docs/chore/perf。不 rebase/amend/squash 旧历史或 force push。
- 用户本轮已明确授权 commit + push origin/main；先审 diff 和测试，提交后确认 local/remote HEAD 一致、工作区 clean。
- 不提交原始 MRI/AMOS、标签、权重、checkpoint、runs、缓存或临时输出。不要修改/删除历史 runs。
- 只维护长期规范/README/config/tests，不为阶段结果创建 report/review markdown。结果直接在对话报告：改动、实际设备/测试、未验证项、Git SHA/push。
