# OrganRelation3D

当前方向：完整四模态 MRI 的脑肿瘤 WT/TC/ET 多标签分割，依据《脑肿瘤分割方法_修订版.docx》。仓库和 Python 包名 `organ_relation` 保留。

新的双感受野编码器 → 三节点动态关系 → 深层残差回写 → 关系尺度自适应解码器。三个区域独立 sigmoid，无标签参与建图，完整 volume 一次前向。**当前完成代码与 CPU 验证，尚未验证完整 MRI 在 AMD 24GB 上的显存或训练效果。**

- [当前方法、公式及待冻结选项](METHOD_SPEC.md)
- [开发约束](AGENTS.md)
- [历史 AMOS 使用说明](docs/legacy_amos_usage.md) / [历史方法](docs/legacy_amos_method.md)

## 代码入口

| 路径 | 职责 |
|---|---|
| `src/organ_relation/models/brain_tumor.py` | 新编码器、完整模型、关系尺度解码器 |
| `src/organ_relation/models/{coarse_head,space_to_node,dynamic_relation,node_to_space,residual_fusion}.py` | 复用的关系模块，旧默认保留 |
| `src/organ_relation/losses.py` | 新 RegionJointLoss 与历史损失隔离 |
| `src/organ_relation/data/brats.py` | 四模态完整 MRI、显式标签语义、患者分组检查 |
| `src/organ_relation/training/brain_tumor.py` | 三重叠区域监测与 TensorBoard 标量 |
| `scripts/brats.py` | inspect / 无更新 check / 受控 train |
| `configs/brats_regions.json` | 显式开发配置，不是正式实验方案 |

新方法固定编码器 32/64/128/256、GN8/LeakyReLU；配置下共 8,330,296 参数。深层 Cr/K/da/Cg=8/2/4/6 等未写死在方法中的选择暂沿用工程值，详见 METHOD_SPEC。没有替换为 micro 模型。

## 环境与 CPU 验证

使用已有 `organrelation3d` 环境，不安装或替换 PyTorch/ROCm/NumPy。新模型依赖现有 torch/NumPy/NiBabel；TensorBoard/tqdm 用于原训练引擎。旧 MONAI 回归仍依赖既有 MONAI。跨设备说明见 [environments](environments/amd-rocm.md)。

```bash
conda activate organrelation3d
python scripts/run_tests.py --suite all
python scripts/brats.py check --device cpu
```

`check` 默认小尺寸合成输入，执行完整实际架构的 forward/loss/backward，输出 shape、参数量、耗时和梯度状态，不创建 run、不 optimizer.step。CPU 测试不是 GPU 显存验收。

## 先确认数据来源

当前下载的三个 ZIP 是 **2023 GLI + 2024 GLI 主集/追加集**，不是三年独立任务。2024 是术后任务、含 RC，当前三区域代码拒绝用其训练。不能将 RC=4 错映为旧术前 ET=4。所给实验论文分别评估 2018/2019/2020，不支持直接把当前三个压缩包混合。

只读 ZIP 中央目录即可盘点，不展开体素、不解压原数据：

```bash
python scripts/brats.py inspect --cohort brats2023_gli \
  --source "$HOME/下载/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip"
```

完整 MRI loader 保留原配准网格，按每文件 scaling 读取，T1/T1ce/T2/FLAIR 逐模态非零 z-score，零值保留零；不裁剪、不重采样、不增加 padding。该强度方案是显式开发配置，待实验协议确认。文件 shape/affine 不匹配即停止，不偷偷配准。

## 后续 AMD 检查（本轮未执行）

经用户确认病例后，指定一个实际 `CASE_ID`，只做一次 forward/loss/backward；不创建训练 run、不更新参数。必须按真实病例完整网格执行，OOM 即停止。

```bash
PYTORCH_MIOPEN_SUGGEST_NHWC=1 MIOPEN_ENABLE_LOGGING=0 \
python scripts/brats.py check --device cuda:0 --cohort brats2023_gli \
  --source "$HOME/下载/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip" \
  --case CASE_ID
```

不要把旧 MONAI/CT 的显存数字用于此新编码器和全分辨率关系查询。布局选项不能证明所有中间算子/solver 都是 NDHWC。

遇到 OOM 时，在同一条 `check` 命令后加 `--profile-memory` 再做一次定位。
该选项仅用于无参数更新的 check，复用已有算子/模块 profiler，分别观察 forward、loss、backward。
stdout 输出失败阶段、活动模块/ATen 算子、输入输出 shape/stride/实际布局、最后成功模块、
各阶段 allocated/reserved 峰值及最大算子增量峰值。OOM 现场在模型局部变量展开前采样，
不重试、不清缓存、不手动释放激活、不改 dtype/公式；退出后移除所有 hooks。
不同阶段的峰值经过合并，不能将嵌套模块时间/峰值相加；诊断时间包含同步开销，不能作为训练速度。
backward 的算子能定位到 ATen 和形状，但不声称反查到了精确的前向模块或 MIOpen 内部算法。

## 可恢复训练能力与实验边界

新 `brats_regions` 模式复用现有 Trainer：AdamW、原子 checkpoint、RNG/病例顺序恢复、ETA、JSONL、TensorBoard。final/coarse BCE、DiceLoss、soft Dice 与内部 WT/TC/ET hard Dice 使用正确三通道语义。独立阈值预测不做层级修正，监测记录违反数。旧 AMOS 功能和已有配置保留为历史入口。

正式数据划分、HD95/官方评估规则、训练周期及实验分组尚未决定。当前入口仅支持显式 step horizon，不套用 AMOS epoch/early-stop 方案。训练必须提供患者互斥且穷尽的 split JSON：`manifest_hash` 为 inspect 输出的 manifest hash，`train`/`dev` 为病例 ID 列表；同一患者的多次扫描不得跨集合。目录名/年份不等于无患者重复，跨队列实验还需核验。

完成协议和 AMD preflight 后才能使用以下参数形式（不是现在启动实验的指令）：

```
python scripts/brats.py train --source SOURCE --cohort brats2023_gli \
  --split AGREED_SPLIT.json --max-steps AGREED_STEPS --device cuda:0 --run-dir RUN_DIR
```

恢复时保留所有参数，追加 `--resume RUN_DIR/last.ckpt`。模型、损失、预处理、阈值、优化器、运行条件、完整 split/manifest、源码身份必须一致。`--stop-after` 仅在完整 optimizer step 边界停止，不改计划。旧 AMOS checkpoint 不能迁移成此模型；原迁移开关仍仅允许其审计过的历史版本。

原始数据、runs、权重、缓存与诊断输出不进入 Git。新增提交采用 Conventional Commits，不重写历史。
