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
| `scripts/brats.py` | inspect / 无更新 check / 单病例 overfit / 受控 train |
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

## AMD 无更新检查

经用户确认病例后，指定一个实际 `CASE_ID`，只做一次 forward/loss/backward；不创建训练 run、不更新参数。必须按真实病例完整网格执行，OOM 即停止。

```bash
PYTORCH_MIOPEN_SUGGEST_NHWC=1 MIOPEN_ENABLE_LOGGING=0 \
python scripts/brats.py check --device cuda:0 --cohort brats2023_gli \
  --source "$HOME/下载/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip" \
  --case CASE_ID
```

不要把旧 MONAI/CT 的显存数字用于此新编码器和全分辨率关系查询。布局选项不能证明所有中间算子/solver 都是 NDHWC。

若 ROCm 的 NDHWC Conv3d 使用低 workspace 但很慢的 naive solver，可在 `check` 或 `train` 显式增加 `--convolution-backend depth_sliced`。默认仍为 `native`。该选项把每个原 Conv3d 的深度核展开为 NCHW Conv2d 求和，每次最多处理 4 个输出深度平面，保留跨平面贡献、边界 padding、stride/dilation 和完整输出；1³ 卷积执行等价线性映射。它不是 2D 分割、patch 输入或 activation checkpointing，权重、初始化、FP32 和 loss 不变。CPU/ROCm 已做输出及梯度容差验证；NVIDIA 尚未实测。首次调用可能包含 MIOpen 算法选择，稳定耗时应预热后测量。

该执行方式及 depth chunk 写入 check JSON 和训练 runtime identity；resume 必须使用相同选项，不放宽旧 checkpoint 身份校验。算子临时内存有界不代表所有 backend workspace 有固定上限，真实整卷仍需检查峰值。正式训练吞吐还包括 optimizer、读取、监测和保存，不能用单次 forward/backward 代替。

`depth_sliced` 的解码融合卷积直接接收 decoded/enhanced 两个原特征，仅在至多 4 个深度平面的 Conv2d 输入内按原通道顺序拼接。避免保存整幅拼接副本，反向分别返回两个输入的梯度；不修改原卷积权重、跨切面贡献或后续 GN/激活。`native` 仍使用原整幅 cat。此工程更新不绕过源码身份校验；无 checkpoint 的失败 run 不能 resume，应保留旧目录并使用新目录重试。

遇到 OOM 时，在同一条 `check` 命令后加 `--profile-memory` 再做一次定位。
该选项仅用于无参数更新的 check，复用已有算子/模块 profiler，分别观察 forward、loss、backward。
stdout 输出失败阶段、活动模块/ATen 算子、输入输出 shape/stride/实际布局、最后成功模块、
各阶段 allocated/reserved 峰值及最大算子增量峰值。OOM 现场在模型局部变量展开前采样，
不重试、不清缓存、不手动释放激活、不改 dtype/公式；退出后移除所有 hooks。
不同阶段的峰值经过合并，不能将嵌套模块时间/峰值相加；诊断时间包含同步开销，不能作为训练速度。
backward 的算子能定位到 ATen 和形状，但不声称反查到了精确的前向模块或 MIOpen 内部算法。
逐算子同步会显著放大分段反向的诊断耗时；定位后使用不带 `--profile-memory` 的普通 `check` 验证完整反向和实际总耗时。当前脑肿瘤实现通过线性投影结合顺序、私有激活缓冲复用与全局 GroupNorm 的分段解析反向减少临时内存，不改变完整扫描、模型参数、FP32 或 loss；分段仅限制逐元素梯度临时张量，不是 patch segmentation。

## 可恢复训练能力与实验边界

新 `brats_regions` 模式复用现有 Trainer：AdamW、原子 checkpoint、RNG/病例顺序恢复、ETA、JSONL、TensorBoard。final/coarse BCE、DiceLoss、soft Dice 与内部 WT/TC/ET hard Dice 使用正确三通道语义。独立阈值预测不做层级修正，监测记录违反数。旧 AMOS 功能和已有配置保留为历史入口。

正式数据划分、HD95/官方评估规则、训练周期及实验分组尚未决定。当前入口仅支持显式 step horizon，不套用 AMOS epoch/early-stop 方案。训练必须提供患者互斥且穷尽的 split JSON：`manifest_hash` 为 inspect 输出的 manifest hash，`train`/`dev` 为病例 ID 列表；同一患者的多次扫描不得跨集合。目录名/年份不等于无患者重复，跨队列实验还需核验。

完成协议和 AMD preflight 后才能使用以下参数形式（不是现在启动实验的指令）：

```
python scripts/brats.py train --source SOURCE --cohort brats2023_gli \
  --split AGREED_SPLIT.json --max-steps AGREED_STEPS --device cuda:0 --run-dir RUN_DIR
```

恢复时保留所有参数，追加 `--resume RUN_DIR/last.ckpt`。模型、损失、预处理、阈值、优化器、运行条件、完整 split/manifest、源码身份必须一致。`--stop-after` 仅在完整 optimizer step 边界停止，不改计划。旧 AMOS checkpoint 不能迁移成此模型；原迁移开关仍仅允许其审计过的历史版本。

单病例学习诊断使用独立 `overfit` 动作和 `configs/train_brats_single_case_overfit.json`：必须给 `--case`、禁止 `--split`，仅选择该病例训练并做 `train_monitor`，明确不是独立验证。科学设置与 `brats_regions.json` 相同；每 5 步检查梯度/参数更新、保存 checkpoint 并输出摘要，每 25 步监测三区域 hard/soft Dice 与预测体积。

```bash
PYTORCH_MIOPEN_SUGGEST_NHWC=1 MIOPEN_ENABLE_LOGGING=0 \
python scripts/brats.py overfit --config configs/train_brats_single_case_overfit.json \
  --device cuda:0 --convolution-backend depth_sliced --cohort brats2023_gli \
  --source "$HOME/下载/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip" \
  --case BraTS-GLI-00000-000 --max-steps 100 \
  --run-dir runs/brats2023_1case_100 --stop-after 5
```

先核对 5 步真实 optimizer 更新、显存与耗时；通过后保持同一配置和源码，删除 `--stop-after 5`，追加 `--resume runs/brats2023_1case_100/last.ckpt` 续到 100。两次都保留 `--max-steps 100`，不重新开始 run，不把 max_steps 从 5 改为 100，不需要 extension。该 GPU 连续训练须实际运行验收，不能用已有无更新检查代替。

原始数据、runs、权重、缓存与诊断输出不进入 Git。新增提交采用 Conventional Commits，不重写历史。

## 只读 checkpoint 概率诊断

`scripts/brats_diagnose.py` 从可信 checkpoint 旁的 `run.json` 读取原始 run 身份，并使用 checkpoint 的执行配置、完整 source manifest 和执行选项；验证 checkpoint 校验和、run 身份、当前数据指纹及 `src/` 源码。唯一跨源码例外是下述已审计 extension 升级，输出明确区分全部一致与 extension-only 兼容；模型/数据/loss/engine 仍必须一致。新诊断脚本可以不同于训练时的入口，但不放宽普通 resume。仅一次完整 image-only、FP32、eval/inference-mode 前向，不创建 optimizer，不反向，不修改 run/checkpoint。输出到终端；可选 PNG 只允许新文件、禁止写入原 run/source。

```bash
PYTORCH_MIOPEN_SUGGEST_NHWC=1 MIOPEN_ENABLE_LOGGING=0 \
python scripts/brats_diagnose.py \
  --checkpoint runs/brats2023_1case_100_v2/last.ckpt \
  --case BraTS-GLI-00000-000 --device cuda:0 \
  --overlay /tmp/brats_step100_overlay.png
```

source 默认沿用 checkpoint 路径，不需重复配置模型/预处理/backend。final/coarse 分别报告 WT/TC/ET 的 hard Dice、预测/GT/TP/FP/FN、GT 内外概率均值、P05/P50/P95、概率总量及区域外低于阈值的概率分布；coarse 先插值 logits 到原网格。区域外指各自 GT 区域之外，不能把 TC/ET 的区域外全部叫作正常组织。按可比字段核对同 step 监测记录，差异只记录，不在输出统计之前做全字典相等断言。CPU 统计与 GPU 原记录的浮点归约允许容差。

PNG 使用 Pillow（当前 AMD 已安装；可省略 `--overlay` 不渲染），每行独立展示一个区域：完整 FLAIR 切片、GT、final/coarse 的 TP/FP/FN 和 final 概率图。GT 最大截面只用于前向结束后的展示选片；无标签引导训练裁剪。显示为原数组 D 切片，不宣称标准放射学左右方向。默认不保存影像体积，不生成额外报告文件。

## 显式延长 BraTS 单病例训练

`--resume ... --extend-to 500` 调用现有 Trainer 的 horizon 机制；仍保留原 `--max-steps 100`，其余配置不变。新总步数必须同时大于旧计划和已完成步数，不改变 max_epochs。原 run_id/run.json、optimizer、RNG、病例顺序及日志连续；checkpoint 在读取下一病例之前原子记录原计划、新计划、父 checkpoint SHA256 和执行 provenance。TensorBoard 从原目录续写，ETA 按新总步数计算。

仅允许干净 `633da43204d1cc6a88185ee4c372b1618e2285f1`、源码指纹 `779cb7d48e6712e3f02e4354d289ea3394abce06177e031cdd260b922816a946` 的 BraTS single-case overfit 跨本次版本升级。白名单仅含 `scripts/brats.py`、`scripts/brats_diagnose.py`、`scripts/run_tests.py` 和 `training/extension.py`；模型、loss、数据、训练引擎、全部科学/运行/环境 identity 保持严格一致。普通 resume 不放宽，也不修改旧 AMOS 的迁移规则。

```bash
PYTORCH_MIOPEN_SUGGEST_NHWC=1 MIOPEN_ENABLE_LOGGING=0 \
python scripts/brats.py overfit --config configs/train_brats_single_case_overfit.json \
  --device cuda:0 --convolution-backend depth_sliced --cohort brats2023_gli \
  --source "$HOME/下载/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip" \
  --case BraTS-GLI-00000-000 --max-steps 100 \
  --run-dir runs/brats2023_1case_100_v2 \
  --resume runs/brats2023_1case_100_v2/last.ckpt --extend-to 500
```

500 是总步数，从 step 100 再训练 400 步。首次 extension 已提交后如中断，保留所有参数并去掉 `--extend-to 500` 普通恢复，horizon 会从 checkpoint 自动读取；不要把 `--max-steps` 改为 500。源码再次变化时，普通恢复仍严格拒绝。
