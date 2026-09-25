# 3D U-Net 骨干 CPU 验证记录

2026-09-25。实现与接口见 [backbone_stage1.md](backbone_stage1.md)。本轮仅 Encoder/Decoder；标准 PyTorch 算子，没有关系推理、粗头、联合损失、优化器训练步骤、真实 CT 网络运行或 GPU 实验。正式配置及 A/B/C spacing 均未冻结。

## 结构和实际形状

微型配置：3 个尺度、channels=[4,8,16]、两次三轴 stride=2。每级两层 3×3×3 Conv3d，GroupNorm(groups=2) 与非原地 LeakyReLU；Decoder 三线性上采样到实际 skip shape、拼接并卷积，1×1×1 输出头得到 16 类原始 logits。共 22,380 个参数。

实际诊断输入 `[1,1,17,18,19]`，skips 为 `[1,4,17,18,19]` 与 `[1,8,9,9,10]`，F 为 `[1,16,5,5,5]`，输出 `[1,16,17,18,19]`。图模块未来位于 encoder 与 decoder 之间，Decoder 已验证可接受同形 Fprime，且梯度仍流经 F 与各 skip。

## 已执行检查

1. **骨干 22 项测试全部通过，无跳过**：配置和解析尺寸、全部三轴奇偶组合、2/3/4 级配置、不同逐轴步幅、各轴单例/1³ 空间、batch 2、前后向有限值、参数手算数量与注册覆盖、所有参数有限非零梯度、Fprime 替换/不原地修改、raw logits/不接收真实标签、非法输入和金字塔拒绝、float64/状态复现、网络输入方向有限差分、插值 gradcheck、角点到 F 的传播及插值坐标斜坡。
2. **原有 53 项回归测试全部通过，无跳过**：在未改动的重采样环境运行 `-p 'test_[!b]*.py'`，涵盖元数据、候选估算和合成保真度测试。没有再次处理 10 例真实 CT。
3. `pip check` 无损坏依赖。独立诊断入口实际完成一次微型前后向；参数数目 22,380、参数/梯度各 89,520 bytes，missing/nonfinite/zero-gradient tensor 清单均为空。
4. **200 例 × A/B/C = 600 组尺寸**完成逐层 shape 代数检查，未读取影像或分配实际尺寸 Tensor；无整除补齐需求。这只验证当前微型配置的尺寸逻辑，不验证正式结构、图表示充分性或显存可行性。
5. CPU 显存字段为 None；没有把参数字节数或 CPU 测试耗时解释为 GPU 显存或训练速度。

测试命令：

```powershell
.\.venv-backbone-cpu\Scripts\python.exe -B -m unittest discover -s tests -p test_backbone.py -v
.\.venv-resampling\Scripts\python.exe -B -m unittest discover -s tests -p 'test_[!b]*.py' -q
.\.venv-backbone-cpu\Scripts\python.exe -m pip check
```

微型插值梯度的 FP32 累加曾出现 `60.0000038` 对理论 `60` 的差异，修正的是测试的严格相等断言，改为 rtol=1e-6/atol=1e-6；未改变实现。float64 gradcheck 和方向有限差分独立通过。

## 环境与来源

Windows CPU，Python 3.12.14，PyTorch 2.8.0+cpu，CUDA/HIP 标识均为 None。独立环境 `.venv-backbone-cpu`，CPU 线程数 2。测试微型配置随机种子及诊断配置均保存在 JSON 中；测试 setUp 使用固定种子 1337，诊断入口使用配置种子 20260923。

CPU 主 wheel 在用户新增“大文件手动下载”要求到达前已完成下载。中断恢复后没有再下载大文件：通用依赖主要从现有 pip wheel 缓存离线安装，setuptools 从 Codex 自带 Python 的记录文件复制；唯一缺少的 MarkupSafe Windows wheel 为 15,105 bytes，通过 PyPI 提供的 SHA256 核对。原 GPU/重采样环境未修改。离线来源记录保存在本地 reports/backbone_offline_setup.json。

已测依赖：filelock 3.19.1、fsspec 2025.10.0、Jinja2 3.1.6、MarkupSafe 3.0.3、mpmath 1.3.0、networkx 3.2.1、setuptools 84.0.0、sympy 1.14.0、typing_extensions 4.15.0。NumPy 不属于此处 Torch 必需依赖，本隔离环境未安装，会出现可选 NumPy 初始化提示；测试只用 Tensor，不掩盖该提示。未来 NIfTI/NumPy 到模型的联合数据管线尚未验证。

最终诊断 JSON 与尺寸报告保存在 reports/backbone_stage1_cpu_20260925.json 和 reports/backbone_shapes_20260925.json，记录实际执行 Git、config/源码哈希、完整环境与结果。开发期间的 *_development/dev 报告仅供调试，不能替代最终来源记录。

## 剩余风险和下一步

- 尺寸对齐不等于物理重建：stride 卷积与插值的名义坐标相位不同，解析测试已明确展示这一点；未声称奇偶输入严格平移等变或信号可逆。正式步幅/插值设置仍须统一记录并验证任务效果。
- 角点脉冲验证的是架构存在边界传播路径，不保证任意训练权重都会利用边界信息。内部零边界也可能影响边缘特征。
- 尚未在 PyTorch 2.11 ROCm 或 RTX 5060 CUDA 执行；AMP、checkpoint、真实 CT 网络、完整方法联合梯度和优化器状态显存均未测试。预留 GPU 接口不是 GPU 验收。
- 单例空间合法不意味着有足够的器官空间表示；channels=[4,8,16] 只用于 CPU 微型检查，不是正式架构建议。可变尺寸 batch>1 仍要求同一 Tensor 的各例 shape 一致；后续不通过未经确认的 padding 拼批。
- 下一阶段建议先验收此骨干接口，再按明确任务实现 16 类粗头与 Space-to-Node 的公式/梯度测试；完整方法与 AMD 验证完成前不启动正式实验。
