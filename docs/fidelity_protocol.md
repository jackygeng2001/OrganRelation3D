# 少量完整 CT 重采样保真度协议

2026-09-23；本轮 CPU 探索测试，非正式模型配置。A=(1.5,1.5,3)、B=(2,2,3)、C=(2,2,5) mm，按 R,A,S 顺序。不实现神经网络、不改变 METHOD_SPEC 公式、不启动 GPU 或正式实验。没有预设小器官合格阈值。

## 选择与资源

只从已审查的 200 例 training CT 中选择 configs/fidelity_pilot.json 列出的 10 例。覆盖典型目标体素量 0043、最大候选体积 0097、最长扫描 0119、最大各向异性 0133、最大原始体积/1.25 mm 薄层 0279、2 mm 0214、2.5 mm 0245、768 平面/负强度截距 0001、512 平面/零截距 0004，以及此前 C 假设补齐极端 0137。0137 本轮没有实施网络 padding。

逐例、逐候选处理，CT float32、标签 uint8；指标按 8 层分块，完成后释放。预先检查保守工作内存估计与可用 RAM，工作集限额 3 GiB，记录进程峰值工作集。阶段检查不是操作系统硬性内存限额。超预算停止并报告，不减少病例/器官/扫描范围。CPU 线程环境变量默认 2；SciPy ndimage 操作通常由单线程执行。只保存 JSON/CSV 和复核 PNG，不保存大体积重采样 CT。

## 强度、世界坐标与边界

1. 分别读取图像和标签自己的 affine 与 NIfTI scaling；NiBabel 读取器应用 `stored*slope+intercept`，不再次加 -1024。独立原始数据小样本验证 CT scaling。标签缩放后必须恰为 0–15 整数，否则拒绝。优先有效 sform，随后 qform；独立自有头解析与 NiBabel affine 比对。本轮已审查数据为 mm 单位、LAS、轴对齐；其他单位/斜切需另行扩展验证，不静默接受。
2. NIfTI affine 描述体素中心。对每轴用体素单元外边界定义覆盖范围 `[lo,hi]`，长度 `L=hi-lo`。目标正向 RAS，`n=ceil(L/t)`，保持世界中心，`lo'=lo-(n*t-L)/2`，首中心 `lo'+t/2`。总扩展 `0<=n*t-L<t`，不裁扫描两端。不把这个少量边界扩展当成真实新增信息，也不将其等同于网络输入倍数补齐。
3. 拉取式采样坐标 `source_index = inverse(source_affine) @ target_affine @ target_index`。图像与标签使用同一目标网格，但各用自己的原始 affine；禁止仅按 shape 缩放。边界选择最近边缘复制，记录体积变化可能受边界延拓影响。这个规则仅为本次候选比较，不冻结正式边界规则。
4. CT 三线性；仅降采样轴做 Gaussian 抗混叠预滤波，原体素单位 `sigma=max((target_spacing/source_spacing-1)/2,0)`，truncate=3，边界 nearest。这是记录明确的探索滤波设置，不是公式改变或正式强度规范。所有候选同一规则；指标体现 spacing 与这一滤波/插值组合的效果。标签最近邻，不滤波。
5. 标签从目标网格最近邻恢复至原标签网格。CT 三线性恢复至原图像网格，返回时不加额外 Gaussian。逆坐标映射可重现，不代表降采样信息可逆。原图像/标签几何差异必须先满足既有 0.001 mm 容差。

参考实现依据：[NiBabel scaling 与 affine](https://nipy.org/nibabel/nifti_images.html)、[SciPy affine_transform 拉取映射与 nearest 边界](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html)。Gaussian sigma 取用常见降采样经验规则，见 [scikit-image resize 的抗混叠说明](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize)，本项目不依赖 scikit-image，也不声称它是此数据的最佳滤波参数。

## 测量与复核

- 每例先在原 CT/标签网格分别执行恒等变换；CT 最大绝对误差 <=1e-4 HU、标签完全一致。先通过合成测试：世界坐标线性斜坡、翻转/置换、中心与完整边界、已知块体积/最近邻、细小结构消失、原本缺失、非法类别、scaling/错位拒绝和 CT 指标分块连续性。
- 全部 15 类记录原始/目标/返回体素数和 mm³ 体积；体素体积取各自 affine 的 3x3 行列式绝对值。体积变化报告 target 对 original 与 roundtrip 对 original 两种。
- 原始标签中类存在时计算原空间往返 Dice；原标签计数为零记 NA 并单列，不据此断言解剖上不存在（仍可能涉及扫描覆盖或标注情况）。原本有标签但目标或返回计数为零分别标记意外消失。此处对存在病例平均仅为保真度描述，不冻结正式训练或评估空类规则，也不把存在状态送入模型。
- CT 返回原网格后记录全局/前景 MAE、各器官均值/标准差、前景相邻体素物理梯度幅度比。梯度只纳入两个端点均为前景的对，但允许跨器官类别，反映总体高频响应，不等同诊断质量评分。
- 局部对比度=原标签内平均 HU 减去外部 2–5 mm 且标签为背景的环内均值。记录原始值、返回值和绝对幅度比；原始对比度为零时比值 NA。背景环不是均匀组织，增强扫描也不同，因此不设置通过阈值。
- 每例保存直接目标网格叠加图 A/B/C，另保存原网格对应切面的 original/A/B/C 对比图（整体、左右肾上腺）。毫米纵横比，固定显示窗 [-160,240] HU；该窗口不作用于数值测量。真实标签仅帮助本次评估选取显示 ROI，绝不用于模型裁剪/节点判断。
- 执行前后核对原始文件 size、mtime_ns、完整头摘要；读取代码无写回路径。记录代码/config/metadata 哈希、Git、依赖、耗时、峰值内存。未声称逐文件做全体素加密哈希验证。

往返 Dice 是采样几何保真度，不是模型分割性能；CT 往返多了一次插值，结果也不是单次预处理的纯损失。5 mm→3 mm 会增加插值切片，无法补回原本未采集的细节。10 例是有目的的极端/典型抽样，不代表 200 例随机总体估计。

## CPU 环境与运行

元数据统计仍只需标准库。真实重采样使用独立 Python 3.11+ venv，无 torch/GPU 包；不得覆盖现有 AMD 或 CUDA 环境。

```powershell
python -m venv .venv-resampling
.\.venv-resampling\Scripts\python.exe -m pip install -r environments/requirements-resampling-cpu.txt
.\.venv-resampling\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv-resampling\Scripts\python.exe -B scripts/test_resampling_fidelity.py --data-root "..\..\amos22" --metadata reports/ct_stats_20260922_final/metadata.json --output-dir reports/fidelity_run01
.\.venv-resampling\Scripts\python.exe -B scripts/summarize_fidelity.py --report-dir reports/fidelity_run01
```

若 venv 已存在无需重建或重装。输出必须是与数据目录分开的新目录。全部病例结果及图像留在被 Git 忽略的 reports/；不上传影像。完成后看 report.md、organ_summary.csv、organ_metrics.csv、ct_organ_metrics.csv、review_cases.csv 和逐例 JSON。未完成运行保留已经完成的逐候选结果，禁止把部分输出当成完整结论。

第二条汇总命令只读已完成的结果和 PNG，不再读取原始体积；增加厚层/薄层分组的 review_summary.md、目标网格概览拼图和显式双向索引矩阵 spatial_transforms.json，并记录汇总代码与输入结果哈希。

尺寸不整除的后续工程研究见 [variable_shape_unet_review.md](variable_shape_unet_review.md)：优先保存各级实际 shape、按 skip size 对齐解码，不先增加改变节点域的 padding 或 mask；仍需未来完整网络尺寸、相位、梯度与 AMD 显存实测。
