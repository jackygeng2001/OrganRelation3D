# 3D U-Net 骨干第一阶段

状态：实现草案，可配置结构与微型 CPU 验证；不是正式实验骨干配置。关系公式、15 节点、全扫描上下文和 A/B/C 候选不变。本轮只实现 Encoder/Decoder；没有粗头、图模块、融合模块或分割训练目标。

## 结构与接口

输入为单通道 `[B,1,D,H,W]` 完整体积。每个尺度两次 `Conv3d(3, padding=1) → GroupNorm → LeakyReLU`；首尺度 stride=1，其余尺度的第一层卷积按配置逐轴 stride=1/2，下采样至少一个轴。第二层始终 stride=1。微型配置使用无卷积偏置、GroupNorm groups=2/eps=1e-5、LeakyReLU slope=0.01。激活不原地修改。归一化还可明确配置为 none，卷积与输出头 bias 分别配置。

尺度数由 channels 长度指定，至少 2 级；每个相邻尺度有独立三轴 stride。卷积核 3、每级两层、拼接式 skip 和三线性上采样是本次 v1 实现的块结构，并非宣称支持所有 U-Net 变体，也不是正式实验结构冻结。

每轴尺寸递推：`n(l+1)=ceil(n(l)/stride(l))`。使用整数 padding=1 的内部卷积边界规则，不使用仅对 stride=1 合法的 `padding='same'` 字符串。没有在模型输入上增加体素、倍数补齐、裁掉两端、缩小输入或新增有效域 mask。

Decoder 从最深层开始，逐级按 `size=skip.shape[-3:]` 三线性上采样，与该 skip 按通道拼接，再用两层卷积块输出该 skip 的通道数。最后 `Conv3d(C0,16,1)` 输出原输入分辨率的 logits，不做 softmax。最高分辨率 skip 的空间 shape 保留了原输入尺寸。

微型配置 configs/backbone_micro.json 的示例（不是正式模型）：

|位置|张量形状|
|---|---|
|image|[B,1,17,18,19]|
|skip[0]|[B,4,17,18,19]|
|skip[1]|[B,8,9,9,10]|
|F|[B,16,5,5,5]|
|解码第一级|[B,8,9,9,10]|
|解码第二级|[B,4,17,18,19]|
|logits|[B,16,17,18,19]|

```python
config = BackboneConfig(**model_config)
encoder = Encoder3D(config)
decoder = Decoder3D(config)
F, skips = encoder(image)  # EncoderFeatures: deepest, skips
logits = decoder(F, skips)  # 本轮仅直接连接骨干，用于接口测试
# 后续完整方法在两者之间显式实现粗头、图、回写和残差融合：
# logits = decoder(Fprime, skips)
```

skips 按高到低分辨率排列，不重复保存最深层 F；返回的 Tensor 不 clone/detach，保留完整梯度，也不原地修改。Decoder 检查 batch、通道、device、dtype、skip 数量及空间金字塔是否匹配。Fprime 必须与 F 同形、同 device/dtype，保留反向路径。GroupNorm 的 norm_eps 仅是骨干归一化参数，不是 METHOD_SPEC 软池化/损失公式中尚待确认的 epsilon。

`UNetBackbone3D` 只是直接连接 Encoder→Decoder 的骨干验证入口，不是完整 Segmentor；不能把其输出当作已实现了本研究方法。未来共享 F 同时交给粗头与关系/回写路径，依然遵守 METHOD_SPEC 的模块顺序。

## 尺寸、边界与相位

微型配置每级每组至少有 2 个通道，因此 `[1,1,1]` 空间形状也可合法前后向。其他配置如果某级 `channels/groups × D×H×W <= 1`，会在卷积前拒绝，避免退化 GroupNorm；不通过补齐或按病例减少层数处理。各轴 stride 可不同，但固定配置不会根据病例临时改变。

按 skip size 对齐能确保拼接与完整输出尺寸正确，**不是 stride 卷积的物理逆变换**。当前 kernel=3、padding=1 的 stride-s 卷积名义采样中心为 `x=s*j`；三线性插值使用另一套坐标规则：

- `align_corners=False`：源坐标 `(j+0.5)*n_in/n_out - 0.5`，边缘钳制；微型配置使用此值。
- `align_corners=True`：非单轴情况下源坐标 `j*(n_in-1)/(n_out-1)`。

所以简单中心采样斜坡下采样再按尺寸插值，可能与原斜坡有相位差，尤其是奇偶混合尺寸；网络特征也不能被当成 NIfTI 世界坐标的点采样 CT。测试分别核对卷积中心、插值解析公式与边界依赖，不把二者组合断言为恒等变换。不暗中更换插值、裁剪 skip 或改节点归一化坐标来“修复”这一行为。正式骨干选择前仍需结合任务表现检查这些配置；本轮不声称奇偶尺寸之间严格平移等变。

卷积内部零边界会影响边缘响应；这和在 CT 网格之外显式添加参与节点统计的输入体素不同。本轮无显式输入 padding 和 mask，未来节点仍按实际 F 的 `(Df,Hf,Wf)` 构造坐标和 N。

参考：[PyTorch 2.8 Conv3d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv3d.html)、[interpolate](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.interpolate.html)、[GroupNorm](https://docs.pytorch.org/docs/2.8/generated/torch.nn.GroupNorm.html)。这些仅作为标准算子行为依据，不替代本项目的设备测试。

## CPU 环境和可复现测试

独立 `.venv-backbone-cpu`，与重采样环境和任何 GPU 环境分开。PyTorch CPU 2.8.0 作为当前可用 API 的测试基线；不声称已验证主设备的 PyTorch 2.11 ROCm。

```powershell
python -m venv .venv-backbone-cpu
.\.venv-backbone-cpu\Scripts\python.exe -m pip install -r environments/requirements-backbone-cpu.txt
.\.venv-backbone-cpu\Scripts\python.exe -B -m unittest discover -s tests -p test_backbone.py -v
.\.venv-backbone-cpu\Scripts\python.exe -B scripts/check_backbone.py --device cpu --output reports/backbone_cpu_run01.json
python -B scripts/audit_backbone_shapes.py --estimates reports/candidates_20260922_final/estimates.json --output reports/backbone_shapes_run01.json
```

CPU 官方 wheel 源对通用依赖解析较慢时，可先从官方 CPU 源安装同一 `torch==2.8.0+cpu --no-deps`，再从 PyPI 补齐 torch 声明的 CPU Python 依赖并执行 pip check；不要改用未知 GPU wheel。安装入口依据 [PyTorch 历史版本说明](https://pytorch.org/get-started/previous-versions/)。重采样测试的可选依赖仍使用其单独 requirements 文件，不能把本 CPU 文件用于 AMD/CUDA 环境。

按用户最新规范，涉及较大 wheel 时由用户手动执行下载/安装命令，助手不再自行启动大文件下载。本轮 CPU wheel 在此要求到达前已下载完成；没有新增 GPU 安装。独立 shape audit 不依赖 torch，仅对已有 200 例 × A/B/C 的尺寸记录计算各级 shape，不读取 NIfTI，也不分配真实尺寸张量。

测试包括全部三轴奇偶组合、可变深度/宽度与逐轴步幅、单轴为 1、前后向有限值、所有参数注册/梯度覆盖、Fprime 替换与 skips 的梯度、参数状态复现、原始 logits/不接收标签、坐标斜坡、各角点抵达 F、插值 gradcheck 与网络输入方向数值梯度。测试用平方均值标量仅探查 autograd，**不是 METHOD_SPEC 的分割目标**；不调用训练优化步骤。

探针默认只允许小型 CPU 输入，配置过大时直接拒绝，不偷偷缩小。模型本身没有这个微型输入限制。报告记录 Git/config/源码哈希、输入/F/skips/logits 形状、参数和梯度数量、依赖、device 及诊断时间。

## AMD/CUDA 后续接口（本轮未执行）

入口接受显式 `--device`，模型内部没有 `.cuda()`、设备编号、数据路径或新建硬编码 CPU 张量。标准 PyTorch `.to(device,dtype)` 由调用者管理。ROCm 与 CUDA 使用同一 `torch.cuda` allocator 接口，由 `torch.version.hip` 区分后端；依据 [PyTorch HIP 语义](https://docs.pytorch.org/docs/2.8/notes/hip.html)。

`DeviceMemoryMonitor.begin()/snapshot(phase)` 提供 allocated/reserved 与自 begin 起的累计峰值，分别在参数/输入、forward、backward 后取样；CPU 返回 None，不把 0 当实测显存。GPU 会同步以使取样具有阶段意义，其他进程/其他分配器开销不由这些数字完整覆盖。后续由用户在既有 AMD 环境中手动运行，例如：

```bash
python -B scripts/check_backbone.py --device cuda:0 --output reports/backbone_amd_micro_run01.json
```

这是未来微型骨干兼容入口，不是完整模型显存验收命令。本轮没有自动执行或安装 GPU 包。输出不含图模块、粗分支联合损失、优化器状态，因此不能用于推断正式方法在 24GB 上可训练。AMP、checkpoint、torch.compile、多 GPU、ROCm/CUDA 数值与真实 CT 前后向均未验证。后续扩展完整模型探针时再记录相应额外阶段，不能让微型宽度或本机 CPU 容量替代正式结构决策。
