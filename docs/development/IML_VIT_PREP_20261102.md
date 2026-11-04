# IML-ViT 准备清单（2026-11-02）

IML-ViT 是 TruFor 之外的局部定位候选；本清单先固定来源与约束。2026-11-04 更新：
**代码已按固定 commit 到位，权重仍缺**；详见 `IML_VIT_VERIFICATION_20261104.md`。

## 来源与许可

| 项目 | 值 |
| --- | --- |
| 官方仓库 | `https://github.com/SunnyHaze/IML-ViT` |
| 固定提交（准备时） | `07dd2be0f4ea27a5c97c9fa5ffbe236733833eac` |
| 论文 | `arXiv:2307.14863`，IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer |
| 代码许可 | **MIT**（LICENSE） |
| 权重 | Google Drive / Baidu NetDisk，README 链接；2024-03-24 发布 CASIAv2 与 CAT-Net/TruFor 协议 checkpoint |
| 本地状态 | **代码已到位**（2026-11-04，固定 commit 解压到 `third_party/IML-ViT/`）；**权重仍缺失**，需要用户上传或提供可达链接 |

## 官方推理路径（待代码到位后逐项核实）

- `Demo.ipynb` / `utils/datasets.py`：图像先 **zero-padding**，再做 ImageNet 归一化；
  `if_return_shape=True` 用于把 map 还原到原图尺寸。
- 输出为 manipulation localization map；**不是**与 TruFor 相同的 score/conf 三件套。
  IML-ViT 不应被强行补成独立分类分数；若需要图像级分数，必须单独定义并记录。
- 预处理、padding、归一化、map 的通道/索引、原始尺寸还原都要与官方代码逐项对照，
  不能按论文名推测。

## 准备步骤（权重到位后）

1. 按固定 commit 下载代码，逐文件 sha256 写入 `SOURCE.json`，不修改官方文件。
2. 先做 CPU 覆盖/预处理检查，再在 AutoSplice q95 的 **dev** 小批上与 TruFor 同输入。
3. 检查 map 形状、有限性、原图尺寸还原、mask 几何关系；真实图误报单列。
4. 不调阈值、不翻 map 方向；阈值/后处理只在 val 上确定。
5. 只有 TruFor + IML-ViT 同输入对照完成后，才比较定位专家并决定接入哪一个；
   在此之前不改 Agent 的区域观察/证据回传/结案流程。

## 当前阻塞（2026-11-04 更新）

- **代码已到位**：固定 commit `07dd2be…` 解压到
  `/root/autodl-tmp/third_party/IML-ViT/IML-ViT-07dd2be0f4ea27a5c97c9fa5ffbe236733833eac`；
  archive sha256 `de140401…`。详细代码/预处理/输出语义见
  `docs/development/IML_VIT_VERIFICATION_20261104.md`。
- **权重仍缺**：Google Drive 直链连接超时，Baidu 需登录，HF 镜像 `HJYOGC/IML-ViT`
  只有 README，没有 checkpoint；没有使用非官方权重。
- 当前实例为 **2 GiB cgroup、GPU 不可见**；未构建模型、未加载权重。需先恢复
  GPU/更大 cgroup，并由用户上传或提供可达的官方 checkpoint，再做同输入对照。
