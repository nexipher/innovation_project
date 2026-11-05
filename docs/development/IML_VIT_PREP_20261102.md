# IML-ViT 准备清单（2026-11-02）

IML-ViT 是 TruFor 之外的局部定位候选；本清单先固定来源与约束。2026-11-05 更新：
**代码与官方权重均已到位，并已完成 220 输入同输入对照**；详见
`IML_VIT_VERIFICATION_20261104.md` 与 `LOCALIZATION_COMPARISON_20261105.md`。

## 来源与许可

| 项目 | 值 |
| --- | --- |
| 官方仓库 | `https://github.com/SunnyHaze/IML-ViT` |
| 固定提交（准备时） | `07dd2be0f4ea27a5c97c9fa5ffbe236733833eac` |
| 论文 | `arXiv:2307.14863`，IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer |
| 代码许可 | **MIT**（LICENSE） |
| 权重 | Google Drive / Baidu NetDisk，README 链接；2024-03-24 发布 CASIAv2 与 CAT-Net/TruFor 协议 checkpoint |
| 本地状态 | **代码与权重均已到位**（代码 2026-11-04；权重 2026-11-05 上传并 strict 加载）；已完成 16/220 输入前向 |

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

## 当前状态（2026-11-05 更新）

- **代码已到位**：固定 commit `07dd2be…` 解压到
  `/root/autodl-tmp/third_party/IML-ViT/IML-ViT-07dd2be0f4ea27a5c97c9fa5ffbe236733833eac`；
  archive sha256 `de140401…`。
- **权重已到位**：`/root/autodl-tmp/third_party/IML-ViT/weights/iml-vit_checkpoint_trufor_20231104.pth`，
  sha256 `9fa9ae88…`，367,195,954 B；212 tensors / 91,778,242 params，strict load 通过。
- **已完成**：16 张和固定 220 输入（100 base）前向；预处理、输出语义和失败/资源均已记录。
- **对照结论**：同输入下 IML-ViT 的 IoU/F1（0.161/0.239）明显低于 TruFor
  （0.537/0.645）；IML 无图像级 score。TruFor 作为初步定位专家候选，IML 不作为
  当前候选；仍受单子集/共享 mask/无未见 test 限制。
