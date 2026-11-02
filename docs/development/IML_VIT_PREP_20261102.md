# IML-ViT 准备清单（2026-11-02）

IML-ViT 是 TruFor 之外的局部定位候选；本清单先固定来源与约束，**代码/权重尚未到位**，
因此不阻塞 TruFor 小批。

## 来源与许可

| 项目 | 值 |
| --- | --- |
| 官方仓库 | `https://github.com/SunnyHaze/IML-ViT` |
| 固定提交（准备时） | `07dd2be0f4ea27a5c97c9fa5ffbe236733833eac` |
| 论文 | `arXiv:2307.14863`，IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer |
| 代码许可 | **MIT**（LICENSE） |
| 权重 | Google Drive / Baidu NetDisk，README 链接；2024-03-24 发布 CASIAv2 与 CAT-Net/TruFor 协议 checkpoint |
| 本地状态 | **代码、权重均缺失**；需要用户上传或提供可达链接 |

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

## 当前阻塞

- 代码/权重缺失（Google Drive/Baidu，本机不可达）。
- GPU 当前不可见；cgroup 回到 2 GiB，连 CPU TruFor 小批也被 SIGKILL。
  需先恢复运行环境，再继续定位专家评测。
