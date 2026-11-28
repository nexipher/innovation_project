# 2026-11-28 判例修订 — 简短改动清单

审核入口：`index.html`。现在共 **26 条**：24 条主样板 + 2 条单专家样板。所有专家调用
都是真实 PROBE/TruFor 重放；工具输出、evidence_id、三图没有改，改的是候选文本、
标签、certainty 和新增样板。

## 1. 整图标签改为 Uncertain（局部 Tampered、总体 Fake）

| sample | 原 whole | 新 whole | certainty |
|---|---|---|---|
| `autosplice__54029__1__q95` | Real | **Uncertain** | High |
| `autosplice__45749__1__q95` | Real | **Uncertain** | High |
| `autosplice__69868__0__q95` | Real | **Uncertain** | Medium |
| `autosplice__48251__0__q95` | Real | **Uncertain** | Medium |
| `autosplice__72548__1__q95` | Real | **Uncertain** | Medium |
| `autosplice__64223__2__q95` | Real | **Uncertain** | Medium |
| `autosplice__45914__2__q95` | Real | **Uncertain** | High |

**对照未改**：`autosplice__55310__1__q95` 保持 `Real / Tampered / Fake`（PROBE 0.0782
低于阈值，整图有独立支持）。

## 2. 边缘响应与 certainty

| sample | 改动 |
|---|---|
| `autosplice__51186__authentic__q95` | 局部 Not tampered、整体 Real 不变；certainty **High → Medium** |
| `autosplice__57332__authentic__q95` | 核对裁剪 `[500,122,1000,494]`：高响应在整图底边/右缘，不是内容区。局部 **Uncertain → Not tampered**，整体 **Uncertain → Real**，certainty **null → Medium** |
| `autosplice__65547__authentic__q95` | 局部 Uncertain 不变；内容区响应但解释不唯一 |
| `autosplice__52769__authentic__q95` | 局部 Uncertain 不变；面部响应所在区域可靠性低 |

## 3. PROBE 调用理由

全部 25 次 PROBE 调用统一为整图来源判断，例如：

- 首轮：`probe — 整图来源尚未测量；PROBE 是整图来源的判别器。`
- 补测：`probe — 整图来源仍未测量；PROBE 是整图来源的判别器，补测以完成整图来源判断。`

不再在 PROBE 的 `Expert Target & Hypothesis` 里写“区分局部/整图”或“补测局部”。

## 4. 新增 2 条单专家样板

| sample | 调用 | 候选结论 | 停止理由 |
|---|---|---|---|
| `single_probe__genimage__Glide__...` | PROBE only | Fake / AI-generated / Not applicable | `model_concluded` |
| `single_trufor__autosplice__54029__1__q95` | TruFor only | Fake / Uncertain / Tampered | `model_concluded` |

两条都是 1 次专家调用、2 轮助手输出；单专家页按实际调用展示图片（PROBE-only 页没有
TruFor 三图，TruFor-only 页没有 PROBE 裁剪）。

## 5. 运行与审计

- 26/26 完成，0 失败，50 次专家调用；config `255ea663511f1f44`；wall 36.5 s。
- 证据/覆盖/certainty 检查均 pass；target/record 26/26 一致。
- `AUDIT_REPORT.md` 26/26 pass、0 问题；`review_assets/` 按 26 条重打包约 15.7 MB。
- 相关测试 107 passed；core 834 passed、1 skipped。

## 6. 2026-11-28 二次文字微调（标签不变）

| sample | 改动 |
|---|---|
| `autosplice__45749__1__q95` | TruFor 补测理由改为「检查企鹅身体是否有局部篡改证据」 |
| `autosplice__51186__authentic__q95` | 高响应描述限定在实际投递的裁剪内；删除「画框边缘/原图右缘」的扩写 |
| `autosplice__57332__authentic__q95` | 删除「整图右缘」，只保留「整图底边」；标签仍 Real / Not tampered / Medium |
| `single_probe__genimage__Glide__...` | 毛发偏软写成弱线索，PROBE 0.9672 作为主要支持；标签仍 Fake / AI-generated / Not applicable |

四条只改助手文字，工具输出、evidence_id、标签、certainty、框和调用顺序不变；
`AUDIT_REPORT.md` 26/26 pass、0 问题。
