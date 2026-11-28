# 30 图验证对照 — 结论摘要

- 数据：30 张 split validation 源，与 24 条训练轨迹的 22 个唯一源图无交集。
- 流程：原版 Qwen3-VL 与加载 LoRA 的同一 Agent、同系统 prompt、同 PROBE/TruFor、
  同贪心 non-thinking 自主调用；只有 adapter 不同。
- 两臂均 30/30 完成、0 失败（base 1488.7 s / peak 21.371 GB；LoRA 1862.0 s /
  peak 21.481 GB）。

## 指标

| 指标 | 原版 | LoRA |
|---|---:|---:|
| 局部篡改检出（10 张） | 2 | 1 |
| 真图局部误报（10 张） | 0 | 0 |
| 真图整图误报（AI-generated） | 2 | 2 |
| 真图总体误报（Fake） | 2 | 2 |
| 两分项同时判对（30 张） | 8 | 7 |
| — 篡改图（10） | 1 | 1 |
| — 生成图（10） | 0 | 0 |
| — 真图（10） | 7 | 6 |
| 弃权（总体 Uncertain） | 0 | 2 |
| 专家调用总数 | 54 | 49 |
| 平均每会话调用 | 1.800 | 1.633 |
| 调用分布 | `{2:24, 1:6}` | `{2:19, 1:11}` |

## 变化要点

- **没有两分项同时判对的增益案例**；LoRA 相对原版失去 1 条两者同时对：
  `val__genimage__Real__d3429f29...`：base Real/Not tampered/Real（both True）→
  LoRA Real/Uncertain/Real（both False），调用 2→1。
- **局部检出失去 1 条**：`val__autosplice__57143_0`：base AI-generated/Tampered/Fake
  → LoRA AI-generated/Uncertain/Uncertain；既失去 local Tampered，也把总体变成弃权。
- LoRA 新增弃权 2 条；真图局部误报仍为 0，但真图整图/总体误报不变（各 2）。
- 生成图两分项同时判对两臂都是 0/10：两个模型都没有把局部写成 `Not applicable`，
  而是 `Not tampered`/`Uncertain`/`Tampered`；这是共同的协议行为，不是 LoRA 独有。
- 调用减少 5 次（54→49），但主要来自少调用一次专家，不是判对率提升。

## 变化最大的其它案例

| sample | group | 原版 | LoRA |
|---|---|---|---|
| `val__genimage__Real__d1b49378...` | real | Real / Uncertain / Real | Real / Uncertain / Uncertain |
| `val__genimage__Midjourney__487...` | generated | Uncertain / Tampered / Fake | Uncertain / Uncertain / Fake |
| `val__autosplice__56086_0` | tampered | AI-generated / Not tampered / Fake | AI-generated / Uncertain / Fake |
| `val__genimage__VQDM...00143` | generated | AI-generated / Uncertain / Fake | AI-generated / Not tampered / Fake |
| `val__genimage__BigGAN__739...` | generated | AI-generated / Not tampered / Fake | AI-generated / Uncertain / Fake |

## 读法

- 30 图样本很小；这不是 LoRA 的最终评测。按本轮数据，**6 步/48 步小试没有显示
  局部检出或两分项同时判对的提升**，反而少 1 条局部检出、少 1 条 both、增加 2 条弃权；
  调用略少。
- 两臂都很少写 `local Not applicable`，这是下一轮数据/判例需要单独处理的行为。
- 未扩 250 条；未碰 test。
