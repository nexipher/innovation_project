# 固定双调用 vs 自主调用：逐图对照

输入：2026-11-10 manifest 的 200 张 q95；自主基线为 2026-11-27 part1+part2；固定模式每图先 `PROBE` 后 `TruFor`，再由同一 Qwen 结案。

## 总体

| 指标 | 自主基线 | 固定双调用 |
|---|---:|---:|
| 会话完成 | 200/200 | 200/200 |
| 总体正确 | 164/200 = 0.8200 | 164/200 = 0.8200 |
| 弃权（Uncertain） | 17/200 = 0.085 | 6/200 = 0.03 |
| 覆盖率 | 0.9150 | 0.9700 |
| 已结案准确率 | 0.8962 | 0.8454 |

| 真值类别 | 基线正确/弃权 | 固定正确/弃权 |
|---|---:|---:|
| real (n=60) | 46/4 | 43/2 |
| whole_generated (n=80) | 71/5 | 74/0 |
| local_tampered (n=60) | 47/8 | 47/4 |

## 整图与局部

| 指标 | 自主基线 | 固定双调用 |
|---|---:|---:|
| 整图生成检出 | 69/80 = 0.8625 | 74/80 = 0.925 |
| 整图真图误报 | 45/120 = 0.375 | 51/120 = 0.425 |
| 局部篡改检出 | 13/60 = 0.2167 | 6/60 = 0.1 |
| AutoSplice 真图局部误报 | 0/30 | 1/30 |
| 篡改图“整图 Real + local Tampered” | 5/60 | 3/60 |

局部标签分布：

- 基线 AutoSplice 真图：`{'Uncertain': 12, 'Not tampered': 18}`
- 固定 AutoSplice 真图：`{'Uncertain': 6, 'Not tampered': 23, 'Tampered': 1}`
- 基线篡改图：`{'Not tampered': 13, 'Not applicable': 2, 'Tampered': 13, 'Uncertain': 32}`
- 固定篡改图：`{'Not tampered': 15, 'Uncertain': 39, 'Tampered': 6}`

## 调用与耗时

| 指标 | 自主基线 | 固定双调用 |
|---|---:|---:|
| 专家调用总数 | 335 | 400 |
| 每会话调用分布 | `{2: 135, 1: 65}` | `{2: 200}` |
| 模型轮次分布 | `{3: 71, 4: 52, 2: 54, 5: 18, 6: 2, 7: 3}` | `{3: 159, 4: 40, 5: 1}` |
| 总耗时 | 6609.8 s | 4111.5 s |
| 平均每会话 | 33.05 s | 20.56 s |

## 重点组

| 基线组 | n | 固定 local Tampered | 固定 local 分布 | 固定 both |
|---|---:|---:|---|---:|
| tampered_agent_not_called | 14 | 1 | `{'Not tampered': 3, 'Uncertain': 10, 'Tampered': 1}` | 0 |
| tampered_agent_called_not_detected | 33 | 3 | `{'Not tampered': 12, 'Uncertain': 18, 'Tampered': 3}` | 1 |
| tampered_agent_detected | 13 | 2 | `{'Uncertain': 11, 'Tampered': 2}` | 2 |

直接回答：

- 原未调用 14 张篡改图：固定模式检出 local Tampered **1/14**。
- 原已调用未检出 33 张：固定模式仍漏检 **30/33**（检出 3/33）。
- 30 张 AutoSplice 真图：固定 local 分布 `{'Uncertain': 6, 'Not tampered': 23, 'Tampered': 1}`。
- 60 张篡改图：固定“整图 Real + local Tampered” **3/60**，自主基线 **5/60**。

## 代表案例

### tampered_agent_not_called

| sample_id | 基线 local/final | 固定 local/final | 固定 whole |
|---|---|---|---|
| autosplice__40591__1__q95 | Not applicable / Fake | Not tampered / Fake | AI-generated |
| autosplice__41859__0__q95 | Uncertain / Fake | Uncertain / Fake | AI-generated |
| autosplice__42357__2__q95 | Uncertain / Uncertain | Uncertain / Fake | AI-generated |
| autosplice__44492__0__q95 | Uncertain / Real | Uncertain / Uncertain | Real |
| autosplice__45563__2__q95 | Uncertain / Fake | Uncertain / Fake | AI-generated |

### tampered_agent_called_not_detected

| sample_id | 基线 local/final | 固定 local/final | 固定 whole |
|---|---|---|---|
| autosplice__39697__0__q95 | Not tampered / Fake | Not tampered / Fake | AI-generated |
| autosplice__40340__2__q95 | Not tampered / Fake | Not tampered / Fake | AI-generated |
| autosplice__41242__1__q95 | Uncertain / Uncertain | Uncertain / Real | Real |
| autosplice__42057__2__q95 | Not tampered / Uncertain | Not tampered / Real | Real |
| autosplice__42911__1__q95 | Uncertain / Fake | Tampered / Fake | AI-generated |

### tampered_agent_detected

| sample_id | 基线 local/final | 固定 local/final | 固定 whole |
|---|---|---|---|
| autosplice__41178__2__q95 | Tampered / Fake | Uncertain / Fake | Real |
| autosplice__42724__0__q95 | Tampered / Fake | Uncertain / Real | Real |
| autosplice__43680__1__q95 | Tampered / Fake | Uncertain / Uncertain | Real |
| autosplice__43821__1__q95 | Tampered / Fake | Uncertain / Fake | AI-generated |
| autosplice__45359__2__q95 | Tampered / Fake | Uncertain / Fake | AI-generated |

## 读法与边界

- 固定模式在前两轮直接返回协议合法的 `call_probe` / `call_trufor`，之后完全使用真实 Qwen；控制器、证据包、coverage/certainty 和结案规则未改。
- 与自主模式的差别不只是“是否调用”，还包括固定调用发生在 Qwen 首轮之前；Qwen 在结案轮仍能看到原图和两份证据，但没有自主模式的首轮视觉规划。因此本对照测量的是这套固定流程，不是把 TruFor 证据单独注入自主对话。
- 固定模式 200/200、400 次专家调用；未出现工具失败或运行错误。整体正确率相同不等于局部能力相同：固定模式把更多样本推向 Fake/Real，弃权减少，但局部 Tampered 反而从 13/60 降到 6/60，整图误报从 45/120 升到 51/120。
- 逐图 CSV：`comparison/per_sample.csv`；本报告为 `comparison/report.md`；原始记录见固定批次和 11/27 两批目录。
