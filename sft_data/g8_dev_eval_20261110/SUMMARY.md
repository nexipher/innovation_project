# g8 200-source development evaluation — automatic report

- batch: `/root/autodl-tmp/innovation_project/sft_data/g8_dev_eval_20261110`
- manifest seed: `20261110`；sessions: 200/200
- manifest sha256: `a4b319556ef075534359d825881734ba78d80b075f1e2e957e63468307dcfa7c`

> 自动结果描述性报告，不是人工验收；Uncertain 不计为 Fake 正例。

## 1. 总体分类（按真实类别）

| 真值类别 | n | 正确 | 错误 | 弃权 | 未形成结论 | 准确率 | 覆盖率 | 已结案准确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 真实图（Real） | 60 | 21 | 9 | 30 | 0 | 0.3500 | 0.5000 | 0.7000 |
| 整图生成 | 80 | 67 | 4 | 9 | 0 | 0.8375 | 0.8875 | 0.9437 |
| 局部篡改 | 60 | 37 | 5 | 18 | 0 | 0.6167 | 0.7000 | 0.8810 |

- 总体准确率：**0.6250**（125/200）
- 平衡准确率（三类 recall 均值）：**0.6014**
- 覆盖率：**0.7150**；已结案样本准确率：**0.8741**
- Fake 类（整图生成 + 局部篡改）precision=0.9204，recall=0.7429，F1=0.8222；Uncertain 不计入 TP/FP/FN 的正例。

## 2. 整图生成分项

| 真值 | n | 正确 | 准确率 | 确定测量覆盖 | Uncertain | 缺失/非法 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real | 120 | 66 | 0.5500 | 0.9500 | 6 | 0 |
| AI-generated | 80 | 67 | 0.8375 | 0.9125 | 7 | 0 |

- AI-generated precision=0.5826，recall=0.8375，F1=0.6872。

| GenImage 组 | n | 整图正确 | Uncertain |
| --- | ---: | ---: | ---: |
| Real | 30 | 28 | 0 |
| ADM | 10 | 7 | 3 |
| BigGAN | 10 | 8 | 0 |
| Glide | 10 | 5 | 1 |
| Midjourney | 10 | 9 | 1 |
| SD14 | 10 | 10 | 0 |
| SD15 | 10 | 10 | 0 |
| VQDM | 10 | 8 | 2 |
| Wukong | 10 | 10 | 0 |

## 3. 局部编辑分项（AutoSplice）

| 真值 | n | Tampered | Not tampered | Uncertain | Not applicable | 缺失 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 真实篡改 | 60 | 0 | 0 | 58 | 2 | 0 |
| 真实真图 | 30 | 0 | 0 | 30 | 0 | 0 |

- 图像级局部：precision=—，recall=0.0000，F1=—；弃权 88；未调用 TruFor 18；TruFor 失败 72。
- GenImage 纯生成图 local 输出：{'Uncertain': 70, 'Not tampered': 8, 'Not applicable': 2}；真值为 Not applicable，按该口径单独评价，不混入 AutoSplice 局部 P/R。
- GenImage Real local 输出：{'Not tampered': 5, 'Uncertain': 25}。

### 区域定位（AutoSplice，base 共享 mask）

- 有 mask 的伪造源：60；输出 bbox：0（0.0000）。
- bbox 中心落 mask：0；IoU≥0.1/0.3/0.5：0/0/0；mean best IoU=—，mean mask coverage=—。
- 共享 mask 的伪造源 60 个；同 base 多 variant（本批只选一个输入）29 个。

## 4. 调用行为

- 首个工具：{'trufor_expert': 123, 'probe_dino_expert': 77}
- 请求调用 719，实际执行 339，有效测量 208，失败 131，重复请求 2。
- 结案前补测 107；覆盖纠正 30（失败 2）；certainty 纠正 47（失败 1）。
- 单专家会话 176；弃权仍引用未尝试专家 1；未测分项越界 8。

## 5. 解释质量候选

> 自动命中只是候选；`review_queue.json`/`REVIEW_GUIDE.md` 给出原句和轮次，需要人工按原对话核对。
- 三图会话 17；layout 记录 17；框核验标记会话 0。
- 跨任务相互验证候选 1；编辑过程断言候选 44；可靠性可能误读候选 0。

## 6. Certainty

- 分布：{'High': 108, 'null': 57, 'Medium': 27, 'Low': 8}

| certainty | n | 总体正确 | 总体准确率 | 整图正确 | 局部正确 |
| --- | ---: | ---: | ---: | ---: | ---: |
| High | 108 | 98 | 0.9074 | 77 | 7 |
| Medium | 27 | 20 | 0.7407 | 13 | 0 |
| Low | 8 | 7 | 0.8750 | 3 | 0 |
| null | 57 | 0 | 0.0000 | 40 | 0 |

- High 且报告含“候选/未校准/缺失/失败/无法确认”等限制的候选：74 条（见 review queue）。

## 7. 失败与复核入口

- 缺失记录：0；额外记录：0；运行失败记录：0。
- 优先复核：错误、弃权、未形成结论、工具失败、纠正案例，再固定随机抽查 20 条总体正确且 High 的会话。
- `index.html`：原始对话、投送图片、三图、预测框、专家结果、纠正、停止原因。
- `review_records.csv` / `records.json`：可导出审核记录。
