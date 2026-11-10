# g8 200-source development evaluation — 结果、失败与下一步

- 日期：2026-11-10
- 批次：`sft_data/g8_dev_eval_20261110/`
- 状态：**development evaluation**；不是人工验收，不是 SFT 数据
- 会话：200/200 完成，0 条会话级失败
- 结论（一句话）：**整图生成维度可解释；局部编辑维度因 TruFor 131 次 CUDA OOM 基本无效；总体指标是在“局部测量缺失”条件下得到的，不能当作完整系统的最终性能。**

> 自动统计是描述性的，不是人工验收。Uncertain 不计为 Fake 正例；自动解释命中只是候选，
> 必须回到原对话的原句和轮次核对。

## 1. 批次身份与固定配置

| 项 | 值 |
|---|---|
| manifest SHA256 | `a4b319556ef075534359d825881734ba78d80b075f1e2e957e63468307dcfa7c` |
| sampling seed | `20261110` |
| 起始 commit | `9a7d9fa897029b3960180ccf38deb2723c5e747f`（worktree clean） |
| 运行时 config SHA256 | `06af5cd74a608152` |
| prompt actual SHA256 | `6b4756ca21bc7a63325236f2f5dea52bfb1100b8e7ffb480c20fdeded527a102` |
| 模型 | Qwen3-VL-8B-Instruct `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，bfloat16，non-thinking，greedy |
| 工具 | 显式 `probe,trufor`；certainty_v1；halting policy v3 |
| PROBE 权重 SHA256 | `c8caa9bad54c029e4d22e1909c4673e46a239d8419a986e0ab461d423a450156` |
| TruFor 权重 SHA256 | `ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844` |
| 运行时间 | 2026-11-10 21:34:38 – 23:38:15（+08:00），约 2 h 04 m |
| 单会话 | mean 37.1 s，median 38.4 s，min 13.3 s，max 130.1 s |

样本组成：GenImage Real 30；AutoSplice Authentic 30；ADM / BigGAN / Glide / Midjourney /
SD14 / SD15 / VQDM / Wukong 各 10；AutoSplice Forged 60。共 200 张 q95 输入、60 个共享 mask。
真值：whole Real 120 / AI-generated 80；local Not tampered 60 / Not applicable 80 / Tampered 60。

## 2. 总体分类

| 真值类别 | n | 正确 | 错误 | 弃权 | 未形成结论 | 准确率 | 覆盖率 | 已结案准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 真实图（Real） | 60 | 21 | 9 | 30 | 0 | 0.3500 | 0.5000 | 0.7000 |
| 整图生成 | 80 | 67 | 4 | 9 | 0 | 0.8375 | 0.8875 | 0.9437 |
| 局部篡改 | 60 | 37 | 5 | 18 | 0 | 0.6167 | 0.7000 | 0.8810 |
| **合计** | **200** | **125** | **18** | **57** | **0** | **0.6250** | **0.7150** | **0.8741** |

- 平衡准确率（三类 recall 均值）：**0.6014**。
- Fake 类（整图生成 + 局部篡改）：precision **0.9204**，recall **0.7429**，F1 **0.8222**；
  Uncertain 不计入正例。
- **关键警告**：局部篡改类的 37 个“总体正确（Fake）”**全部**由
  `whole_image_generation=AI-generated` 驱动；`local_manipulation=Tampered` 为 **0**。
  因此 0.6167 不是局部检出率，只是“PROBE 把 AutoSplice 判成整图生成”的副产品。
- 真实图 30 条弃权中，30 条为 AutoSplice Authentic；其总体 Uncertain 主要因为
  TruFor 失败/未测，无法回答局部问题。总体 Real 的 9 条错误全部是 PROBE 高分导致。

## 3. 整图生成维度

| 真值 | n | 正确 | 准确率 | 确定测量覆盖 | Uncertain | 缺失/非法 |
|---|---:|---:|---:|---:|---:|---:|
| Real | 120 | 66 | 0.5500 | 0.9500 | 6 | 0 |
| AI-generated | 80 | 67 | 0.8375 | 0.9125 | 7 | 0 |

- AI-generated precision **0.5826**，recall **0.8375**，F1 **0.6872**。
- 48/120 Real 被 PROBE 判为 AI-generated；其中 AutoSplice 46 条（Authentic 8、Forged 38），
  GenImage Real 2 条。这是总体错误的最主要来源。

| 组 | n | 整图正确 | Uncertain | 主要错误 |
|---|---:|---:|---:|---|
| GenImage Real | 30 | 28 | 0 | 2 条 PROBE 高分误报 |
| ADM | 10 | 7 | 3 | — |
| BigGAN | 10 | 8 | 0 | 2 条低分漏检 |
| Glide | 10 | 5 | 1 | 4 条低分漏检 |
| Midjourney | 10 | 9 | 1 | — |
| SD14 | 10 | 10 | 0 | — |
| SD15 | 10 | 10 | 0 | — |
| VQDM | 10 | 8 | 2 | — |
| Wukong | 10 | 10 | 0 | — |
| AutoSplice Authentic | 30 | 21 | 1 | 8 条 PROBE 误报 |
| AutoSplice Forged | 60 | 17 | 5 | 38 条 PROBE 误报 |

整图维度结论：GenImage 的 SD14/SD15/Wukong 很稳，Midjourney 9/10；Glide 5/10、BigGAN 8/10
较弱。AutoSplice 的低分辨率/重编码真实照片出现明显 domain shift：46/90 被判 AI-generated，
远高于 GenImage Real 的 2/30。该问题与 TruFor 无关，是本批最可靠的发现之一。

## 4. 局部编辑维度（AutoSplice，TruFor）

| 真值 | n | Tampered | Not tampered | Uncertain | Not applicable | 缺失 |
|---|---:|---:|---:|---:|---:|---:|
| 真实篡改 | 60 | **0** | 0 | 58 | 2 | 0 |
| 真实真图 | 30 | 0 | **0** | 30 | 0 | 0 |

- 图像级局部：TP=0，FP=0，precision 未定义，recall **0.0000**，F1 未定义；弃权 88；
  未调用 TruFor 18（伪造 14、真图 4）；TruFor 失败 72（伪造 46、真图 26）。
- **60 条真实篡改没有任何一条得到 TruFor 有效测量**；因此局部召回 0 不是模型行为，
  而是测量缺失。
- 2 条真实篡改被写成 `Not applicable`（`autosplice__40591__1`、`autosplice__70419__1`）：
  PROBE 误报整图 AI-generated，模型据此把局部问题当作不适用；按 AutoSplice 真值这是错误，
  也说明 `Not applicable` 规则在“整图误报”下会被滥用。
- GenImage 纯生成图 local 输出：`Uncertain` 70、`Not tampered` 8、`Not applicable` 2。
  真值为 Not applicable，按该口径单独评价，不混入 AutoSplice 局部 P/R；8 条 Not tampered
  是语义不一致，70 条 Uncertain 是保守未结案。
- GenImage Real local 输出：`Not tampered` 5、`Uncertain` 25。
- 区域定位：60 个伪造源都有共享 mask，但输出 bbox 为 **0**；bbox 中心落 mask、IoU、
  mask coverage/precision 全部无法计算。29 个选中 forged base 有多个变体，本批每 base
  只取一个输入；mask 是 base 级共享 mask，不是逐变体 mask。定位质量本批不可评价。

TruFor 状态（200 会话）：17 条有效、129 条至少一次失败（131 次失败调用，2 条会话重复失败）、
54 条从未调用。17 条有效全部出现在 21:35–22:08，且全部属于 GenImage Real/ADM/BigGAN；
AutoSplice 90 条中 0 条有效。

## 5. 调用行为

| 指标 | 值 |
|---|---:|
| 首工具 TruFor / PROBE | 123 / 77 |
| 模型输出中的 call 标签（请求） | 719 |
| 实际执行 | 339 |
| 有效测量 | 208 |
| 失败调用 | 131 |
| 重复请求 | 2 |
| 结案前补测会话 | 107（101 次补测，6 次仍结案） |
| 覆盖纠正通过 / 失败 | 30 / 2 |
| certainty 纠正通过 / 失败 | 47 / 1 |
| 单专家会话（仅 1 个有效来源） | 176 |
| 弃权仍引用未尝试专家 | 1（计数受 failed/untried 记录问题影响，见下） |
| 初始“确定分项但无测量”越界 | 8（全部被覆盖纠正；最终 0） |

- 单专家 176 = 175 条仅有 PROBE + 1 条仅有 TruFor。175 条 PROBE-only 中，129 条是
  TruFor 失败（外部测量失败），54 条是模型从未尝试 TruFor；1 条 TruFor-only 是
  `genimage__ADM__391_adm_153`（总体 Uncertain，PROBE 未测）。
- **8 条“无任何有效测量”的弃权**（`66`、`92`、`99`、`127`、`128`、`155`、`165`、`171`）：
  只调用了 TruFor，TruFor 失败后没有调用仍可用的 PROBE，直接输出 Uncertain。
  它们有预算、没有 preclose 检查；属于“可用工具未尝试的提前结案”。其中 `99` 重复失败两次。
- 107 次结案前检查中，101 次触发补测，6 次模型说明理由后仍结案；但 8 条上述弃权没有
  触发检查，因为模型只写“整图生成未测量”，没有点名 PROBE，`_cited_untried_experts`
  没有匹配。另有 1 条 TruFor-only 也没有触发。
- 初始 8 条“确定分项无测量”越界（7 条 local 无 TruFor、1 条 whole 无 PROBE）全部被
  一次覆盖纠正修好；最终 `definite_without_measurement=0`。
- 2 条覆盖纠正失败、1 条 certainty 纠正失败，见第 8 节。

## 6. 解释质量候选

- 三图会话 17；layout 记录 17；框核验标记 0。仅这 17 条有 TruFor map，可靠性图/三图
  对应关系无法覆盖全批。
- 跨任务“相互验证/共同支持”自动候选 1 条（`autosplice__47410__1`，turn 4），但原句是
  “PROBE 高分与视觉特征共同支持整图 AI 生成”，不是 PROBE 与 TruFor 相互验证；
  按“两者测量不同问题”的规则，这是**自动误报**。最终覆盖检查没有命中真正的跨任务措辞。
- 具体编辑过程断言自动候选 44 条，抽查显示绝大多数是假设、否定或规划句：
  “可能暗示后期添加”“未见后期编辑痕迹”“排除后期合成或局部修改的可能性”。
  自动扫描对否定/假设/规划的作用域过宽；**不能把 44 当作 44 个缺陷**，需要人工按原句核对。
- 可靠性误读候选 0；但只有 17 条 map，不能据此声称可靠性理解已通过。
- 最终 `joint_verdict_issues` 非空 2 条（`autosplice__42357__2`、`autosplice__62011__authentic`），
  都是“总体 Uncertain + whole AI-generated”在一次纠正后仍未修好；见第 8 节。

## 7. Certainty

| certainty | n | 总体正确 | 总体准确率 | 整图正确 | 局部正确 |
|---|---:|---:|---:|---:|---:|
| High | 108 | 98 | 0.9074 | 77 | 7 |
| Medium | 27 | 20 | 0.7407 | 13 | 0 |
| Low | 8 | 7 | 0.8750 | 3 | 0 |
| null | 57 | 0 | — | 40 | 0 |

- High 占已结案（Real/Fake）的 **75.5%**（108/143）；Medium 27、Low 8。High 比例偏高。
- High 的总体准确率 90.7%，但其中 74 条报告同时写了“未校准/候选/缺失/失败/无法确认”
  等限制（自动候选）。这些不自动等于违规（PROBE 未校准几乎每条都写），但需要人工判断
  High 是否与“局部测量失败/未测”这一主要限制并存。
- null 的 57 条全是 Uncertain；表中“总体准确率 0”是定义上的（没有正例标签），不能当作
  57 条错误。
- Low 只有 8 条且准确率 0.875，高于 Medium 的 0.741；分档单调性不成立，样本小，
  不能据此校准。

## 8. 失败分析：TruFor CUDA OOM

### 8.1 现象

- 131 次 TruFor 调用失败，错误统一为
  `RuntimeError: trufor_expert：未产出 npz（退出码 0）`；trace 的 stderr 尾部明确为
  `RuntimeError: CUDA error: out of memory`（TruFor 子进程）。
- 17 条有效 TruFor 全部集中在 21:35–22:08；22:08 之后 **没有一条成功**。失败最早出现在
  index 6，早期是间歇性的（与图像尺寸/当时的显存余量有关），后期变为连续失败。
- 有效结果分布：GenImage Real 6/30、ADM 4/10、BigGAN 7/10；AutoSplice 0/90。
  `trufor_raw` 只有 17 个 `result.npz`，与有效记录数一致。
- 200 条会话记录都写成 `status=completed`，没有会话级异常；失败保存在 evidence token
  `status=failed` 和 trace metadata 中，没有覆盖任何原始结果。

### 8.2 根因判断

Qwen3-VL-8B 在 GPU 上常驻占用约 **17.53 GB**（RTX 4090 共 24.56 GB）。TruFor 按每次调用
启动一个子进程；子进程在加载模型并做卷积时需要额外数 GB。随着会话推进，父进程的
generation/PyTorch 缓存保留块增长，留给子进程的连续显存不足，子进程在 conv 处 OOM；
TruFor `test.py` 用裸 `except` 吞掉异常并以退出码 0 结束，所以外层只看到“未产出 npz”。

辅助证据：在 Qwen 已加载但尚未 generation 的探针中（`torch.cuda` 分配 17.53 GB），
连续 10 次 TruFor 实际调用全部成功（每次约 6 s）；说明基础占用本身够用，问题与
generation 后的缓存/碎片或会话累积有关，而不是 TruFor 权重或图像格式错误。

### 8.3 影响边界

- **整图维度**：PROBE 是独立测量，131 次 TruFor 失败不改变 PROBE 分数；整图分项统计可解释。
- **总体维度**：模型在局部测量缺失时仍会结案；总体标签受“缺失条件下如何综合”的影响，
  不能当作完整双专家系统的总体性能。
- **局部维度**：60 条真实篡改 0 条有效 TruFor，局部召回/误报/定位均不可评价。
- **解释/三图/可靠性**：只有 17 条有 map，不能评价全批。
- 本批原始记录、trace、页面、mask 均保留；没有重跑或覆盖。

### 8.4 未完成的修复验证

因为用户要求停止继续实验，以下只是待验证方案，不是已验证结论：

1. 在 `TruForExpert._score` 启动子进程前调用 `torch.cuda.empty_cache()`（必要时
   `gc.collect()`），并在每个会话结束后清理父进程缓存；
2. 或让 TruFor 使用常驻 worker/单独进程池，避免反复创建 CUDA context；
3. 或把 TruFor 放到 CPU/第二张 GPU（会改变运行时配置，需单独记录）；
4. 下一轮先用 10–20 条 AutoSplice 连续会话验证 100% TruFor 成功，再考虑 200 条重跑。

## 9. 代表案例

| 页面 | 案例 | 说明 |
|---|---|---|
| `010__genimage__Real__8c58ba72-...html` | GenImage Real → Fake | PROBE 0.5648、TruFor 0.3111，仍判整图 AI；High |
| `030__genimage__Real__f45d5949-...html` | GenImage Real → Fake | PROBE 0.5182 刚过阈值，整图 AI；High |
| `185__autosplice__58184__authentic__q95.html` | AutoSplice 真图 → Fake | PROBE 0.9233 误报；TruFor 失败；High |
| `173__autosplice__44758__authentic__q95.html` | AutoSplice 真图 → Fake | PROBE 0.6560 误报；Medium |
| `123__autosplice__44492__0__q95.html` | 真实局部篡改 → Real | PROBE 0.0128，TruFor 失败，未测局部；High |
| `169__autosplice__73258__0__q95.html` | 真实局部篡改 → Real | PROBE 0.0085，TruFor 失败；High |
| `113__autosplice__40591__1__q95.html` | 真实局部篡改 → Not applicable | PROBE 0.99 误报整图 AI，局部被当作不适用；High |
| `164__autosplice__70419__1__q95.html` | 真实局部篡改 → Not applicable | PROBE 0.742 误报，局部不适用；High |
| `127__autosplice__46499__1__q95.html` | TruFor 失败后未调 PROBE | 只测 TruFor，失败即 Uncertain；可用 PROBE 未尝试 |
| `066__genimage__Midjourney__452_...html` | TruFor 失败后未调 PROBE | 同上，整图问题也未测 |
| `118__autosplice__42357__2__q95.html` | 总体 Uncertain + whole AI | 覆盖纠正失败；certainty 也失败；最终 joint issues 非空 |
| `192__autosplice__62011__authentic__q95.html` | 总体 Uncertain + whole AI | 覆盖纠正失败；最终 joint issues 非空 |
| `051__genimage__Glide__...glide_00039.html` | 生成图 → Real | PROBE 0.4107 低分漏检；Medium |
| `047__genimage__BigGAN__917_biggan_00020.html` | 生成图 → Real | PROBE 低分漏检；High |

## 10. 审核队列与工作台

- 工作台：`sft_data/g8_dev_eval_20261110/index.html`
- 导出：`records.json`、`records.jsonl`、`review_records.csv`
- 自动报告：`SUMMARY.md`、`report.json`
- 审核队列：`review_queue.json`（156 条）、`REVIEW_GUIDE.md`
- 队列构成：129 条工具失败、57 条弃权、18 条错误、32 条覆盖纠正、48 条 certainty 纠正、
  20 条固定随机“总体正确且 High”（其中 9 条仅因该抽样入队）。
- 优先顺序：18 条错误 → 2 条最终一致性问题 → 8 条无测量弃权 → 129 条工具失败 →
  32/48 条纠正 → 20 条随机 High。自动解释命中只是候选，必须看原句/轮次。
- 每条页面包含原始对话、实际投送图、三图（若有 TruFor map）、预测框叠加（本批为 0）、
  专家结果、纠正过程和停止原因。

## 11. 下一步建议

1. **先修测量，再谈局部能力**：按 §8.4 验证 TruFor 显存修复；在 10–20 条连续 AutoSplice
   会话上要求 TruFor 100% 有效，再决定是否重跑 200 条。修复前不要用本批局部指标做结论。
2. **处理 PROBE 的 AutoSplice domain shift**：46/90 误报 AI 是总体错误的主因；按
   AutoSplice/low-res/q95 分层评估 PROBE，考虑条件阈值或明确标注“该条件下 PROBE 不可靠”。
   调参只能用 dev/val，不能碰 test。
3. **收紧 Not applicable**：真实照片的局部编辑被 PROBE 误报整图 AI 后，模型会把局部写成
   Not applicable。建议要求“整图 AI-generated 的确定证据 + 局部无 Tampered 证据”才允许
   Not applicable；否则用 Uncertain，避免把“整图误报”升级成“局部不适用”。
4. **补提前结案门**：8 条无测量弃权 + 1 条 TruFor-only 说明“整图未测量”这类泛称没有触发
   preclose 检查。应扩展 `_cited_untried_experts` 匹配“整图/局部未测量/未调用”，在仍有
   可用工具且预算允许时至少提示一次；是否强制调用仍由模型决定，但必须留下理由。
5. **最终一致性硬门**：2 条“总体 Uncertain + whole AI-generated”在一次纠正后仍失败。
   建议把这类情况在 final_v2/审核门中直接标记为需人工复核或拒收，不要作为可用结论。
   certainty 为 Uncertain 写非 null 仍有 48 次，需要更强的 schema 约束或第二次纠正。
6. **High 校准**：75.5% 已结案为 High，74 条 High 报告含限制措辞。建议人工抽查 High 中
   “局部测量失败/未测”的样本，判断是否应降为 Medium；不要机械按分数改 certainty。
7. **解释扫描器降噪**：44 条 edit-process 候选绝大多数是假设/否定/规划；跨任务 mutual
   需要同时出现 PROBE 与 TruFor 测量才命中。否则审核队列会被噪声淹没。
8. **人工审核范围**：先看 18 条错误、2 条最终不一致、8 条无测量弃权；再按队列看工具失败
   与纠正；最后 20 条随机 High。不要用总体 62.5% 单独判断系统，因为类别不平衡且局部
   测量缺失。
9. **边界不变**：不训练/SFT，不扩到 test，不覆盖本批原始结果；本批只用于问题汇总和
   下一轮决策。原 24 条控制与历史批次保持不变。

## 12. 附：本批不能支持的说法

- 不能说“局部篡改召回 0”是模型能力：60 条真实篡改没有一条有效 TruFor。
- 不能说“总体准确率 62.5%”是完整系统性能：57 条弃权中 30 条来自局部测量缺失，
  且局部篡改类的 37 个 Fake 全部由整图误报驱动。
- 不能说“三图/可靠性理解已通过”：只有 17 条 map，且没有 bbox。
- 不能说“自动解释候选 44 条就是 44 个缺陷”：候选扫描未处理否定/假设/规划。
- 不能把本批与旧 confidence 批次直接比较 certainty 分布；语义和样本都不同。
