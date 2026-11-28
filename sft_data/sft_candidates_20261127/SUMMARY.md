# 首轮 SFT 候选样板 — sft_candidates_20261127

- 候选会话：26/26 完成（24 条主样板 + 2 条单专家样板）；失败 0；wall 36.5s。
- assistant：候选写手拟写（`sft_candidate_scripted`），不是 Qwen 原始输出。
- 工具：真实 PROBE + TruFor；配置 `255ea663511f1f44`；
  manifest `c3692a105990144d833fb90f1beaa2768d443a9e1882af84fe09c2028256fd6a`。
- 审核入口：`index.html`；重点案例：`CANDIDATE_CASES.md`。
- 审核图片：`review_assets/`（逐条原图/源图/可用的专家图/叠加；
  页面使用批次内相对路径）。
- 收尾审计：`AUDIT_REPORT.md`（26/26 pass，0 问题）。
- 已确认判例与复核点：`REVIEW_QUESTIONS.md`。
- 判例修订清单：`REVISION_NOTES.md`（标签/置信度/调用理由逐项列出）。
- 文字补丁：`POST_RUN_PATCHES.md`（Midjourney 一条去掉未投递的 0.39 数字；
  不影响工具输出、证据 ID、框或标签）。

## 与真值对照（仅供审核，不是验收）

- 总体：21/26
- 整图维度：16/26
- 局部维度：22/26

| # | sample | 真值 | 候选 overall | whole | local | certainty | calls |
|---|---|---|---|---|---|---|---|
| 1 | `autosplice__54029__1__q95` | Fake | Fake | Uncertain | Tampered | High | 2 |
| 2 | `autosplice__55310__1__q95` | Fake | Fake | Real | Tampered | High | 2 |
| 3 | `autosplice__45749__1__q95` | Fake | Fake | Uncertain | Tampered | High | 2 |
| 4 | `autosplice__69868__0__q95` | Fake | Fake | Uncertain | Tampered | Medium | 2 |
| 5 | `autosplice__48251__0__q95` | Fake | Fake | Uncertain | Tampered | Medium | 2 |
| 6 | `autosplice__51186__authentic__q95` | Real | Real | Real | Not tampered | Medium | 2 |
| 7 | `autosplice__65547__authentic__q95` | Real | Uncertain | Real | Uncertain | None | 2 |
| 8 | `genimage__SD15__459_sdv5_00195__q95` | Fake | Fake | AI-generated | Not applicable | High | 2 |
| 9 | `single_probe__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95` | Fake | Fake | AI-generated | Not applicable | High | 1 |
| 10 | `single_trufor__autosplice__54029__1__q95` | Fake | Fake | Uncertain | Tampered | Medium | 1 |
| 11 | `autosplice__52769__authentic__q95` | Real | Uncertain | Real | Uncertain | None | 2 |
| 12 | `autosplice__57332__authentic__q95` | Real | Real | Real | Not tampered | Medium | 2 |
| 13 | `autosplice__67006__authentic__q95` | Real | Uncertain | Uncertain | Uncertain | None | 2 |
| 14 | `autosplice__70684__authentic__q95` | Real | Uncertain | Real | Uncertain | None | 2 |
| 15 | `autosplice__45914__2__q95` | Fake | Fake | Uncertain | Tampered | High | 2 |
| 16 | `autosplice__64223__2__q95` | Fake | Fake | Uncertain | Tampered | Medium | 2 |
| 17 | `autosplice__72548__1__q95` | Fake | Fake | Uncertain | Tampered | Medium | 2 |
| 18 | `genimage__ADM__921_adm_7__q95` | Fake | Fake | AI-generated | Not applicable | High | 2 |
| 19 | `genimage__BigGAN__545_biggan_00074__q95` | Fake | Fake | AI-generated | Not applicable | High | 2 |
| 20 | `genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95` | Fake | Fake | AI-generated | Not applicable | High | 2 |
| 21 | `genimage__Midjourney__229_midjourney_197__q95` | Fake | Fake | AI-generated | Not applicable | High | 2 |
| 22 | `genimage__Wukong__217_wukong_image88__q95` | Fake | Fake | AI-generated | Not applicable | High | 2 |
| 23 | `genimage__Real__96d68646-b0a2-450e-ac0a-a5a1b58aa6c8__q95` | Real | Real | Real | Not tampered | High | 2 |
| 24 | `genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95` | Real | Uncertain | Uncertain | Not tampered | None | 2 |
| 25 | `genimage__Real__b3cfabea-3cae-414b-a9e7-37f421542552__q95` | Real | Real | Real | Not tampered | High | 2 |
| 26 | `genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95` | Real | Real | Real | Not tampered | Medium | 2 |

## 运行与审核边界

- 26 条全部来自 split.json 的训练侧；同源变体同划分；未读取 test。
- 真值和 mask 只写在 record/页面/本汇总，不进助手上下文。
- 候选文本仍需人工审核；通过审核不等于进入训练，也不等于 final_v2 准入。
- 本轮不扩样、不训练、不改原 200/24 批次。
