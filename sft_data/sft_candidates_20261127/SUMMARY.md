# 首轮 SFT 候选样板 — sft_candidates_20261127

- 候选会话：24/24 完成；失败 0；wall 60.1s。
- assistant：候选写手拟写（`sft_candidate_scripted`），不是 Qwen 原始输出。
- 工具：真实 PROBE + TruFor；配置 `6eb4980f50971d5d`；
  manifest `9d1524944c97ee65f7fff118eb73e27ca4c2822b411c8d041453959b85f195d9`。
- 审核入口：`index.html`；重点案例：`CANDIDATE_CASES.md`。

## 与真值对照（仅供审核，不是验收）

- 总体：18/24
- 整图维度：22/24
- 局部维度：19/24

| # | sample | 真值 | 候选 overall | whole | local | certainty | calls |
|---|---|---|---|---|---|---|---|
| 1 | `autosplice__54029__1__q95` | Fake | Fake | Real | Tampered | High | 2 |
| 2 | `autosplice__55310__1__q95` | Fake | Fake | Real | Tampered | High | 2 |
| 3 | `autosplice__45749__1__q95` | Fake | Fake | Real | Tampered | High | 2 |
| 4 | `autosplice__69868__0__q95` | Fake | Fake | Real | Tampered | Medium | 2 |
| 5 | `autosplice__48251__0__q95` | Fake | Fake | Real | Tampered | Medium | 2 |
| 6 | `autosplice__51186__authentic__q95` | Real | Real | Real | Not tampered | High | 2 |
| 7 | `autosplice__65547__authentic__q95` | Real | Uncertain | Real | Uncertain | None | 2 |
| 8 | `genimage__SD15__459_sdv5_00195__q95` | Fake | Fake | AI-generated | Not applicable | High | 2 |
| 9 | `autosplice__52769__authentic__q95` | Real | Uncertain | Real | Uncertain | None | 2 |
| 10 | `autosplice__57332__authentic__q95` | Real | Uncertain | Real | Uncertain | None | 2 |
| 11 | `autosplice__67006__authentic__q95` | Real | Uncertain | Uncertain | Uncertain | None | 2 |
| 12 | `autosplice__70684__authentic__q95` | Real | Uncertain | Real | Uncertain | None | 2 |
| 13 | `autosplice__45914__2__q95` | Fake | Fake | Real | Tampered | High | 2 |
| 14 | `autosplice__64223__2__q95` | Fake | Fake | Real | Tampered | Medium | 2 |
| 15 | `autosplice__72548__1__q95` | Fake | Fake | Real | Tampered | Medium | 2 |
| 16 | `genimage__ADM__921_adm_7__q95` | Fake | Fake | AI-generated | Not applicable | High | 2 |
| 17 | `genimage__BigGAN__545_biggan_00074__q95` | Fake | Fake | AI-generated | Not applicable | High | 2 |
| 18 | `genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95` | Fake | Fake | AI-generated | Not applicable | High | 2 |
| 19 | `genimage__Midjourney__229_midjourney_197__q95` | Fake | Fake | AI-generated | Not applicable | High | 2 |
| 20 | `genimage__Wukong__217_wukong_image88__q95` | Fake | Fake | AI-generated | Not applicable | High | 2 |
| 21 | `genimage__Real__96d68646-b0a2-450e-ac0a-a5a1b58aa6c8__q95` | Real | Real | Real | Not tampered | High | 2 |
| 22 | `genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95` | Real | Uncertain | Uncertain | Not tampered | None | 2 |
| 23 | `genimage__Real__b3cfabea-3cae-414b-a9e7-37f421542552__q95` | Real | Real | Real | Not tampered | High | 2 |
| 24 | `genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95` | Real | Real | Real | Not tampered | Medium | 2 |

## 运行与审核边界

- 24 条全部来自 split.json 的训练侧；同源变体同划分；未读取 test。
- 真值和 mask 只写在 record/页面/本汇总，不进助手上下文。
- 候选文本仍需人工审核；通过审核不等于进入训练，也不等于 final_v2 准入。
- 本轮不扩样、不训练、不改原 200/24 批次。
