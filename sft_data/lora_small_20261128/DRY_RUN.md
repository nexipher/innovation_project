# 24 条 LoRA 小试 — Qwen3-VL dry-run

使用 `scripts/train_lora_candidate_smoke.py --selection ... --dry-run`，
与训练完全相同的数据组装、processor、assistant span 和 loss 掩码。

| sample | tokens | images | images/tool turn | calls | supervised turns | supervised tokens |
|---|---:|---:|---|---|---:|---:|
| `genimage__Midjourney__229_midjourney_197__q95` | 11677 | 5 | [1, 3] | probe → trufor | 3 | 468 |
| `genimage__SD15__459_sdv5_00195__q95` | 9608 | 5 | [3, 1] | trufor → probe | 3 | 554 |
| `genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95` | 9420 | 5 | [3, 1] | trufor → probe | 3 | 560 |
| `genimage__Wukong__217_wukong_image88__q95` | 9171 | 5 | [1, 3] | probe → trufor | 3 | 512 |
| `autosplice__64223__2__q95` | 9057 | 5 | [3, 1] | trufor → probe | 3 | 634 |
| `genimage__Real__b3cfabea-3cae-414b-a9e7-37f421542552__q95` | 9055 | 5 | [3, 1] | trufor → probe | 3 | 471 |
| `autosplice__65547__authentic__q95` | 9019 | 5 | [3, 1] | trufor → probe | 3 | 686 |
| `autosplice__69868__0__q95` | 9005 | 5 | [3, 1] | trufor → probe | 3 | 623 |
| `autosplice__54029__1__q95` | 8972 | 5 | [3, 1] | trufor → probe | 3 | 635 |
| `autosplice__48251__0__q95` | 8958 | 5 | [1, 3] | probe → trufor | 3 | 611 |
| `autosplice__51186__authentic__q95` | 8917 | 5 | [3, 1] | trufor → probe | 3 | 615 |
| `genimage__Real__96d68646-b0a2-450e-ac0a-a5a1b58aa6c8__q95` | 8892 | 5 | [1, 3] | probe → trufor | 3 | 505 |
| `autosplice__45914__2__q95` | 8879 | 5 | [1, 3] | probe → trufor | 3 | 590 |
| `autosplice__57332__authentic__q95` | 8874 | 5 | [3, 1] | trufor → probe | 3 | 552 |
| `autosplice__70684__authentic__q95` | 8871 | 5 | [3, 1] | trufor → probe | 3 | 626 |
| `autosplice__72548__1__q95` | 8854 | 5 | [3, 1] | trufor → probe | 3 | 603 |
| `autosplice__55310__1__q95` | 8853 | 5 | [3, 1] | trufor → probe | 3 | 606 |
| `autosplice__45749__1__q95` | 8837 | 5 | [1, 3] | probe → trufor | 3 | 627 |
| `autosplice__52769__authentic__q95` | 8800 | 5 | [3, 1] | trufor → probe | 3 | 615 |
| `genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95` | 8757 | 5 | [1, 3] | probe → trufor | 3 | 538 |
| `genimage__ADM__921_adm_7__q95` | 8727 | 5 | [1, 3] | probe → trufor | 3 | 467 |
| `genimage__BigGAN__545_biggan_00074__q95` | 8669 | 5 | [1, 3] | probe → trufor | 3 | 467 |
| `single_trufor__autosplice__54029__1__q95` | 7669 | 4 | [3] | trufor | 2 | 423 |
| `single_probe__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95` | 6626 | 2 | [1] | probe | 2 | 324 |

## 极值

- 最长：`genimage__Midjourney__229_midjourney_197__q95` 11677 tokens。
- 图最多：`genimage__Midjourney__229_midjourney_197__q95` 5 图（22 条双调用样本均为 5 图）。
- 10 步 profiling 集：genimage__Midjourney__229_midjourney_197__q95, genimage__SD15__459_sdv5_00195__q95, genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95。
- 超过 max_length=10000：['genimage__Midjourney__229_midjourney_197__q95']。

## 边界

- dry-run 只加载 processor，不加载模型、不使用 GPU。
- 每条的原图在首 user turn；TruFor 三图在 TruFor 调用后的 tool turn；
  PROBE 裁剪在 PROBE 调用后的 turn；工具结果/系统状态 labels=-100。
