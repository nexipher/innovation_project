# LoRA 冒烟数据审计

## 三条轨迹

| sample | 助手轮 | 工具轮 | 工具轮图片数 | 总图数 | 总 token | 监督轮 | 监督 token | 掩码 token |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `autosplice__55310__1__q95` | 3 | 2 | 3（TruFor 三图）+ 1（PROBE 裁剪） | 5 | 8853 | 3 | 606 | 8247 |
| `autosplice__65547__authentic__q95` | 3 | 2 | 3（TruFor 三图）+ 1（PROBE 裁剪） | 5 | 9019 | 3 | 686 | 8333 |
| `single_trufor__autosplice__54029__1__q95` | 2 | 1 | 3（TruFor 三图） | 4 | 7669 | 2 | 423 | 7246 |

逐轮监督 token（`[call1, call2, closing]`）：

- 55310：`[165, 153, 288]`
- 65547：`[177, 178, 331]`
- 54029：`[156, 267]`（call + closing）

## 轮次送达

- 原图：第一条 user turn 的 image block，三条各 1 张。
- TruFor 三图：紧跟 TruFor 调用的 tool/user turn，三图分别为
  `model_original_crop`、`model_anomaly_map`、`model_confidence_map`。
- PROBE 裁剪：紧跟 PROBE 调用的 tool/user turn，为 `region_...png`。
- 55310、65547 的调用顺序是 TruFor → PROBE；54029 是 TruFor only。

## Loss 覆盖

- `assistant_spans` 只监督 assistant 轮：两条调用轮 + 结案轮（55310/65547），
  调用轮 + 结案轮（54029）。
- 工具结果、系统状态、用户任务全部是上下文，labels 为 -100；监督 token 数与
  assistant 轮数一致，未把工具结果当训练目标。
- 监督 token 占全部 token 的 6.8%（55310）、7.6%（65547）、5.5%（54029）。

## 复现

数据审计由 `scripts/train_lora_candidate_smoke.py --dry-run` 生成；训练使用同一
`build_messages` 和 `assistant_spans`。原始候选轨迹见
`sft_data/sft_candidates_20261127/traces/`。
