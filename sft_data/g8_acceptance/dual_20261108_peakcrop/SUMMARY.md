# dual_20261108_peakcrop — automatic summary

> Descriptive only; not a human acceptance verdict. 24 sessions, same manifest.
> Qwen3-VL-8B-Instruct, greedy, `probe+trufor`; joint prompt digest
> `d179aa70a1f4cf91`; wall ~1030 s (13:38:15–13:55:25).

## Headline vs `dual_20261108_closing`

| 指标 | closing | peakcrop |
| --- | ---: | ---: |
| 整图判断正确 | 16 | 16 |
| 局部判断正确 | 12 | 11 |
| 两维都正确 | 7 | 8 |
| 局部误报 | 5 | 2 |
| joint_verdict_issues | 1 | 0 |
| 局部结论缺 TruFor | 1 | 0 |
| 未尝试专家 | 5 | 2 |
| 全图 bbox | 3 | 0 |
| 强相互验证措辞 | 3 | 1 |
| confidence=0.95 | 18 | 17 |
| 结案前检查 | 3 | 6 |
| 覆盖检查后重答 | 未记录 | 4 |
| 覆盖检查失败 | 未记录 | 0 |

## What changed in this batch

- No-bbox TruFor calls now receive an anomaly-peak inspection crop
  (`region_source=anomaly_peak`), not a full-frame view. Full-frame views do not
  expose a bbox-like `region_normalized_1000`.
- `35b3b70e` no longer writes local `Not tampered` without TruFor.
- `39890_0/1` correct the whole-image answer to `Uncertain` while keeping local
  `Tampered`; coverage correction preserved original/feedback/re-answer.
- Full-image bbox is 0; region review flags are 1 (`ADM` location/text).
- Local false positives drop to 2 (`51136`, `ADM`).

## Remaining

- confidence still high (17 × 0.95), not acceptable as calibrated certainty.
- `51136` local Tampered without a local region; `ADM`/`51136` false positives.
- One strong mutual-wording case (`ADM`) predates the clause-scope negation fix;
  next run should correct it.

## Decision

Human review first; no expansion, no SFT.
