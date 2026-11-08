# dual_20261108_closing — automatic summary

> Descriptive only; not a human acceptance verdict. 24 sessions, same manifest as
> `dual_20261107`. Qwen3-VL-8B-Instruct, greedy, `probe+trufor`; joint prompt digest
> `f1b2309a9dbd7dfa`; wall 847.1 s.

## Headline vs `dual_20261108` (previous rules)

| 指标 | 上一批 | 本批 |
| --- | ---: | ---: |
| 整图判断正确 | 11 | 16 |
| 局部判断正确 | 7 | 12 |
| 两维都正确 | 3 | 7 |
| expert call = 2 | 9 | 18 |
| 总体 Uncertain | 10 | 1 |
| confidence = 0.95 | 10 | 18 |
| 局部误报 | 3 | 5 |
| joint_verdict_issues | 0 | 1 |
| 仍有未尝试专家 | 未记录 | 5 |
| 结案前检查触发 | 未记录 | 3（3/3 后补测） |
| 弃权仍引用未尝试专家 | 未记录 | 0 |

## 原始五个停结案例

| 会话 | 上一批 | 本批 | 调用 |
| --- | --- | --- | --- |
| BigGAN | Uncertain/Uncertain 0.6 | AI-generated/Uncertain 0.95 | TruFor → PROBE |
| VQDM | Uncertain/Uncertain 0.3 | AI-generated/Not tampered 0.95 | TruFor → PROBE |
| 58610 | Uncertain/Uncertain 0.5 | Real/Tampered 0.95 | TruFor → PROBE |
| Midjourney | Uncertain/Uncertain 0.8 | AI-generated/Uncertain 0.95 | TruFor 失败 → PROBE |
| 41187_0 | Uncertain/Tampered 0.7 | Real/Tampered 0.95 | TruFor → PROBE |

The exact “PROBE 未调用，所以 Uncertain” pattern is gone (0 sessions cite an untried
expert in the adopted abstention). But BigGAN still uses one expert and gives a
definite whole-image label; its local question remains unmeasured.

## Remaining problems

- Confidence became more mechanical: 18 × 0.95, 13 of them wrong on at least one dimension.
- Local false positives rose 3→5: `51136`, `53059`, `58610`, plus `ADM`/`SD14`
  (`Not applicable` → Tampered).
- `35b3b70e` says local `Not tampered` without any TruFor measurement
  (`joint_verdict_issues=1`).
- Cross-task wording remains: `58610`, `SD14`, `57501` say “相互支持”;
  `39890_0` uses TruFor local evidence to conclude whole-image AI-generated.
- Full-image bbox still 3; `ADM` bbox is top-left while the text says “下方”.
- Three preclose checks (ADM, SD15, 72399_2) all led to the missing call; no
  “kept Uncertain without measuring” case.

## Decision

Human review before any further step. Do not expand to new source images and do not
train. The next targeted iteration should address confidence calibration,
single-expert definite conclusions, local Tampered on authentic images, and cross-task
wording; it should not simply force two experts.
