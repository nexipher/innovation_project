# b2 人工审核导出（只读原始产物）

`review_g8_b2.json` 是审核工作台导出的原始人工审核文件，生成批次 `g8/b2`，
审核人 `hj`，导出时间 `2026-10-18T13:01:11`。

| 字段 | 值 |
| --- | --- |
| schema | `g8_review_decisions/1` |
| 文件大小 | 12,218 B |
| sha256 | `0564b83b472ceb31e1fca3b799524ef194ded34526bf75890f53bb4c5d21a926` |
| 决定数 | 32 |
| 结果 | **27 revise / 5 pass** |
| 内容指纹 | 32 条均带 `fingerprint`；`g8_review_import.py --dry-run` 结果为 32 接受、0 拒收、0 过期 |

## 处置

- 本文件作为**原始审核证据**纳入 Git 跟踪，内容不应改写、重排或自动补写。
- 当前**未导入** `dispositions.jsonl`，b2 仍为未通过验收、未冻结；外部 Agent
  门禁保持关闭。不得把 5 条 pass 当成整批通过。
- 如需导入，先运行：
  `python scripts/g8_review_import.py sft_data/g8/review/b2/review_g8_b2.json --generation g8 --batch b2 --dry-run`
  确认逐条指纹后再显式导入。导入只写验收台账，不改训练产物，也不等于冻结。
- 审核中确认的问题与分类见 `plan.md` §4.16；本文件不被任何运行时代码读取。
