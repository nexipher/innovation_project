# 2026-11-28 局部漏检与结案修复：方法与复验入口

## 输入

- 基线：`f0ab9ce` 的 30 图验证（`sft_data/lora_small_20261128/validation/`）。
- 独立 TruFor：`scripts/analyze_trufor_validation.py` 对 15 张 AutoSplice
  validation 图（10 篡改 + 5 真图）重跑；输出 `analysis/report.md`、
  `analysis/per_sample.csv`、`analysis/analysis.json`、`analysis/trufor_maps/`。
- 重点 trace：41993_0、43686_1、57143_0、67065_0、73275_1、d3429f29、
  BigGAN 739、VQDM 00143、Midjourney 487。

## 修复

1. **三图证据呈现**（`experts/trufor.py::render_region_artifacts`，
   `state_machine/controller.py`）
   - 无 bbox 时不再只取异常峰值的半边长裁剪；改为异常图 `>=0.5` 的
     high-response 像素包围盒，加 5% 上下文。
   - 高响应像素少于 0.5% 时视为孤立小峰，回退旧 peak crop；完全没有高响应时
     full-frame。
   - 三图仍为原图/异常图/可靠性图同内容、同内容尺寸；TruFor 不重跑。
2. **补测提示**（`utils/coverage.py`）
   - `local_manipulation=Uncertain` 且 TruFor 可调用未测：要求先
     `<call_trufor>`；补测后高响应写 Tampered，低分/无高响应按整图状态写
     Not applicable/Not tampered。
   - `local_manipulation` 已是 definite 但 TruFor 未测：提示先补测；补测后
     高分且同区域高响应写 Tampered。
   - 两位专家都已测量时明确禁止重复补测；`local_tampered_candidate_only`
     明确“whole=AI-generated 时总体保持 Fake”，不再把局部改成 Uncertain 后
     误把总体改成 Uncertain。
   - `whole=AI-generated`、PROBE>=0.9、TruFor 低分/无高响应时，局部改
     Not applicable；PROBE 中段不触发，避免把 Real 的整图误判改写成
     Not applicable。
   - 允许一次**窄的第二纠正**：第一次纠正促成了新的专家测量，且第二次只剩
     `generated_whole_local_not_applicable` 这个标签关系问题时，才再提示一次；
     不做通用额外纠正预算。
   - 收紧 `local_tampered_candidate_only`：只有“仅提供/仅为候选”等明确表述、
     且没有 `suspicious_regions` 定位时触发；“非证明”这种谨慎措辞，或已经
     给出同区域 bbox/content 的 Tampered，不再被改成 Uncertain。
3. **复验工具**
   - `scripts/run_lora_validation.py` 记录 `coverage_check` 和
     `joint_verdict_issues`。
   - `scripts/compare_fix_validation.py`：30 图 before/after 指标、分项一致性、
     重点案例。
   - `scripts/compare_crop_coverage.py`：交付三图对 AutoSplice mask 的覆盖率。
   - `scripts/replay_same_evidence.py`：同一 clean evidence prefix（原图、
     各专家首次调用/工具返回/状态、三图）分别喂给原版和 LoRA，不执行工具。

## 三图覆盖（同一 TruFor map，CPU 复算）

10 张篡改图的编辑 mask 覆盖率，旧 peak crop → 新 high-response crop：

| sample | before | after | after source |
|---|---:|---:|---|
| 40489 | 0.265 | 0.378 | high-response |
| 41993 | 0.369 | 0.966 | high-response |
| 42958 | 0.594 | 1.000 | high-response |
| 43686 | 0.440 | 1.000 | high-response |
| 56086 | 0.679 | 0.768 | high-response |
| 57143 | 0.520 | 0.981 | high-response |
| 57276 | 0.305 | 0.305 | peak fallback |
| 60788 | 0.199 | 0.501 | high-response |
| 67065 | 未调用 | 1.000 | high-response |
| 73275 | 未调用 | 0.967 | high-response |

均值（有旧裁剪的 8 张）从 0.421 升到 0.787；57276 的高响应是远离 mask 的孤立小峰，
回退旧 peak crop。覆盖率只说明证据呈现，不等于模型判对。

## 复验命令

```bash
FORENSIC_ALLOW_GPU=1 python scripts/run_lora_validation.py --tag base \
  --manifest sft_data/lora_small_20261128/validation_manifest.json \
  --out sft_data/lora_small_20261128/validation_fixed
FORENSIC_ALLOW_GPU=1 python scripts/run_lora_validation.py --tag lora \
  --manifest sft_data/lora_small_20261128/validation_manifest.json \
  --adapter sft_data/lora_small_20261128/adapter \
  --out sft_data/lora_small_20261128/validation_fixed
python scripts/compare_fix_validation.py
python scripts/compare_crop_coverage.py
# 先由 base 轨迹生成 replay_cases.json（每条含 sample_id + trace），再分别跑：
FORENSIC_ALLOW_GPU=1 python scripts/replay_same_evidence.py \
  --selection sft_data/lora_small_20261128/replay_cases.json \
  --tag base --out sft_data/lora_small_20261128/replay_fixed
FORENSIC_ALLOW_GPU=1 python scripts/replay_same_evidence.py \
  --selection sft_data/lora_small_20261128/replay_cases.json \
  --tag lora --adapter sft_data/lora_small_20261128/adapter \
  --out sft_data/lora_small_20261128/replay_fixed
```

## 边界

- 未重训、未扩 250 条、未改专家权重/阈值、未碰 test。
- 30 图样本小；修复后的数字只说明这批案例的变化，不是最终评测。
