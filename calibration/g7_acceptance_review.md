# G7 入口 3 验收审阅单

- 模式：`gpu`｜样本 120｜批次源 24
- 汇总：{"samples": 120, "probe_called_in": 110, "probe_token_rate": 0.9167, "agreement_with_probe_bins_mapping": {"agent_abstained": 24, "agree": 94, "disagree": 2}, "agreement_with_probe_logistic": {"agent_abstained": 24, "agree": 94, "disagree": 2}, "the_two_references_agree": {"same": 116, "different": 4}, "agent_verdicts": {"Uncertain": 24, "Fake": 34, "Real": 62}, "halting_reasons": {"model_stalled": 25, "no_expected_gain": 25, "budget_exhausted": 4, "candidate_matches_posterior": 66}, "call_counts": {"probe": 110, "noise": 24, "jpeg": 26, "freq": 15}, "first_calls": {"probe": 108, "jpeg": 8, "noise": 4}, "landed_tokens_per_tag": {"probe": 110, "noise": 24, "jpeg": 26, "freq": 15}, "per_treatment": {"jpeg_q70": {"agent_Uncertain": 7, "agent_Real": 13, "agent_Fake": 4}, "jpeg_q85": {"agent_Fake": 9, "agent_Real": 12, "agent_Uncertain": 3}, "jpeg_q95": {"agent_Uncertain": 5, "agent_Fake": 7, "agent_Real": 12}, "native": {"agent_Uncertain": 5, "agent_Fake": 6, "agent_Real": 13}, "png": {"agent_Fake": 8, "agent_Real": 12, "agent_Uncertain": 4}}, "skipped_duplicate_calls": 154, "executed_calls": 175, "mean_seconds_per_sample": 12.86, "mean_model_turns": 3.5}

> 自动标记只说明「这一行值得人看」，不构成结论（D2 的教训：视觉预筛能看不能判，
> 文本层标记同理——它们负责把 120 条缩到该细看的那几条）。

## 自动标记（103 处）

- `duplicate_call`：103 处

## 逐会话

### fake_adm_0000_native ｜GT Fake｜native｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_adm_0000_png ｜GT Fake｜png｜B
- 判定 Fake（后验 0.8121）｜停止 no_expected_gain｜调用 ['probe', 'noise', 'jpeg']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.848，inverted:high-metric-means-real，一致性 None）
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_adm_0000_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Uncertain（后验 0.4749）｜停止 budget_exhausted｜调用 ['probe', 'freq', 'jpeg']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.449，weak:marginally-above-chance，一致性 None）
  - `jpeg_expert` → Uncertain（P(Fake)=0.449，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 7 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_adm_0000_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）

### fake_adm_0000_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_biggan_0007_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_biggan_0007_png ｜GT Fake｜png｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）

### fake_biggan_0007_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_biggan_0007_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_biggan_0007_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Uncertain（后验 0.4689）｜停止 model_stalled｜调用 ['jpeg']
  - `jpeg_expert` → Uncertain（P(Fake)=0.456，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_glide_0012_native ｜GT Fake｜native｜B
- 判定 Real（后验 0.1897）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_glide_0012_png ｜GT Fake｜png｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_glide_0012_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Real（后验 0.2386）｜停止 candidate_matches_posterior｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.641，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_glide_0012_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_glide_0012_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Real（后验 0.1979）｜停止 no_expected_gain｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `jpeg_expert` → Uncertain（P(Fake)=0.456，inverted:high-metric-means-real，一致性 None）

### fake_midjourney_0019_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_midjourney_0019_png ｜GT Fake｜png｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_midjourney_0019_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_midjourney_0019_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）

### fake_midjourney_0019_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_sd14_0027_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_sd14_0027_png ｜GT Fake｜png｜B
- 判定 Fake（后验 0.9112）｜停止 no_expected_gain｜调用 ['probe', 'freq', 'noise', 'jpeg']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - `frequency_expert_v2` → AI-generated（P(Fake)=0.709，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Uncertain（P(Fake)=0.564，inverted:high-metric-means-real，一致性 None）
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）

### fake_sd14_0027_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.7968）｜停止 no_expected_gain｜调用 ['probe', 'noise', 'jpeg']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - `noise_expert` → Uncertain（P(Fake)=0.564，inverted:high-metric-means-real，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.385，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 3 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_sd14_0027_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_sd14_0027_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Uncertain（后验 0.7232）｜停止 no_expected_gain｜调用 ['jpeg', 'probe', 'noise']
  - `jpeg_expert` → Real（P(Fake)=0.295，inverted:high-metric-means-real，一致性 None）
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - `noise_expert` → Uncertain（P(Fake)=0.564，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 3 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_sd15_0030_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）

### fake_sd15_0030_png ｜GT Fake｜png｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_sd15_0030_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）

### fake_sd15_0030_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）

### fake_sd15_0030_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_vqdm_0038_native ｜GT Fake｜native｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_vqdm_0038_png ｜GT Fake｜png｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_vqdm_0038_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.8512）｜停止 no_expected_gain｜调用 ['probe', 'noise']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.641，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_vqdm_0038_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_vqdm_0038_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_wukong_0043_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_wukong_0043_png ｜GT Fake｜png｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_wukong_0043_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_wukong_0043_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### fake_wukong_0043_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Uncertain（后验 0.6414）｜停止 no_expected_gain｜调用 ['probe', 'freq', 'noise', 'jpeg']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - `frequency_expert_v2` → Real（P(Fake)=0.392，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.282，inverted:high-metric-means-real，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.295，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0009_native ｜GT Real｜native｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0009_png ｜GT Real｜png｜B
- 判定 Real（后验 0.1297）｜停止 no_expected_gain｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.282，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0009_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.1297）｜停止 no_expected_gain｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.282，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0009_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.1158）｜停止 no_expected_gain｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.295，inverted:high-metric-means-real，一致性 None）

### real_0009_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.1264）｜停止 no_expected_gain｜调用 ['probe', 'freq', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `frequency_expert_v2` → Real（P(Fake)=0.392，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.282，inverted:high-metric-means-real，一致性 None）

### real_0015_native ｜GT Real｜native｜B
- 判定 Uncertain（后验 0.4273）｜停止 budget_exhausted｜调用 ['probe', 'freq', 'jpeg']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.41，weak:marginally-above-chance，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.385，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0015_png ｜GT Real｜png｜B
- 判定 Uncertain（后验 0.3723）｜停止 model_stalled｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.295，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0015_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Uncertain（后验 0.5082）｜停止 model_stalled｜调用 ['probe', 'freq']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.41，weak:marginally-above-chance，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0015_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.3173）｜停止 candidate_matches_posterior｜调用 ['probe', 'freq', 'noise']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.41，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0015_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0016_native ｜GT Real｜native｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0016_png ｜GT Real｜png｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0016_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.2184）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 3 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0016_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Uncertain（后验 0.3652）｜停止 budget_exhausted｜调用 ['probe', 'freq', 'jpeg']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `frequency_expert_v2` → Real（P(Fake)=0.392，weak:marginally-above-chance，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.295，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0016_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0021_native ｜GT Real｜native｜B
- 判定 Real（后验 0.1897）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 3 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0021_png ｜GT Real｜png｜B
- 判定 Real（后验 0.3102）｜停止 candidate_matches_posterior｜调用 ['noise']
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0021_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0021_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0021_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.1713）｜停止 no_expected_gain｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `jpeg_expert` → Uncertain（P(Fake)=0.456，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0024_native ｜GT Real｜native｜B
- 判定 Real（后验 0.1875）｜停止 candidate_matches_posterior｜调用 ['probe', 'freq']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.449，weak:marginally-above-chance，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0024_png ｜GT Real｜png｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0024_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.1897）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0024_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0024_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0025_native ｜GT Real｜native｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0025_png ｜GT Real｜png｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0025_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0025_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0025_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0026_native ｜GT Real｜native｜B
- 判定 Real（后验 0.1113）｜停止 no_expected_gain｜调用 ['probe', 'freq', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.487，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 3 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0026_png ｜GT Real｜png｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0026_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0026_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0026_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0042_native ｜GT Real｜native｜B
- 判定 Real（后验 0.0953）｜停止 no_expected_gain｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0042_png ｜GT Real｜png｜B
- 判定 Real（后验 0.1897）｜停止 budget_exhausted｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 6 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0042_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0042_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0042_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0043_native ｜GT Real｜native｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0043_png ｜GT Real｜png｜B
- 判定 Real（后验 0.3102）｜停止 candidate_matches_posterior｜调用 ['noise']
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0043_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0043_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0043_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0047_native ｜GT Real｜native｜B
- 判定 Real（后验 0.3102）｜停止 candidate_matches_posterior｜调用 ['noise']
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0047_png ｜GT Real｜png｜B
- 判定 Uncertain（后验 0.464）｜停止 model_stalled｜调用 ['jpeg']
  - `jpeg_expert` → Uncertain（P(Fake)=0.449，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0047_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.3102）｜停止 candidate_matches_posterior｜调用 ['noise']
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0047_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Uncertain（后验 0.4193）｜停止 model_stalled｜调用 ['jpeg']
  - `jpeg_expert` → Real（P(Fake)=0.385，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### real_0047_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.1087）｜停止 no_expected_gain｜调用 ['probe', 'freq', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `frequency_expert_v2` → Real（P(Fake)=0.392，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_real_0000_native ｜GT Real｜native｜val
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_real_0000_png ｜GT Real｜png｜val
- 判定 Real（后验 0.1117）｜停止 no_expected_gain｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）

### acc_real_0000_jpeg_q95 ｜GT Real｜jpeg_q95｜val
- 判定 Real（后验 0.1102）｜停止 no_expected_gain｜调用 ['probe', 'freq', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.449，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 3 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_real_0000_jpeg_q85 ｜GT Real｜jpeg_q85｜val
- 判定 Real（后验 0.1102）｜停止 no_expected_gain｜调用 ['probe', 'freq', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.449，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 3 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_real_0000_jpeg_q70 ｜GT Real｜jpeg_q70｜val
- 判定 Real（后验 0.1102）｜停止 no_expected_gain｜调用 ['probe', 'freq', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.449，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 3 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_real_0001_native ｜GT Real｜native｜val
- 判定 Uncertain（后验 0.3723）｜停止 model_stalled｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.295，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_real_0001_png ｜GT Real｜png｜val
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_real_0001_jpeg_q95 ｜GT Real｜jpeg_q95｜val
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_real_0001_jpeg_q85 ｜GT Real｜jpeg_q85｜val
- 判定 Real（后验 0.2435）｜停止 no_expected_gain｜调用 ['probe', 'freq', 'noise', 'jpeg']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.487，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.295，inverted:high-metric-means-real，一致性 None）

### acc_real_0001_jpeg_q70 ｜GT Real｜jpeg_q70｜val
- 判定 Real（后验 0.1979）｜停止 no_expected_gain｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `jpeg_expert` → Uncertain（P(Fake)=0.456，inverted:high-metric-means-real，一致性 None）

### acc_real_0002_native ｜GT Real｜native｜val
- 判定 Real（后验 0.1679）｜停止 no_expected_gain｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.385，inverted:high-metric-means-real，一致性 None）

### acc_real_0002_png ｜GT Real｜png｜val
- 判定 Real（后验 0.1948）｜停止 no_expected_gain｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `jpeg_expert` → Uncertain（P(Fake)=0.449，inverted:high-metric-means-real，一致性 None）

### acc_real_0002_jpeg_q95 ｜GT Real｜jpeg_q95｜val
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_real_0002_jpeg_q85 ｜GT Real｜jpeg_q85｜val
- 判定 Real（后验 0.2723）｜停止 candidate_matches_posterior｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.641，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_real_0002_jpeg_q70 ｜GT Real｜jpeg_q70｜val
- 判定 Real（后验 0.1979）｜停止 no_expected_gain｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `jpeg_expert` → Uncertain（P(Fake)=0.456，inverted:high-metric-means-real，一致性 None）

### acc_fake_adm_native ｜GT Fake｜native｜val
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_fake_adm_png ｜GT Fake｜png｜val
- 判定 Real（后验 0.2184）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_fake_adm_jpeg_q95 ｜GT Fake｜jpeg_q95｜val
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_fake_adm_jpeg_q85 ｜GT Fake｜jpeg_q85｜val
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_fake_adm_jpeg_q70 ｜GT Fake｜jpeg_q70｜val
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_fake_biggan_native ｜GT Fake｜native｜val
- 判定 Fake（后验 0.7354）｜停止 candidate_matches_posterior｜调用 ['jpeg']
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_fake_biggan_png ｜GT Fake｜png｜val
- 判定 Fake（后验 0.7354）｜停止 model_stalled｜调用 ['jpeg']
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_fake_biggan_jpeg_q95 ｜GT Fake｜jpeg_q95｜val
- 判定 Fake（后验 0.7871）｜停止 no_expected_gain｜调用 ['jpeg', 'probe']
  - `jpeg_expert` → Uncertain（P(Fake)=0.449，inverted:high-metric-means-real，一致性 None）
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_fake_biggan_jpeg_q85 ｜GT Fake｜jpeg_q85｜val
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）

### acc_fake_biggan_jpeg_q70 ｜GT Fake｜jpeg_q70｜val
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_fake_glide_native ｜GT Fake｜native｜val
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_fake_glide_png ｜GT Fake｜png｜val
- 判定 Fake（后验 0.7354）｜停止 candidate_matches_posterior｜调用 ['jpeg']
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_fake_glide_jpeg_q95 ｜GT Fake｜jpeg_q95｜val
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 2 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_fake_glide_jpeg_q85 ｜GT Fake｜jpeg_q85｜val
- 判定 Fake（后验 0.7466）｜停止 candidate_matches_posterior｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 5 次、执行后抑制 0 次（前者不耗预算，后者是兜底）

### acc_fake_glide_jpeg_q70 ｜GT Fake｜jpeg_q70｜val
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用：执行前跳过 1 次、执行后抑制 0 次（前者不耗预算，后者是兜底）
