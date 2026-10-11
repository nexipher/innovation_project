# G7 入口 3 验收审阅单

- 模式：`gpu`｜样本 120｜批次源 24
- 汇总：{"samples": 120, "probe_called_in": 100, "probe_token_rate": 0.8333, "agreement_with_probe_bins_mapping": {"agent_abstained": 25, "agree": 91, "disagree": 4}, "agreement_with_probe_logistic": {"agent_abstained": 25, "agree": 90, "disagree": 5}, "the_two_references_agree": {"same": 116, "different": 4}, "agent_verdicts": {"Uncertain": 25, "Fake": 37, "Real": 58}, "halting_reasons": {"model_stalled": 23, "candidate_matches_posterior": 68, "no_expected_gain": 15, "budget_exhausted": 14}, "call_counts": {"probe": 100, "noise": 20, "jpeg": 32, "freq": 9}, "first_calls": {"probe": 97, "jpeg": 19, "noise": 4}, "landed_tokens_per_tag": {"probe": 100, "noise": 20, "jpeg": 32, "freq": 9}, "per_treatment": {"jpeg_q70": {"agent_Uncertain": 8, "agent_Real": 12, "agent_Fake": 4}, "jpeg_q85": {"agent_Fake": 9, "agent_Real": 10, "agent_Uncertain": 5}, "jpeg_q95": {"agent_Uncertain": 4, "agent_Fake": 8, "agent_Real": 12}, "native": {"agent_Uncertain": 3, "agent_Fake": 9, "agent_Real": 12}, "png": {"agent_Uncertain": 5, "agent_Fake": 7, "agent_Real": 12}}, "mean_seconds_per_sample": 11.66, "mean_model_turns": 3.48}

> 自动标记只说明「这一行值得人看」，不构成结论（D2 的教训：视觉预筛能看不能判，
> 文本层标记同理——它们负责把 120 条缩到该细看的那几条）。

## 自动标记（112 处）

- `duplicate_call`：108 处
- `score_as_artifact`：4 处

## 逐会话

### fake_adm_0000_native ｜GT Fake｜native｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_adm_0000_png ｜GT Fake｜png｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_adm_0000_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_adm_0000_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_adm_0000_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Uncertain（后验 0.5662）｜停止 no_expected_gain｜调用 ['probe', 'noise', 'jpeg']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.641，inverted:high-metric-means-real，一致性 None）
  - `jpeg_expert` → Uncertain（P(Fake)=0.456，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_biggan_0007_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.7354）｜停止 candidate_matches_posterior｜调用 ['jpeg']
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 2 次

### fake_biggan_0007_png ｜GT Fake｜png｜B
- 判定 Fake（后验 0.7354）｜停止 model_stalled｜调用 ['jpeg']
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_biggan_0007_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_biggan_0007_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_biggan_0007_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Uncertain（后验 0.4689）｜停止 model_stalled｜调用 ['jpeg']
  - `jpeg_expert` → Uncertain（P(Fake)=0.456，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_glide_0012_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.7354）｜停止 candidate_matches_posterior｜调用 ['jpeg']
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_glide_0012_png ｜GT Fake｜png｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_glide_0012_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.7354）｜停止 candidate_matches_posterior｜调用 ['jpeg']
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_glide_0012_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Real（后验 0.2313）｜停止 no_expected_gain｜调用 ['probe', 'noise', 'jpeg']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.641，inverted:high-metric-means-real，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.385，inverted:high-metric-means-real，一致性 None）

### fake_glide_0012_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_midjourney_0019_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.8103）｜停止 budget_exhausted｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_midjourney_0019_png ｜GT Fake｜png｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_midjourney_0019_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）

### fake_midjourney_0019_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_midjourney_0019_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Fake（后验 0.8192）｜停止 candidate_matches_posterior｜调用 ['probe', 'freq']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - `frequency_expert_v2` → AI-generated（P(Fake)=0.709，weak:marginally-above-chance，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_sd14_0027_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_sd14_0027_png ｜GT Fake｜png｜B
- 判定 Fake（后验 0.8192）｜停止 candidate_matches_posterior｜调用 ['probe', 'freq']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - `frequency_expert_v2` → AI-generated（P(Fake)=0.709，weak:marginally-above-chance，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_sd14_0027_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_sd14_0027_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.7552）｜停止 budget_exhausted｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.385，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_sd14_0027_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Uncertain（后验 0.3588）｜停止 model_stalled｜调用 ['jpeg']
  - `jpeg_expert` → Real（P(Fake)=0.295，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_sd15_0030_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_sd15_0030_png ｜GT Fake｜png｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_sd15_0030_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_sd15_0030_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.8103）｜停止 budget_exhausted｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 4 次

### fake_sd15_0030_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **score_as_artifact**：{"verdict": "Fake", "confidence": 0.9999, "primary_evidence": ["aigi_discriminator_score"], "report": "图像中的纹理和颜色异常一致地指向AI生成，AI生成鉴别器的得分显示图像高度倾向于AI生成
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_vqdm_0038_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.7466）｜停止 candidate_matches_posterior｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_vqdm_0038_png ｜GT Fake｜png｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_vqdm_0038_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_vqdm_0038_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_vqdm_0038_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_wukong_0043_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.8103）｜停止 budget_exhausted｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 4 次

### fake_wukong_0043_png ｜GT Fake｜png｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_wukong_0043_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.8103）｜停止 budget_exhausted｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 5 次

### fake_wukong_0043_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### fake_wukong_0043_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 2 次

### real_0009_native ｜GT Real｜native｜B
- 判定 Real（后验 0.1875）｜停止 candidate_matches_posterior｜调用 ['probe', 'freq']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.449，weak:marginally-above-chance，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0009_png ｜GT Real｜png｜B
- 判定 Uncertain（后验 0.464）｜停止 model_stalled｜调用 ['jpeg']
  - `jpeg_expert` → Uncertain（P(Fake)=0.449，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0009_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0009_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 2 次

### real_0009_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.1264）｜停止 no_expected_gain｜调用 ['probe', 'freq', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `frequency_expert_v2` → Real（P(Fake)=0.392，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.282，inverted:high-metric-means-real，一致性 None）

### real_0015_native ｜GT Real｜native｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0015_png ｜GT Real｜png｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0015_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0015_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0015_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0016_native ｜GT Real｜native｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0016_png ｜GT Real｜png｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0016_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0016_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0016_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0021_native ｜GT Real｜native｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0021_png ｜GT Real｜png｜B
- 判定 Real（后验 0.0953）｜停止 no_expected_gain｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 2 次

### real_0021_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0021_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.1087）｜停止 budget_exhausted｜调用 ['probe', 'freq', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `frequency_expert_v2` → Real（P(Fake)=0.392，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 3 次

### real_0021_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 3 次

### real_0024_native ｜GT Real｜native｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0024_png ｜GT Real｜png｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 2 次

### real_0024_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.1158）｜停止 no_expected_gain｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.295，inverted:high-metric-means-real，一致性 None）

### real_0024_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0024_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）

### real_0025_native ｜GT Real｜native｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0025_png ｜GT Real｜png｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0025_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0025_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0025_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.1493）｜停止 budget_exhausted｜调用 ['probe', 'freq', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.449，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.282，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 3 次

### real_0026_native ｜GT Real｜native｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0026_png ｜GT Real｜png｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0026_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0026_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0026_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0042_native ｜GT Real｜native｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0042_png ｜GT Real｜png｜B
- 判定 Real（后验 0.1446）｜停止 no_expected_gain｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.385，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0042_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0042_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.1897）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0042_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.0953）｜停止 no_expected_gain｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.0，strong，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 2 次

### real_0043_native ｜GT Real｜native｜B
- 判定 Real（后验 0.3102）｜停止 candidate_matches_posterior｜调用 ['noise']
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 2 次

### real_0043_png ｜GT Real｜png｜B
- 判定 Real（后验 0.3102）｜停止 candidate_matches_posterior｜调用 ['noise']
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0043_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 2 次

### real_0043_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 2 次

### real_0043_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0047_native ｜GT Real｜native｜B
- 判定 Real（后验 0.1948）｜停止 budget_exhausted｜调用 ['jpeg', 'probe']
  - `jpeg_expert` → Uncertain（P(Fake)=0.449，inverted:high-metric-means-real，一致性 None）
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 3 次

### real_0047_png ｜GT Real｜png｜B
- 判定 Real（后验 0.1117）｜停止 no_expected_gain｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0047_jpeg_q95 ｜GT Real｜jpeg_q95｜B
- 判定 Real（后验 0.3102）｜停止 candidate_matches_posterior｜调用 ['noise']
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0047_jpeg_q85 ｜GT Real｜jpeg_q85｜B
- 判定 Uncertain（后验 0.4193）｜停止 model_stalled｜调用 ['jpeg']
  - `jpeg_expert` → Real（P(Fake)=0.385，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### real_0047_jpeg_q70 ｜GT Real｜jpeg_q70｜B
- 判定 Uncertain（后验 0.4689）｜停止 model_stalled｜调用 ['jpeg']
  - `jpeg_expert` → Uncertain（P(Fake)=0.456，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_real_0000_native ｜GT Real｜native｜val
- 判定 Real（后验 0.1117）｜停止 no_expected_gain｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）

### acc_real_0000_png ｜GT Real｜png｜val
- 判定 Real（后验 0.1117）｜停止 budget_exhausted｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 4 次

### acc_real_0000_jpeg_q95 ｜GT Real｜jpeg_q95｜val
- 判定 Real（后验 0.1117）｜停止 budget_exhausted｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **score_as_artifact**：尽管存在面部细节模糊的视觉疑点，但AI生成鉴别器和微噪声残留水平的证据指向图像为真实拍摄
  - ⚠️ **score_as_artifact**：{"verdict": "Real", "confidence": 0.9362, "primary_evidence": ["aigi_discriminator_score", "noise_residual_inconsistency"], "report": "图像中存在面部细节模糊的疑点，但AI生成鉴别器和微噪声残留水平的证据指向图像为真实拍摄
  - ⚠️ **duplicate_call**：重复调用被抑制 4 次

### acc_real_0000_jpeg_q85 ｜GT Real｜jpeg_q85｜val
- 判定 Real（后验 0.1117）｜停止 budget_exhausted｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **score_as_artifact**：{"verdict": "Real", "confidence": 0.939, "primary_evidence": ["aigi_discriminator_score", "noise_residual_inconsistency"], "report": "图像中存在AI生成鉴别器和微噪声水平的证据，均指向真实性
  - ⚠️ **duplicate_call**：重复调用被抑制 4 次

### acc_real_0000_jpeg_q70 ｜GT Real｜jpeg_q70｜val
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_real_0001_native ｜GT Real｜native｜val
- 判定 Real（后验 0.2395）｜停止 no_expected_gain｜调用 ['probe', 'freq', 'noise', 'jpeg']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.41，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.295，inverted:high-metric-means-real，一致性 None）

### acc_real_0001_png ｜GT Real｜png｜val
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_real_0001_jpeg_q95 ｜GT Real｜jpeg_q95｜val
- 判定 Real（后验 0.2395）｜停止 no_expected_gain｜调用 ['probe', 'freq', 'noise', 'jpeg']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `frequency_expert_v2` → Uncertain（P(Fake)=0.41，weak:marginally-above-chance，一致性 None）
  - `noise_expert` → Real（P(Fake)=0.114，inverted:high-metric-means-real，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.295，inverted:high-metric-means-real，一致性 None）

### acc_real_0001_jpeg_q85 ｜GT Real｜jpeg_q85｜val
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

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
- 判定 Real（后验 0.1679）｜停止 no_expected_gain｜调用 ['probe', 'jpeg']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `jpeg_expert` → Real（P(Fake)=0.385，inverted:high-metric-means-real，一致性 None）

### acc_real_0002_jpeg_q85 ｜GT Real｜jpeg_q85｜val
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_real_0002_jpeg_q70 ｜GT Real｜jpeg_q70｜val
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_fake_adm_native ｜GT Fake｜native｜val
- 判定 Uncertain（后验 0.3578）｜停止 budget_exhausted｜调用 ['noise', 'probe', 'freq']
  - `noise_expert` → AI-generated（P(Fake)=0.848，inverted:high-metric-means-real，一致性 None）
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - `frequency_expert_v2` → Real（P(Fake)=0.392，weak:marginally-above-chance，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 3 次

### acc_fake_adm_png ｜GT Fake｜png｜val
- 判定 Real（后验 0.2184）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → Real（P(Fake)=0.061，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_fake_adm_jpeg_q95 ｜GT Fake｜jpeg_q95｜val
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_fake_adm_jpeg_q85 ｜GT Fake｜jpeg_q85｜val
- 判定 Uncertain（后验 0.5145）｜停止 model_stalled｜调用 ['probe']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_fake_adm_jpeg_q70 ｜GT Fake｜jpeg_q70｜val
- 判定 Uncertain（后验 0.4193）｜停止 model_stalled｜调用 ['jpeg']
  - `jpeg_expert` → Real（P(Fake)=0.385，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_fake_biggan_native ｜GT Fake｜native｜val
- 判定 Fake（后验 0.7354）｜停止 candidate_matches_posterior｜调用 ['jpeg']
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_fake_biggan_png ｜GT Fake｜png｜val
- 判定 Fake（后验 0.7354）｜停止 candidate_matches_posterior｜调用 ['jpeg']
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_fake_biggan_jpeg_q95 ｜GT Fake｜jpeg_q95｜val
- 判定 Uncertain（后验 0.464）｜停止 model_stalled｜调用 ['jpeg']
  - `jpeg_expert` → Uncertain（P(Fake)=0.449，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_fake_biggan_jpeg_q85 ｜GT Fake｜jpeg_q85｜val
- 判定 Fake（后验 0.8103）｜停止 candidate_matches_posterior｜调用 ['probe']
  - `probe_dino_expert` → AI-generated（P(Fake)=1.0，strong，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_fake_biggan_jpeg_q70 ｜GT Fake｜jpeg_q70｜val
- 判定 Uncertain（后验 0.4689）｜停止 model_stalled｜调用 ['jpeg']
  - `jpeg_expert` → Uncertain（P(Fake)=0.456，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_fake_glide_native ｜GT Fake｜native｜val
- 判定 Fake（后验 0.8268）｜停止 budget_exhausted｜调用 ['jpeg', 'probe', 'noise']
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.848，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 3 次

### acc_fake_glide_png ｜GT Fake｜png｜val
- 判定 Fake（后验 0.7354）｜停止 candidate_matches_posterior｜调用 ['jpeg']
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_fake_glide_jpeg_q95 ｜GT Fake｜jpeg_q95｜val
- 判定 Fake（后验 0.7354）｜停止 candidate_matches_posterior｜调用 ['jpeg']
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_fake_glide_jpeg_q85 ｜GT Fake｜jpeg_q85｜val
- 判定 Fake（后验 0.7354）｜停止 candidate_matches_posterior｜调用 ['jpeg']
  - `jpeg_expert` → AI-generated（P(Fake)=0.861，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 1 次

### acc_fake_glide_jpeg_q70 ｜GT Fake｜jpeg_q70｜val
- 判定 Uncertain（后验 0.5662）｜停止 budget_exhausted｜调用 ['probe', 'noise', 'jpeg']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.641，inverted:high-metric-means-real，一致性 None）
  - `jpeg_expert` → Uncertain（P(Fake)=0.456，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **duplicate_call**：重复调用被抑制 3 次
