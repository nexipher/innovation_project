# G7 入口 3 验收审阅单

- 模式：`dry_run`｜样本 8｜批次源 24
- 汇总：{"samples": 8, "probe_called_in": 8, "probe_token_rate": 1.0, "agreement_with_probe_bins_mapping": {"agree": 8}, "agreement_with_probe_logistic": {"agree": 8}, "the_two_references_agree": {"same": 8}, "agent_verdicts": {"Fake": 8}, "halting_reasons": {"candidate_matches_posterior": 8}, "call_counts": {"probe": 8, "noise": 8}, "first_calls": {"probe": 8}, "landed_tokens_per_tag": {"probe": 8, "noise": 8}, "per_treatment": {"jpeg_q70": {"agent_Fake": 1}, "jpeg_q85": {"agent_Fake": 1}, "jpeg_q95": {"agent_Fake": 2}, "native": {"agent_Fake": 2}, "png": {"agent_Fake": 2}}, "mean_seconds_per_sample": 0.03, "mean_model_turns": 3.0}

> 自动标记只说明「这一行值得人看」，不构成结论（D2 的教训：视觉预筛能看不能判，
> 文本层标记同理——它们负责把 120 条缩到该细看的那几条）。

## 自动标记（8 处）

- `score_as_artifact`：8 处

## 逐会话

### fake_adm_0000_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.6853）｜停止 candidate_matches_posterior｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.848，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **score_as_artifact**：{"verdict": "Fake", "confidence": 0.8774, "primary_evidence": ["probe", "noise"], "report": "经多轮法证分析，图像在频域和噪声层面展现出典型的AI生成特征

### fake_adm_0000_png ｜GT Fake｜png｜B
- 判定 Fake（后验 0.6853）｜停止 candidate_matches_posterior｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.848，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **score_as_artifact**：{"verdict": "Fake", "confidence": 0.8774, "primary_evidence": ["probe", "noise"], "report": "经多轮法证分析，图像在频域和噪声层面展现出典型的AI生成特征

### fake_adm_0000_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.6853）｜停止 candidate_matches_posterior｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.848，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **score_as_artifact**：{"verdict": "Fake", "confidence": 0.8774, "primary_evidence": ["probe", "noise"], "report": "经多轮法证分析，图像在频域和噪声层面展现出典型的AI生成特征

### fake_adm_0000_jpeg_q85 ｜GT Fake｜jpeg_q85｜B
- 判定 Fake（后验 0.6853）｜停止 candidate_matches_posterior｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.848，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **score_as_artifact**：{"verdict": "Fake", "confidence": 0.8774, "primary_evidence": ["probe", "noise"], "report": "经多轮法证分析，图像在频域和噪声层面展现出典型的AI生成特征

### fake_adm_0000_jpeg_q70 ｜GT Fake｜jpeg_q70｜B
- 判定 Fake（后验 0.6853）｜停止 candidate_matches_posterior｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.848，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **score_as_artifact**：{"verdict": "Fake", "confidence": 0.8774, "primary_evidence": ["probe", "noise"], "report": "经多轮法证分析，图像在频域和噪声层面展现出典型的AI生成特征

### fake_biggan_0007_native ｜GT Fake｜native｜B
- 判定 Fake（后验 0.6853）｜停止 candidate_matches_posterior｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.848，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **score_as_artifact**：{"verdict": "Fake", "confidence": 0.9263, "primary_evidence": ["probe", "noise"], "report": "经多轮法证分析，图像在频域和噪声层面展现出典型的AI生成特征

### fake_biggan_0007_png ｜GT Fake｜png｜B
- 判定 Fake（后验 0.6853）｜停止 candidate_matches_posterior｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.848，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **score_as_artifact**：{"verdict": "Fake", "confidence": 0.9263, "primary_evidence": ["probe", "noise"], "report": "经多轮法证分析，图像在频域和噪声层面展现出典型的AI生成特征

### fake_biggan_0007_jpeg_q95 ｜GT Fake｜jpeg_q95｜B
- 判定 Fake（后验 0.6853）｜停止 candidate_matches_posterior｜调用 ['probe', 'noise']
  - `probe_dino_expert` → Uncertain（P(Fake)=0.52，strong，一致性 None）
  - `noise_expert` → AI-generated（P(Fake)=0.848，inverted:high-metric-means-real，一致性 None）
  - ⚠️ **score_as_artifact**：{"verdict": "Fake", "confidence": 0.9263, "primary_evidence": ["probe", "noise"], "report": "经多轮法证分析，图像在频域和噪声层面展现出典型的AI生成特征
