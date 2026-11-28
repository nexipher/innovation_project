# 加载 LoRA 后的真实会话

- Adapter：`sft_data/lora_smoke_20261128/adapter/`
- 图片：`autosplice__45749__1__q95`（企鹅；不在 3 条训练样本内）
- 模型：Qwen3-VL-8B-Instruct，non-thinking，greedy
- 结果：`model_concluded`
- 模型轮次：2；专家调用：**1 次 PROBE**；未调用 TruFor
- 结案：`Fake` / 整图 `AI-generated` / 局部 `Uncertain` / certainty `High`
- 报告：主要支持 PROBE 0.9994 指向 AI 生成；主要限制无 TruFor 测量，局部未知。
- 耗时 44.57 s；峰值 allocated 20.457 GB，reserved 20.594 GB。
- Trace：`sft_data/lora_smoke_20261128/real_session/traces/forensic_sft_session_20261128_114220_45749_1.json`
- 摘要：`sft_data/lora_smoke_20261128/real_session_summary.json`

这次会话验证了 adapter 能被现有 `Qwen3VLClient` 加载、控制器能完成真实工具调用和
结案；它没有证明 adapter 已改善判断，也没有强制它调用 TruFor。
