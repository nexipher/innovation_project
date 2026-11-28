# Qwen3-VL-8B-Instruct LoRA 冒烟

## 配置

- 模型：`Qwen/Qwen3-VL-8B-Instruct`，revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- dtype：bfloat16；梯度检查点；vision tower 冻结
- LoRA：rank 16，alpha 32，dropout 0.05，**attention-only**
  （`q_proj,k_proj,v_proj,o_proj`），43.6M 可训练参数（0.50%）
- 优化器：SGD，lr 1e-3；6 steps（3 条轨迹循环两轮）
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 训练结果

| step | sample | loss |
|---:|---|---:|
| 1 | 55310 | 1.404611 |
| 2 | 65547 | 1.331620 |
| 3 | 54029 single TruFor | 1.371110 |
| 4 | 55310 | 1.408393 |
| 5 | 65547 | 1.333055 |
| 6 | 54029 single TruFor | 1.370175 |

- 峰值 allocated：**23.176 GB**；峰值 reserved：**23.530 GB**（24 GB 卡）。
- Adapter：`sft_data/lora_smoke_20261128/adapter/`（59 MB safetensors）。

## OOM 与调整（已报告）

1. AdamW + 全部 LoRA target（含 MLP）：step 1 cross_entropy OOM，
   尝试分配 352 MiB，已分配 22.44 GB。
2. SGD + 全部 LoRA target：step 2 forward dropout OOM，尝试分配 424 MiB。
3. SGD + attention-only LoRA + 直接取 decoder last hidden state：6/6 完成。
   MLP 适配器每层会多出 `[tokens, 11008]` 分支，是第二步 OOM 的主因；本次只缩小
   LoRA 作用范围，没有改模型、精度、输入图或 loss 掩码规则。

## 未做

- 没有继续扩到 250 条。
- 没有把 3 条训练样本当成质量评估；6 步只验证数据/目标/加载链路。
