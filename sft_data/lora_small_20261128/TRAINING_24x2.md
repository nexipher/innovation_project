# 24 条 Qwen3-VL LoRA 小试 — 预检与正式训练

## 环境

- vGPU：NVIDIA RTX 4080 vGPU，32760 MiB（约 32.8 GB），0 MiB 在用。
- 磁盘：`/root/autodl-tmp` 50G/47G 已用/**3.9G 可用（93%）**；`/` 30G/14G/17G。
- 模型 snapshot、config、26 条 records、24 条 selection 均可读。

## 配置

- BF16；attention-only LoRA（q/k/v/o），rank 16 / alpha 32，dropout 0.05
- SGD，lr 1e-3；`max_length=12000`，完整轨迹，不截断
- gradient checkpointing；vision tower 冻结；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 可训练参数：15,335,424（0.1746%）

## 10 步预检（Midjourney 229、SD15 459、Real e4ce）

| step | sample | loss | step_s |
|---:|---|---:|---:|
| 1 | Midjourney 229 | 1.569940 | 10.437 |
| 2 | SD15 459 | 1.537181 | 7.406 |
| 3 | Real e4ce | 1.555736 | 7.187 |
| 4 | Midjourney 229 | 1.568990 | 9.500 |
| 5 | SD15 459 | 1.535043 | 7.421 |
| 6 | Real e4ce | 1.551410 | 7.190 |
| 7 | Midjourney 229 | 1.570178 | 9.526 |
| 8 | SD15 459 | 1.537898 | 7.432 |
| 9 | Real e4ce | 1.552958 | 7.198 |
| 10 | Midjourney 229 | 1.569105 | 9.519 |

- 平均 **8.282 s/step**，最大 **10.437 s**；峰值 allocated **24.881 GB**、reserved **25.287 GB**。
- 250 条 × 2 epoch = 500 步线性估算：按均值约 **69.0 min**，按最大约 **87.0 min**（不含模型加载/数据预处理）。

## 正式训练（24 条 × 2 epoch = 48 步）

- 从原版 Qwen3-VL 重新加载；24 条按 selection 循环两轮。
- 48/48 完成，无 OOM。
- loss 首/末：**1.729107 / 1.527636**；48 步均值约 1.58（逐 step 见 `run_manifest.json`）。
- 单步耗时：均值 **6.774 s**，最大 **9.526 s**；48 步求和 **325.17 s（约 5.42 min）**，不含模型加载/数据预处理。
- 峰值 allocated **24.881 GB**、reserved **25.313 GB**。
- 用 48 步实际均值估算 250×2=500 步：**约 56.5 min 计算**；按 10 步预检均值/最大为 **69.0 / 87.0 min**。实际 wall 会更长，建议按 **60–90 min** 估算。
- Adapter：`sft_data/lora_small_20261128/adapter/`（目录 70 MB，权重 59 MB）。
- 训练日志：`train_24x2_log.txt`；manifest：`run_manifest.json`。

## 未做

- 未扩到 250 条。
- 未改候选标签或训练数据选择。
