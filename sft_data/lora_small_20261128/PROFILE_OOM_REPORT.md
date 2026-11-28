# 24 条 LoRA 小试 — 10 步 profiling OOM 报告

时间：2026-11-28 12:01。**已停止，未继续训练，未跑 24×2 epoch，未做验证对照。**

## 磁盘

```text
$ df -h /root/autodl-tmp /
Filesystem      Size  Used Avail Use% Mounted on
/dev/md0         50G   47G  3.9G  93% /root/autodl-tmp
overlay          30G   14G   17G  44% /
```

`/root/autodl-tmp` 剩余 3.9 GB；adapter（约 59 MB）和日志可写，但没有大检查点空间。

## Profiling 配置

- 3 条：Midjourney 229（11677 token、5 图）、SD15 459（9608/5）、Real e4ce（9420/5）
- attention-only LoRA（q/k/v/o），rank 16 / alpha 32；SGD，lr 1e-3
- bf16、gradient checkpointing、vision tower 冻结
- `max_length=12000`，完整轨迹，不截断
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 可训练参数：15,335,424（0.1746%）

## 结果

在 **第 1 步 backward** OOM：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 274.00 MiB.
GPU 0 total 23.52 GiB; free 32.69 MiB; process 23.48 GiB;
PyTorch allocated 22.90 GiB; reserved 109.29 MiB.
```

没有完成任何一步，没有保存 adapter。完整日志：
`sft_data/lora_small_20261128/profile_10step_oom.log`。

## 与已跑通冒烟的区别

- 上次 6 步冒烟：`max_length=10000`，最长 9019 token（65547），peak allocated
  23.176 GB、reserved 23.530 GB，6/6 完成。
- 本次最长 11677 token（+2658 token），在 24 GB 卡上第一步 backward 即 OOM。
  这不是配置写错，而是完整 Midjourney 轨迹在当前模型/精度/LoRA 配置下超出显存。

## 可选处理（等你决定；均未执行）

1. **保留完整轨迹，降低序列占用**：只对训练数据去掉重复的
   `[System: 本会话状态…]` 消息（工具结果和助手调用仍保留），可减少约 500–800
   token；但不再是逐字完整轨迹，需要你确认是否接受。
2. **保留完整轨迹，缩小图像输入**：设置 Qwen processor 的 `max_pixels`，只减少
   视觉 token；本次 Midjourney 的视觉 token 很少，预计节省有限，且训练/推理
   预处理会不一致。
3. **换更省显存的训练方式**：4-bit/8-bit 量化或 CPU offload；当前未安装
   bitsandbytes，且会改精度/设备分布，未经你同意不执行。
4. **降低 LoRA target/rank**：attention-only、rank 16 已是很省；继续降到 rank 8
   对激活内存帮助有限。
5. **不训练 Midjourney 229**：从 24 条清单去掉或替换，但这会改变你已确认的清单。
6. **用更小 max_length 截断**：会截断完整轨迹，不建议默认执行。

## 当前状态

- profile_10step 目录无 adapter；GPU 已释放（0 MiB）。
- 未启动正式 24×2 epoch，未做原版 vs LoRA 的 30 图对照。
- 等你决定采用哪种处理后再继续。
