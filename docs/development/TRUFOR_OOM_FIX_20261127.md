# TruFor 同卡 CUDA OOM：复现、根因与修复（2026-11-27）

- 范围：Qwen3-VL-8B-Instruct + PROBE-DINOv2 + TruFor 同卡（RTX 4090 24 GB）
- 历史批次：`sft_data/g8_dev_eval_20261110/`（200 会话，TruFor 17 次成功 / 131 次 OOM）
- 本轮**没有重跑 200 条**，没有迁移 CPU/其他 GPU，没有改精度、输入尺寸或模型配置；
  原始批次记录、trace、页面未改动。

## 1. 复现结果

用历史成功图 `genimage__Real__3d275712…`（500×333）和历史失败图
`autosplice__39697__0`（383×256）。TruFor 使用全新 `results_dir` 且 `cache=False`，
颜色顺序与 controller 一致（`ImageUtils.load_image` 的 BGR），避开结果缓存。

| 状态 | 调用前 allocated / reserved / 整卡 free | 成功图 | 失败图 |
|---|---|---|---|
| 仅 Qwen 加载 | 17.534 / 17.555 / 7.282 GB | 成功，7.79 s | 成功，8.23 s |
| Qwen 完成一次生成 | 17.543 / 19.214 / 5.537 GB | 成功 | 成功 |
| PROBE 首次推理后 | 18.760 / 18.849 / 5.910 GB | 成功 | 成功 |
| 生成 + PROBE（单会话完整状态） | 18.761 / 19.214 / 5.537 GB | 成功 | 成功 |

- 单张图、单会话的四种状态都**没有复现 OOM**；历史失败图在同状态下也能产出
  score/map/conf。分数与历史记录一致：成功图 `0.116657`（历史记录同一值），
  AutoSplice 图 `0.400073`。
- 重复同一输入做 6 次生成，reserved 稳定在 **19.214 GB**，不再增长；同一输入的重复
  调用不是原因。
- 旧代码连续 12 条 AutoSplice 会话：reserved 从 **19.292 → 22.070 GB**，free 从
  **5.460 → 2.681 GB**；这 12 条中 10 条调用 TruFor 并产出图，2 条模型未调用，
  **0 条 OOM**。说明仅 AutoSplice 小图还不足以跨过阈值。
- 200 条历史批次中，17 次成功集中在 21:35–22:08；之后 131 次失败，AutoSplice 0/90
  有效。1024×1024 Midjourney 等大图也在失败之列；部分 trace 的 stderr 尾部在 OOM
  之前被截断，原记录只显示“未产出 npz”。

## 2. 根因

- Qwen 常驻 17.53 GB，PROBE 首次推理后 allocated 约 18.76 GB。
- 不同长度、不同尺寸的会话让 PyTorch **reserved 缓存**涨到 22–23 GB；TruFor 子进程
  需要自己的 CUDA context，整卡 free 不足时在 conv 处 OOM。
- 不是权重、预处理或单图尺寸错误：同一输入/同一状态在 fresh process 中成功；失败
  依赖父进程累积的 reserved 缓存。
- TruFor 上游 `test.py` 用裸 `except` 吞掉异常并以退出码 0 结束，父进程只看到
  “未产出 npz”；旧代码的 stderr 尾部 500 字有时不含 OOM 关键字。

## 3. 最小修复

- `experts/trufor.py`
  - `_release_parent_cuda_cache()`：在启动 TruFor 单图/批量子进程前执行
    `gc.collect()` + `torch.cuda.empty_cache()`，把父进程可回收缓存还给整卡。
    不改变 device、weights、dtype、输入尺寸或模型配置；CPU 模式为 no-op。
  - `TruForCUDAOutOfMemory`（`failure_kind="cuda_oom"`）与 `_is_cuda_oom()`：
    扫描完整 stdout/stderr，明确区分 OOM 和普通执行失败。
  - 无 npz 时抛专用异常，错误消息包含 `CUDA OOM`。
- `state_machine/controller.py`
  - failed evidence token 增加 `failure_kind`；`failed_tool_calls` 保留异常类名和
    完整消息，trace 可明确看到 `TruForCUDAOutOfMemory` / `CUDA OOM`。
- 测试
  - `tests/test_external_experts.py`：OOM 正则、专用异常、子进程 OOM 路径。
  - `tests/test_tool_absence.py`：failed token 的 `failure_kind`、trace 中的 OOM。
  - `core`：**831 passed, 1 skipped, 855 deselected**。

## 4. 修复后验证

18 条连续 AutoSplice 联合会话（9 forged + 9 authentic），同一 Qwen/PROBE/TruFor
配置，全新 TruFor 缓存目录：

- 18/18 会话完成，0 session error。
- 14 条实际调用 TruFor；**14/14 产出 score、anomaly map、confidence map**，
  0 次失败。另 4 条模型选择不调用 TruFor（不是 OOM）。
- 子进程启动前 reserved 最高 **22.941 GB**、free 最低 **1.811 GB**；
  `_release_parent_cuda_cache()` 后 reserved **18.925 GB**、free **5.827 GB**。
  在这 14 次调用中全部成功，未再出现 OOM。
- 典型释放：`19.292 → 17.555 GB`（free 5.460 → 7.196），
  `22.941 → 18.925 GB`（free 1.811 → 5.827）。
- 结果文件：`/root/autodl-tmp/trufor_oom_fixed_20261127/validation.json`（本机临时目录，
  不属于 200 条批次）。

## 5. 边界与后续

- 本修复只解决同卡缓存/子进程启动顺序；没有改变测量语义。
- 未重跑 200 条。若后续重跑，需要重新固定 commit/config identity 并刷新 preflight；
  原 `g8_dev_eval_20261110` 结果保持原样。
- 本次验证中仍有 4/18 条会话未调用 TruFor；这是既有调用策略行为，另行记录，不与
  OOM 修复混为一谈。
- 若未来在更大图/更多并发下仍 OOM，应停下并汇报容量证据，不擅自迁移设备或改精度。
