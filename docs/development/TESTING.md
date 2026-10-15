# 测试分层与运行方式

> 目标：**不要每次改动都跑全套**。三层，各有明确用途与代价。最后核对：2026-10-13。

## 三层与命令

| 层 | 命令 | 内容 | 实测耗时 |
|----|------|------|----------|
| **核心不变量** | `scripts/run_tests.sh core` | 账本、门禁、协议解析、v2/v3 策略、转换与忠实性检查、校准来源、token/message 构造 | **约 30 秒**（341 项） |
| **模块测试** | `scripts/run_tests.sh module` | 上述之外的全部单元测试；**不含**会话级/端到端（`slow`） | 内存充足时一次跑完；紧张时分块，约 3–5 分钟 |
| **全景回归** | `scripts/run_tests.sh full` | 全部，含真实图像与 CPU 专家的会话级用例 | 同上；实测（本容器内存紧张时 7 块）1140 通过 + 1 跳过，约 5.5 分钟 |

也可以直接用 pytest：

```bash
python -m pytest tests -m core            # 快速
python -m pytest tests -m "not slow"      # 模块
python -m pytest tests                    # 全景（内存紧张时用 run_tests.sh，它分块跑）
```

**分块是自适应，不是固定要求**：`run_tests.sh` 会读当前容器的
`/sys/fs/cgroup/memory.max` 与 `memory.current`——
**余量 > 800 MB 就一次跑完；余量小则自动按每块 8 个测试文件分块**。
在内存受限的容器里，一次加载全部图像/专家进程会被 OOM 杀掉，而被杀的那次
**不报告已通过的用例**（本机曾发生三次，2 GB 上限、常驻 1.5 GB 左右）。
读不到 cgroup 文件的机器（本机之外、macOS、CI）按"内存充足"处理，
所以这条**不是对所有环境的要求**；跑全套前想知道余量：`cat /sys/fs/cgroup/memory.current`。

## 何时跑哪一层

| 改动 | 至少跑 |
|------|--------|
| 文档、注释、README/plan | 不跑 |
| 账本/门禁/协议/策略/解析器 | `core` |
| 专家算法、校准表、转换器、工作台 | `module` |
| 控制器主循环、提示词、端到端流程 | `full` |
| 换专家、改预算/停止规则、发布前 | `full` |

## 标记约定

- `pytestmark = pytest.mark.core` —— 核心不变量文件（改动前后都该通过）
- `pytestmark = pytest.mark.slow` —— 会话级/端到端文件（真的跑 CPU 专家、读写图像与轨迹）

新增测试时：凡是**不加载图像、不建会话**的纯逻辑测试，放进 `core` 组（或新建 core 文件）；
凡是**要跑一次完整会话**的，标 `slow`。

## 专家更换后哪些测试会退役

新专家接入、jpeg / noise / frequency 退出工具箱时，以下测试组**不再需要常跑**（保留代码，
从 `core`/`module` 的默认集合里移出即可）：

| 测试文件 | 退役理由 |
|----------|----------|
| `test_jpeg_expert.py`、`test_noise_expert.py`、`test_frequency_expert.py` | 对应专家退出工具箱 |
| `test_expert_admission.py`、`test_g7_source_selection.py` 中的适用性断言 | 适用性表随专家与校准重做 |
| `test_calibration_source.py` 中针对旧专家的逐格式断言 | 校准表重做后按新表重写 |
| `test_replay_halting_g3.py`、`test_qwen_gain_baseline.py` | 历史复算用，换专家后不再作为当前回归 |

**不退役**的：账本与门禁、单动作协议、v3 策略、转换忠实性、工作台批次隔离——
这些与专家无关，是流程的骨架。

## 与环境相关的注意事项

- `nvidia-smi` 不可用时（GPU 已关），任何 GPU 依赖的测试与脚本都应跳过或明说，不要静默降级。
- 会话级测试会写 `traces/sft_sessions/`（测试用 conftest 隔离到临时目录 ✓）。
- 运行全套前看一眼内存：`cat /sys/fs/cgroup/memory.current`（本机曾因共享 cgroup 被杀）。
