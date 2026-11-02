# 资产与环境清点（2026-11-02）

本文件是 `NEXT_WORKFLOW_20261102.md` 阶段一/二的起点记录；数据图片本身不入 Git，
逐文件哈希见 `sft_data/g8_inventory/genimage_inventory_20261102.json`，机器可读汇总见
`sft_data/g8_inventory/asset_inventory_20261102.json`。

## 1. 代码与审核状态

- 起点提交：`e0e794b`（b2 原始审核导出纳入跟踪）；清点时工作区仅有本次生成的
  `sft_data/g8_inventory/`。
- `sft_data/g8/review/b2/review_g8_b2.json`：32 条决定，27 revise / 5 pass，
  sha256 `0564b83b…`；`g8_review_import.py --dry-run` 为 **32 接受、0 拒收、
  0 过期**。文件已跟踪，但**未导入台账、未冻结、未自动标通过**。

## 2. 数据资产

| 数据 | 现场结果 | 限制 / 缺项 |
| --- | --- | --- |
| GenImage_Test 八个生成器 | **8 × 1000 = 8000 PNG，2.8 GB**；ADM 1000 张 **RGBA** 256×256；BigGAN 128×128；Glide/VQDM 256×256；SD14/SD15/Wukong 512×512；Midjourney 1024×1024；**q95 已派生 8000 张**（`dataset/GenImage_q95/`，与 Real 合计约 763 MB） | 无本地真/假配对 manifest；SD15 有 **1 对重复哈希**（`459_sdv5_00039.png` 与 `680_sdv5_00100.png`），清单已标记 |
| GenImage Real | `dataset/Real/` **1000 JPG，35 MB**，500px 级、纵横比不一 | 作为八组共享真图；跨组复用，组间不独立；与假图分辨率和来源不同 |
| MNW | 4 组假图 **4×250=1000** + DIV2K 真图 **100**，1.1 GB | 现有 `manifest.json` 1100 条，含字节/哈希；仅评测用途 |
| MNW 开发清单 | `inputs_manifest.json` **200 条 = 100 源 × (original + q95)**；`inputs/` 下 q95 100 张 | q95-only 子集为 100 条；旧 original 仅历史参考 |
| MNW 分区 | 每组 `dev_current 20 / dev_extension 20 / final 20`；final 未使用 | 最终留出继续封存 |
| AutoSplice | **已解压并派生 q95**：原数据 16,558 文件/494 MB；Authentic 2273、Forged_JPEG100/90/75 各 3621、Mask/Caption 各 1710；q95 输入 **5894（2273 Real + 3621 Fake，227 MB）**；配对清单、审计和划分已生成 | 1710 个 base 对应 1–3 个 forged 变体；每个 base 一份 mask/caption（shared-by-base）；30 个 base/60 个变体抽查 mask 内差异均大于外，mask 与 fake 尺寸 3621/3621 一致；逐变体独立 mask 不存在；`Forged_JPEG90/75` 未用；学术非商业 |
| IML-ViT | **缺失**（无代码/权重）；官方 commit `07dd2be0…`、MIT、权重 Google Drive/Baidu | 局部定位对照组无法开始；准备清单见 `IML_VIT_PREP_20261102.md`；TruFor 已在本机但环境阻塞 |

## 3. 可复用测量

- MNW q95：六个已有专家（PROBE、SPAI、AIDE progan、DRCT、AIDE sd14、
  AIDE GenImage）各有 **100 条 q95 测量**；可直接按 q95-only 子集复用，不能把
  original 混入新方案。
- GenImage q95：已按固定协议派生 **8000 假 + 1000 真**；清单为 `dataset/GenImage_q95/manifest.json`（源/派生哈希、尺寸、mode、编码器、重复标记）。尚无旧测量，下一步才是评测。
- AutoSplice q95：已按固定协议派生 **2273 Real + 3621 Fake**；清单为 `dataset/AutoSplice_q95/inputs_manifest.json`，配对清单 `pairing_manifest.json`（含 30 base/60 变体 audit）。尚无旧测量，下一步是 TruFor/IML-ViT 小批。

## 4. 模型与运行环境

| 资产 | 现场结果 |
| --- | --- |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct`，snapshot `0c351dd0…`，缓存约 **17.5 GB**；本轮未加载 |
| TruFor | 官方 repo commit `ae54475d…`；`trufor.pth.tar` 281,496,429 B，sha256 `ac1d90e3…`；**小批被环境阻塞**（无 GPU、2 GiB cgroup） |
| PROBE / SPAI / DRCT / AIDE | 权重均在盘；SPAI/AIDE/DRCT 的路径与 sha256 已在 `experts/candidate_weights.json` |
| 磁盘 | `/root/autodl-tmp` 可用约 **9.0 GiB**（83% 已用；GenImage/AutoSplice q95 后）；overlay 余量约 1.6 GB |
| cgroup | **max 2 GiB**（2026-11-02 重新出现；此前一度读到 120 GiB）；当前 `memory.current` ~1.7 GB，events `max=6285`、`oom_kill=0`；TruFor CPU 进程在 ~0.52 GB RSS 时被 SIGKILL |
| GPU | **当前不可见**：`torch.cuda.is_available()=False`、`/dev/nvidia*` 不存在、`nvidia-smi -L` 无设备；此前为 RTX 4090 24 GB |
| 测试 | 清点时 `core` 728 passed；当前 2 GiB cgroup 下完整 `core` 被 SIGKILL；`tests/test_new_splits.py` 单独 3 passed |

## 5. 阻塞项与下一步

1. **TruFor 小批环境阻塞**：GPU 当前不可见；CPU 回退在 2 GiB cgroup 下被 SIGKILL。
   需恢复 GPU/更大 cgroup 后重跑；尚无 map/conf/score，不能把失败当模型结论。
2. **IML-ViT 缺失**：需要代码和权重（Google Drive/Baidu）；TruFor 小批恢复后可并行
   准备接口/预处理，但双定位器比较仍无法开始。
3. **AutoSplice mask 语义**：mask 是 base 级 shared-by-base，30/60 抽查支持
   mask 内差异大于外；没有逐变体独立 mask，报告需保留这一限制。
4. **GenImage 配对/去重**：无本地逐源真假配对 manifest；共享 Real 1000 跨八组复用，
   组间不独立；SD15 `680_sdv5_00100` 固定排除，canonical `459_sdv5_00039`。
5. **划分已完成**：AutoSplice dev/val/test = 1591/340/342 源；GenImage 各生成器
   ~700/150/150（SD15 699），共享 Real 700/150/150；test 封存。
6. **磁盘**：q95 后可用约 9.0 GiB；后续轨迹/AutoSplice q95/定位 raw 需继续控制。
