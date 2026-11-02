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
| AutoSplice | **已上传并解压**：`dataset/AutoSplice/` 16,558 文件、494 MB；Authentic 2273、Forged_JPEG100/90/75 各 3621、Mask 1710、Caption 1710；README 计数一致 | 1710 个 base 源对应 1–3 个 forged 变体；每个 base 一份 mask/caption；mask 与对应 fake 尺寸 3621/3621 一致，但 mask 对每个变体的语义尚需确认；**q95 尚未派生**；学术非商业。archive 473,548,367 B，sha256 `0d94e2df…`，解压后已删除 |
| IML-ViT | **缺失**（无代码/权重） | 局部定位对照组无法开始；TruFor 已在本机 |

## 3. 可复用测量

- MNW q95：六个已有专家（PROBE、SPAI、AIDE progan、DRCT、AIDE sd14、
  AIDE GenImage）各有 **100 条 q95 测量**；可直接按 q95-only 子集复用，不能把
  original 混入新方案。
- GenImage q95：已按固定协议派生 **8000 假 + 1000 真**；清单为 `dataset/GenImage_q95/manifest.json`（源/派生哈希、尺寸、mode、编码器、重复标记）。尚无旧测量，下一步才是评测。
- AutoSplice q95：数据缺失，尚不可派生。

## 4. 模型与运行环境

| 资产 | 现场结果 |
| --- | --- |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct`，snapshot `0c351dd0…`，缓存约 **17.5 GB**；本轮未加载 |
| TruFor | 官方 repo commit `ae54475d…`；`trufor.pth.tar` 281,496,429 B，sha256 `ac1d90e3…` |
| PROBE / SPAI / DRCT / AIDE | 权重均在盘；SPAI/AIDE/DRCT 的路径与 sha256 已在 `experts/candidate_weights.json` |
| 磁盘 | `/root/autodl-tmp` 可用 **9.2 GiB**（82% 已用；q95 + AutoSplice 解压后）；overlay 余量约 1.6 GB |
| cgroup | max 120 GiB、events 全 0；当前 `memory.current` 约 24 GB，其中 anon ~0.98 GB、file/page-cache ~23 GB，不是进程常驻 |
| GPU | RTX 4090 24 GB，当前 1 MiB，无 compute app |
| 代码/测试 | 清点时 `core` 728 passed；本清点未改运行时代码 |

## 5. 阻塞项与下一步

1. **AutoSplice q95 未派生**：当前只解压原数据；需按 q95 协议从
   `Authentic` + `Forged_JPEG100` 派生，`Forged_JPEG90/75` 保留不用。mask 与
   1–3 个 forged 变体的对应语义需先核实，不能把一份 mask 直接复制给所有变体。
2. **IML-ViT 缺失**：需要代码和权重；TruFor 已可先做接口核对，但双定位器比较
   无法开始。
3. **GenImage 配对**：没有本地逐源真/假配对 manifest；当前方案是每组 1000 假图
   对共享 `dataset/Real` 1000 真图，组间复用、不独立；SD15 重复对已在 q95 清单标记，
   评测前按固定规则去重。
4. **磁盘**：q95 + AutoSplice 解压后可用 **9.2 GiB**；AutoSplice 原图 494 MB，
   q95 派生还需额外空间，注意 overlay 只剩约 1.6 GB。
5. **GPU**：新方向的真实 Agent 冒烟/评测仍需单独授权；本次只做 CPU 解压/核对。
