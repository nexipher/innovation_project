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
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct`，snapshot `0c351dd0…`；**已迁移至数据盘** `dataset` 同级 `/root/autodl-tmp/models/Qwen3-VL-8B-Instruct`；旧缓存路径为符号链接；约 **17.5 GB**；已做实际加载检查（7.1 s，峰值 17.53 GB，未推理） |
| TruFor | 官方 repo commit `ae54475d…`；`trufor.pth.tar` 281,496,429 B，sha256 `ac1d90e3…`；**小批被环境阻塞**（无 GPU、2 GiB cgroup） |
| PROBE / SPAI / AIDE / DRCT | PROBE、SPAI 权重保留；**AIDE 三份与 DRCT 权重已于 2026-11-02 删除**；代码、注册表和历史结果保留 |
| 磁盘 | `/root/autodl-tmp` 可用约 **7.6 GiB**（85% 已用；Qwen 迁移后）；overlay 可用约 **18.1 GiB**（40% 已用） |
| cgroup | 迁移检查时回到 **120 GiB**；当前 `memory.current` 约 17.4 GB（含 Qwen 加载/page cache），events 全 0 |
| GPU | **已恢复**：RTX 4090 24 GB，`nvidia-smi` 可用；Qwen 加载后显存回到 0 MiB |
| 测试 | 清点时 `core` 728 passed；`tests/test_new_splits.py` 3 passed；完整 core 在 2 GiB 限制期间曾被 SIGKILL，环境恢复后可重跑 |

## 4.1 旧专家资产删除（2026-11-02）

按用户要求完成删除前核对并删除以下六项；AIDE/DRCT 代码、注册表和历史报告保留，Qwen3 未迁移。

| 资产 | 路径 | 大小 | sha256 | 哈希来源 |
| --- | --- | ---: | --- | --- |
| AIDE progan_train | `third_party/aide/weights/progan_train.pth` | 3,592,077,976 | `ce3a9d66…` | 2026-11-02 复算 |
| AIDE sd14_train | `third_party/aide/weights/sd14_train.pth` | 3,592,077,976 | `382a50d1…` | 2026-11-02 复算 |
| AIDE GenImage_train | `third_party/aide/weights/GenImage_train.pth` | 3,592,100,460 | `f084346d…` | 2026-11-02 复算 |
| DRCT pretrained.zip | `third_party/drct/weights/pretrained.zip` | 4,178,790,241 | `235c0069…` | 既有记录，大小未变 |
| DRCT ConvB checkpoint | `third_party/drct/weights/DRCT-2M/sdv14/.../14_acc0.9996.pth` | 354,605,453 | `d3de0fe4…` | 既有记录，大小未变 |
| SSP 压缩包 | `third_party/ssp.zip` | 754,886,830 | `71d280a8…` | 既有记录，大小未变 |

- 六项均已删除；对应的 `.sha256` sidecar 一并删除。
- 删除清单：`sft_data/g8_inventory/deleted_assets_20261102.json`。
- `/root/autodl-tmp` 可用空间从约 9.0 GiB 增至 **23.9 GiB**。
- **Qwen3-VL 未移动、未删除**；迁移按用户要求暂不开展。

## 4.2 Qwen3 迁移（2026-11-02）

- 源：`/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct`（旧系统盘目录）
- 目标：`/root/autodl-tmp/models/Qwen3-VL-8B-Instruct`
- 方式：`rsync -aH` 完整复制，核对 **33/33** 个条目、相对路径、大小与符号链接目标完全一致；
  复制完成后把旧路径替换为指向目标的符号链接，再删除旧系统盘目录。
- 结果：旧系统盘 Qwen 数据已删除；旧路径仅剩符号链接；overlay 可用从约 1.9 GiB 增至
  **约 18.1 GiB**；数据盘可用约 7.6 GiB。
- 运行检查：`snapshot_download`（固定 revision）解析成功；`AutoConfig`/`AutoProcessor`
  本地加载成功；实际 `Qwen3VLClient._ensure_loaded()` **7.1 s**、峰值显存 **17.53 GB**、
  4 个 shard、`thinking_supported=false`；未执行推理。
- 记录：`sft_data/g8_inventory/qwen_migration_20261102.json`。

## 5. 阻塞项与下一步

1. **TruFor 小批待重跑**：此前的失败是环境阻塞（GPU 不可见、2 GiB cgroup）；现在 GPU 和
   120 GiB cgroup 已恢复，但本轮只做 Qwen 迁移，尚未重跑；不能把旧失败当模型结论。
2. **IML-ViT 缺失**：需要代码和权重（Google Drive/Baidu）；TruFor 小批恢复后可并行
   准备接口/预处理，但双定位器比较仍无法开始。
3. **AutoSplice mask 语义**：mask 是 base 级 shared-by-base，30/60 抽查支持
   mask 内差异大于外；没有逐变体独立 mask，报告需保留这一限制。
4. **GenImage 配对/去重**：无本地逐源真假配对 manifest；共享 Real 1000 跨八组复用，
   组间不独立；SD15 `680_sdv5_00100` 固定排除，canonical `459_sdv5_00039`。
5. **划分已完成**：AutoSplice dev/val/test = 1591/340/342 源；GenImage 各生成器
   ~700/150/150（SD15 699），共享 Real 700/150/150；test 封存。
6. **磁盘**：删除旧专家后可用约 **23.9 GiB**；Qwen3 迁移暂缓。
