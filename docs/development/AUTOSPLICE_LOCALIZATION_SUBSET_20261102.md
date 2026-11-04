# AutoSplice q95 定位开发子集（2026-11-02 固定）

种子 `20261102`；先写清单、后跑模型。清单：`sft_data/g8_localization/autosplice_dev_subset_20261102.json`（sha256 `fe3c8f5a8b094cbdd197d5f822cc628b34d0a4247a1f192038b7e22eb5629e7c`）。

| 项目 | 数量 |
| --- | ---: |
| base | 100 |
| 输入 | 220 |
| Real | 100 |
| Fake | 120 |
| authentic-only base | 30 |
| 1 变体 base | 35 |
| 2 变体 base | 20 |
| 3 变体 base | 15 |

mask 面积切点：[0.26443371, 0.51886664]；authentic-only 尺寸切点：[361.0, 384.0]。

所有选中 base 的 authentic + 全部 forged 变体一起进入；mask 为 base 级共享，同源变体不能当独立源。q95、源图和 mask 的哈希已逐项与清单核对：q95 220、源图 220、mask 190，失败 0。

本轮无 GPU/2 GiB cgroup，未跑 TruFor/IML-ViT；子集只做固定，不产出分数。
