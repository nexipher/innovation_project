# 30 图验证集

- 来源：`sft_candidates_20261127/split.json` 的 **validation 侧**；与 24 条训练轨迹
  的全部 22 个唯一源图不同源，已用脚本断言无交集。
- 组成：10 AutoSplice 局部篡改、5 AutoSplice 真图、10 GenImage 整图生成、5 Real。
- GenImage 生成覆盖 ADM、BigGAN(2)、Glide、Midjourney(2)、SD14、SD15、VQDM、Wukong。
- 清单：`validation_manifest.json`；每图 q95 哈希已校验。
- 两臂使用相同清单、顺序、系统提示、PROBE/TruFor、贪心 non-thinking 自主调用流程；
  仅 Qwen 是否加载 LoRA 不同。
