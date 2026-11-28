# GenImage 整图生成 / 局部编辑配对（2026-11-28）

本轮交付 10 对数据、配对 manifest 和审核页；未运行 TruFor / Agent。

## 目录

- `manifest.json`：逐对记录源图、生成器、编辑类型、两图与 mask 路径、真值、编码参数。
- `pairs/<pair_id>/original.jpg`：原图（整图 AI-generated，局部 Not tampered）。
- `pairs/<pair_id>/edited.jpg`：局部编辑版（整图 AI-generated，局部 Tampered）。
- `pairs/<pair_id>/mask.png`：编码前实际变化像素，白色为编辑区域。
- `review/index.html`：原图 / 编辑版 / mask 并排审核页。
- `selection_audit.csv`：源图 split、no_known_use、近期使用审计。
- `encode_audit.csv`：尺寸、mask 像素、JPEG 参数、非编辑区变化审计。

## 数量

- 添加 4、删除 3、
  替换 3。
- 生成器分布：{"ADM": 1, "BigGAN": 1, "Glide": 4, "SD14": 1, "VQDM": 2, "Wukong": 1}。

## 选源与真值

- 全部来自 `dataset/GenImage_q95` 的 dev split，且 `no_known_use=true`。
- 已排除近期训练、验证、SFT 和固定对照 manifests 使用的源；test split 未使用。
- 原图是 GenImage generator-only q95 派生，无局部编辑记录；编辑版由本批次施加一次
  局部添加/删除/替换，mask 由编码前逐像素差异生成。
- 两图整图真值均为 `AI-generated`；局部真值分别为 `Not tampered` / `Tampered`。

## 编码

- Pillow `quality=95, subsampling=2, optimize=False, progressive=False`。
- 原图和编辑版都从同一份解码源图出发，用相同参数编码。
- 已校验两图 JPEG quantization tables 一致；mask 外像素在编码前变化为 0。

## 边界

- 只做数据、manifest、审核页；没有跑 TruFor、Agent、Qwen，也没有改专家或阈值。
- 编辑自然度和 mask 对齐需人工复核；未通过的对可以丢弃或重做。
