# IML-ViT 代码、预处理与输出语义核实（2026-11-04）

## 状态摘要

- **代码：已到位**。固定 commit `07dd2be0f4ea27a5c97c9fa5ffbe236733833eac`，
  从 GitHub archive（经 `ghfast.top`）下载；archive sha256
  `de140401ff941cd9c722b2b42d37cdece0ad310d6a66522cff9b2473579784fb`。
- 本地目录：
  `/root/autodl-tmp/third_party/IML-ViT/IML-ViT-07dd2be0f4ea27a5c97c9fa5ffbe236733833eac`
  （仓库外，不随项目分发；MIT）。
- **权重：仍缺失**。官方 Google Drive 直链在 20 s 内连接超时；Baidu 需要登录；
  HuggingFace 镜像 `HJYOGC/IML-ViT` 只有 README，没有权重。没有使用非官方权重。
- 当前环境 **2 GiB cgroup、GPU 不可见**：只做了不加载权重的代码/预处理检查，
  没有构建或运行模型。
- 代码目录内的关键文件哈希：
  - `iml_vit_model.py` `6d3b4ba1749bbf6b188e68f9657fff961f8a42bce818042fc1ba064fd64b7048`
  - `utils/datasets.py` `2710c246c39e19007ba6367d8dcd8b5b278c19c472afd759c35eaa477c146dd9`
  - `utils/iml_transforms.py` `3abfda4ed3777f93db55dc05922c7f745a3fcb72d952409a195f43e1b33b36f2`
  - `Demo.ipynb` `98eff570ed0b98ac6f4db99d01ae2b07fb55684efce030f7df334f80cfbac4ba`
  - `LICENSE` `7bce5d24d372c0abbf618951988ee2dc072e60027c55615f0229bdab0dad73c3`

## 官方推理/预处理路径

### 图像与 mask

`utils/datasets.py::base_dataset.__getitem__`：

1. `pil_loader` 把输入和 GT 都转为 **RGB**。
2. GT 转成 `mean(RGB, axis=2) > 127.5` 的单通道二值数组；不是直接使用三通道。
3. `get_albu_transforms(type_='pad', outputsize=1024)`：
   - `PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, value=0,
     position='top_left', mask_value=0)`；
   - `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`；
   - `Crop(0,0,1024,1024)`；
   - `ToTensorV2(transpose_mask=True)`。
4. 大于 1024 的边会被 **crop 左上角**，不缩放；当前固定的 AutoSplice 子集最大边
   535，尚未触发该路径。
5. `if_return_shape=True` 返回原图 `(h, w)`；推理输出用
   `[0:h, 0:w]` 从 1024×1024 map 左上角裁回原图。

### 模型前向

`Demo.ipynb` cell 8/10：

- `model = iml_vit_model()`；`torch.load(checkpoint)` 后
  `load_state_dict(..., strict=True)`；若失败再尝试 `checkpoint['model']`。
- `model.eval()`，`torch.no_grad()`。
- 官方 demo 即使推理也传入 `gt` 和 `edge_mask`：
  `predict_loss, mask_pred, edge_loss = model(img, gt, edge_mask)`。
  `iml_vit_model.forward` 内部用 `BCEWithLogitsLoss` 计算 loss，再把 `mask_pred`
  上采样到 1024×1024 并 **sigmoid**。
- **输出语义**：`mask_pred` 是 1024×1024 的 manipulation 概率图；没有 TruFor 式的
  image-level `score`/`conf`。不能把 loss 当检测分数，也不能强行补造 score。
- 对未知真值的图像，适配器必须传入形状正确的 dummy mask/edge mask（例如全零），
  只取 `mask_pred`；不能依赖 GT。

## 已执行的 CPU 检查

1. `import iml_vit_model` 成功（fvcore 已安装；timm 1.0.30 有 deprecation warning）。
2. 用官方 transform 对 AutoSplice 子集样本 `39697_0`（源 256×383）做预处理：
   - image tensor `3×1024×1024`，float32，范围约 `[-2.1179, 2.6400]`；
   - mask（按官方 RGB→mean>127.5 转二值）为 `1024×1024`，取值 `{0,1}`；
   - 结果与“zero-padding + ImageNet norm + top-left crop”一致。
   进一步对固定子集 **220/220** 输入逐个跑同一 transform：全部 tensor
   `3×1024×1024`、mask `1024×1024`/二值、无失败；源图尺寸范围 256×260 到
   459×459（均小于 1024，不触发 crop 丢内容）。
3. 当前官方 requirements 要求 `albumentations==1.3.0`；本机为 **1.4.14**（为 SPAI
   固定）。上面的 pad/normalize 路径在 1.4.14 下通过，但这不是完整的版本兼容性
   验证；真正加载权重前应固定独立环境或做逐项对照。
4. 没有构建模型、没有 `torch.load` 权重、没有跑前向。

## 权重获取状态

官方入口：

- Google Drive 文件 id `1xXJGJPW1i5j9Pc1JKd7fJmIAQkvt9jY7`；
- Baidu `https://pan.baidu.com/s/1IqyJdFLUo9bXEhvqOJkvvg?pwd=2k0o`；
- 期望落盘：`.../checkpoints/iml-vit_checkpoint.pth`。

实测：Google Drive `curl` 连接超时；Baidu 需要登录；HF 镜像 `HJYOGC/IML-ViT`
没有权重文件。**因此权重未核实、未下载、未使用。**

## 下一步（需要环境/资产）

1. 用户上传官方 checkpoint（或提供可达链接）后，先记录 bytes/sha256，再按官方
   `torch.load` 路径检查 state dict 覆盖（strict）。
2. 在 GPU/更大 cgroup 恢复后，用固定的 AutoSplice 100 base / 220 输入子集跑
   IML-ViT；输出仅 map，裁剪回原图尺寸。
3. 与 TruFor 同输入对照：逐图/按源 IoU、F1、precision、recall；真实图异常面积；
   同源多变体按 base 聚合；保留 shared-mask 限制。
4. 在定位专家确定前不改 Agent 区域观察、证据回传或结案流程。
