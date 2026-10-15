"""
设备解析：真实专家的默认设备。

阶段 3 的评测是**显式** `--device cuda` 跑的（报告里记着 `device: cuda`），而专家桥
的默认值是 `cpu`——于是任何"默认构造"的真实会话（`main.build_pipeline`、冒烟、
生成入口）都会把三个专家放在 CPU 上：慢一个量级，UnivFD 的官方 .pt 还会直接抛异常
（TorchScript 图内 dtype 不匹配，实测 CPU 三条路都失败），而异常会被 4.4 的失败包装
记成一次 `failed` 工具调用——会话照跑、数据照收。

所以真实专家的默认设备由这里决定，而且**说出来**：`identity()` 与启动横幅都带上
解析后的设备名，"这批数据是在哪块设备上量的"不该靠猜。
"""

from __future__ import annotations

from typing import Optional


def resolve_device(requested: Optional[str] = None) -> str:
    """
    显式给了就用它（照原样返回），没给就按可用性选：cuda → cpu。

    不在这里检查权重是否存在：那是 `unavailable_reason()` 的职责，混在一起会让
    "没有 GPU"和"没有权重"报同一个错。
    """
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:                     # torch 装不上/坏了也不该在 import 期炸
        return "cpu"
