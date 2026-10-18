"""
外部评测的 200 个输入文件（2026-10-18 用户方案）。

100 张固定源图 × 2 个条件：

  * `original`  —— **原文件不动**，清单里直接引用它（记录 sha256/bytes 以便核对）；
  * `jpeg_q95`  —— 同一编码器、同一参数重编码，**不缩放**；真假两半走同一条代码路径。

写出的清单 `dataset/mnw_div2k/inputs_manifest.json` 记录每个输入的来源、哈希、
尺寸，以及 JPEG 条件的**编码器身份**（PIL 版本 + libjpeg 版本）与全部参数——
"同一编码"必须能被核对，不能只写在文档里。

边界（与 plan §4.9 一致，且不因实现而放宽）：
  * 统一 JPEG 只控制**最终编码流程**：MNW 源本身是 JPEG（历史压缩代数未知），
    DIV2K 是 PNG，因此两个来源的"编码代数"不同；不得把 JPEG→PNG 解释为未压缩。
  * 只做评测输入，不生成任何训练数据；不按分数挑图（清单在跑分之前已固定）。

用法：
  python scripts/build_external_inputs.py            # 生成 + 核对
  python scripts/build_external_inputs.py --dry-run  # 只核对原图与清单，不写副本
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, features  # noqa: E402

from config import PROJECT_ROOT  # noqa: E402

BASE = os.path.join(PROJECT_ROOT, "dataset", "mnw_div2k")
PLAN = os.path.join(BASE, "eval_plan.json")
OUTPUT_DIR = os.path.join(BASE, "inputs", "jpeg_q95")
MANIFEST = os.path.join(BASE, "inputs_manifest.json")

#: JPEG 条件：真假两半完全相同的一组参数（不缩放、不锐化、不做色彩管理）
JPEG_PARAMS = {"quality": 95, "subsampling": 2, "optimize": False, "progressive": False}


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encoder_identity() -> dict:
    """编码器身份：换一台机器、换一个 Pillow/libjpeg，条件就变了，必须写下来。"""
    return {"library": f"Pillow {Image.__version__}",
            "jpeg_backend": features.version("jpg") or "unknown",
            "params": dict(JPEG_PARAMS)}


def build(dry_run: bool = False) -> dict:
    plan = json.load(open(PLAN, encoding="utf-8"))
    encoder = encoder_identity()
    entries, problems = [], []
    seen_ids = set()

    for group, samples in sorted(plan["groups"].items()):
        for sample in samples:
            source = os.path.join(BASE, sample["path"])
            if not os.path.exists(source):
                problems.append(f"源文件缺失：{sample['path']}")
                continue
            if sha256_of(source) != sample["sha256"]:
                problems.append(f"源文件与固定清单不符：{sample['path']}")
                continue
            stem = os.path.splitext(os.path.basename(sample["path"]))[0]

            # 条件 1：原图 —— 不复制、不改动，直接引用。
            # 注意：eval_plan 里的 path 是**相对 dataset/mnw_div2k** 的，而批次入口按
            # `PROJECT_ROOT + path` 打开文件；清单必须统一成**相对仓库根**，否则外部
            # 评测会在 PROJECT_ROOT 下找 `Fake/...` 而失败。
            original_id = f"{group}__{stem}__original"
            entries.append({
                "variant_id": original_id, "condition": "original",
                "path": os.path.relpath(source, PROJECT_ROOT),
                "sha256": sample["sha256"],
                "bytes": sample["bytes"], "source_sha256": sample["sha256"],
                "source_path": sample["path"], "group": group,
                "label": sample["label"], "encoder": None, "params": None,
            })
            seen_ids.add(original_id)

            # 条件 2：统一 JPEG —— 同一编码器/参数，不缩放
            relative = os.path.splitext(sample["path"])[0] + ".jpg"
            target = os.path.join(OUTPUT_DIR, relative)
            jpeg_id = f"{group}__{stem}__jpeg_q95"
            if jpeg_id in seen_ids:
                problems.append(f"variant_id 冲突：{jpeg_id}")
            seen_ids.add(jpeg_id)
            if not dry_run:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with Image.open(source) as image:
                    mode_before = image.mode
                    rgb = image.convert("RGB")
                    width, height = rgb.size
                    rgb.save(target, format="JPEG", **JPEG_PARAMS)
            if os.path.exists(target):
                with Image.open(target) as check:
                    if check.size != (width, height):
                        problems.append(f"尺寸变了：{relative}")
                entries.append({
                    "variant_id": jpeg_id, "condition": "jpeg_q95",
                    "path": os.path.relpath(target, PROJECT_ROOT),
                    "sha256": sha256_of(target), "bytes": os.path.getsize(target),
                    "source_sha256": sample["sha256"], "source_path": sample["path"],
                    "group": group, "label": sample["label"],
                    "encoder": encoder, "width": width, "height": height,
                    "mode_converted_from": None if mode_before == "RGB" else mode_before,
                })
            elif not dry_run:
                problems.append(f"未生成：{relative}")

    if problems:
        raise SystemExit("外部输入构建失败：\n  " + "\n  ".join(problems))
    if dry_run:
        return {"entries": entries, "problems": problems, "written": False}

    manifest = {
        "schema": "external_inputs/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_plan": os.path.relpath(PLAN, PROJECT_ROOT),
        "source_plan_sha256": sha256_of(PLAN),
        "conditions": ["original", "jpeg_q95"],
        "encoder": encoder,
        "rules": {
            "no_resize": True,
            "same_encoder_for_real_and_fake": True,
            "originals_untouched": True,
            "no_threshold_fitting": True,
            "note": "JPEG→PNG 不得解释为未压缩；MNW 源本身是 JPEG，编码代数与 DIV2K 不同",
        },
        "counts": {"total": len(entries),
                   "original": sum(1 for e in entries if e["condition"] == "original"),
                   "jpeg_q95": sum(1 for e in entries if e["condition"] == "jpeg_q95")},
        "inputs": entries,
    }
    with open(MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    # 使用历史：这批输入是什么时候、为哪个阶段生成的
    plan.setdefault("usage_history", []).append({
        "at": datetime.now().isoformat(timespec="seconds"),
        "action": "built_inputs", "inputs": len(entries),
        "conditions": ["original", "jpeg_q95"],
        "encoder": encoder,
    })
    with open(PLAN, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
    return {"entries": entries, "problems": problems, "written": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = build(dry_run=args.dry_run)
    counts = {"original": 0, "jpeg_q95": 0}
    for entry in result["entries"]:
        counts[entry["condition"]] += 1
    print(f"输入：original {counts['original']}｜jpeg_q95 {counts['jpeg_q95']}"
          f"｜合计 {len(result['entries'])}")
    if result["written"]:
        print(f"清单：{os.path.relpath(MANIFEST, PROJECT_ROOT)}")
        print(f"副本目录：{os.path.relpath(OUTPUT_DIR, PROJECT_ROOT)}")
    else:
        print("（dry-run）未写任何文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
