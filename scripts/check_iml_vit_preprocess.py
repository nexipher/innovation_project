"""CPU-only check of IML-ViT's official pad/normalize transform.

This imports only ``utils.datasets``/``utils.iml_transforms`` from the pinned
official checkout.  It does not build the model or load weights.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--subset", default=None)
    args = parser.parse_args()
    if not args.image and not args.subset:
        raise SystemExit("provide --image or --subset")
    sys.path.insert(0, args.repo)
    from utils.iml_transforms import get_albu_transforms

    transform = get_albu_transforms(type_="pad", outputsize=1024)

    def check_one(image_path, mask_path):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        if mask_path:
            # Mirror base_dataset: RGB -> mean > 127.5 -> a single HxW binary mask.
            mask_rgb = np.asarray(Image.open(mask_path).convert("RGB"))
            mask = (np.mean(mask_rgb, axis=2, keepdims=True) > 127.5).astype(np.float32)
            mask = mask[:, :, 0]
            out = transform(image=image, masks=[mask])
            tensor = out["image"]
            mask_tensor = out["masks"][0]
            mask_shape = list(mask_tensor.shape)
            mask_binary = bool(set(np.unique(mask_tensor.numpy()).tolist()) <= {0.0, 1.0})
            return {
                "image": image_path, "mask": mask_path,
                "source_hw": list(image.shape[:2]),
                "tensor_shape": list(tensor.shape),
                "tensor_dtype": str(tensor.dtype),
                "tensor_min": float(tensor.min()),
                "tensor_max": float(tensor.max()),
                "mask_shape": mask_shape,
                "mask_binary": mask_binary,
            }
        tensor = transform(image=image)["image"]
        return {
            "image": image_path, "mask": None,
            "source_hw": list(image.shape[:2]),
            "tensor_shape": list(tensor.shape),
            "tensor_dtype": str(tensor.dtype),
            "tensor_min": float(tensor.min()),
            "tensor_max": float(tensor.max()),
            "mask_shape": None, "mask_binary": None,
        }

    if args.image:
        print(json.dumps({"repo": args.repo, **check_one(args.image, args.mask)},
                         ensure_ascii=False, indent=2))
        return

    subset = json.load(open(args.subset, encoding="utf-8"))
    results = []
    failures = []
    for row in subset["inputs"]:
        image_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), row["path"]) if not os.path.isabs(row["path"]) else row["path"]
        mask_path = None
        if row.get("mask_path"):
            mask_path = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), row["mask_path"])
            if not os.path.exists(mask_path):
                mask_path = None
        try:
            result = check_one(image_path, mask_path)
            results.append(result)
            if result["tensor_shape"] != [3, 1024, 1024]:
                failures.append({"variant_id": row["variant_id"],
                                 "error": f"shape {result['tensor_shape']}"})
            if not np.isfinite([result["tensor_min"], result["tensor_max"]]).all():
                failures.append({"variant_id": row["variant_id"],
                                 "error": "non-finite tensor range"})
            if mask_path and result["mask_shape"] != [1024, 1024]:
                failures.append({"variant_id": row["variant_id"],
                                 "error": f"mask shape {result['mask_shape']}"})
            if mask_path and not result["mask_binary"]:
                failures.append({"variant_id": row["variant_id"],
                                 "error": "non-binary mask"})
        except Exception as exc:
            failures.append({"variant_id": row["variant_id"],
                             "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps({
        "repo": args.repo,
        "subset": args.subset,
        "checked": len(results),
        "failures": failures,
        "tensor_shapes": sorted({tuple(result["tensor_shape"]) for result in results}),
        "mask_shapes": sorted({tuple(result["mask_shape"]) for result in results
                               if result["mask_shape"]}),
        "tensor_min": min((result["tensor_min"] for result in results), default=None),
        "tensor_max": max((result["tensor_max"] for result in results), default=None),
        "source_hw_min": min((tuple(result["source_hw"]) for result in results),
                             key=lambda x: x[0] * x[1], default=None),
        "source_hw_max": max((tuple(result["source_hw"]) for result in results),
                             key=lambda x: x[0] * x[1], default=None),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
