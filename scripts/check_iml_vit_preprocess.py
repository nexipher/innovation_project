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
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", default=None)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    from utils.iml_transforms import get_albu_transforms

    image = np.asarray(Image.open(args.image).convert("RGB"))
    transform = get_albu_transforms(type_="pad", outputsize=1024)
    if args.mask:
        # Mirror base_dataset: RGB -> mean > 127.5 -> a single HxW binary mask.
        mask_rgb = np.asarray(Image.open(args.mask).convert("RGB"))
        mask = (np.mean(mask_rgb, axis=2, keepdims=True) > 127.5).astype(np.float32)
        mask = mask[:, :, 0]
        out = transform(image=image, masks=[mask])
        tensor = out["image"]
        mask_tensor = out["masks"][0]
        mask_shape = list(mask_tensor.shape)
        mask_min = float(mask_tensor.min())
        mask_max = float(mask_tensor.max())
        mask_binary = bool(set(np.unique(mask_tensor.numpy()).tolist()) <= {0.0, 1.0})
    else:
        tensor = transform(image=image)["image"]
        mask_shape = None
        mask_min = mask_max = None
        mask_binary = None
    print(json.dumps({
        "repo": args.repo,
        "image": args.image,
        "mask": args.mask,
        "source_hw": list(image.shape[:2]),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_min": float(tensor.min()),
        "tensor_max": float(tensor.max()),
        "mask_shape": mask_shape,
        "mask_min": mask_min,
        "mask_max": mask_max,
        "mask_binary": mask_binary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
