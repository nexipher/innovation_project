"""
Image loading, cropping, and format conversion utilities.
All methods are stateless and operate on numpy arrays (BGR, OpenCV convention).
"""

import cv2
import numpy as np
from typing import Tuple


class ImageUtils:
    """Stateless image utility functions."""

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """
        Load an image from disk as a BGR numpy array.

        Handles JPEG, PNG, and RGBA→RGB conversion. Returns uint8 BGR image.
        """
        # Use cv2.imread for BGR output
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # Try with PIL for edge cases (e.g., certain PNG modes)
            from PIL import Image
            pil_img = Image.open(path).convert("RGB")
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    @staticmethod
    def handle_alpha(img: np.ndarray) -> np.ndarray:
        """
        Convert an RGBA / BGRA image to BGR by dropping the alpha channel.
        Idempotent for 3-channel images.
        """
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    @staticmethod
    def crop_bbox(img: np.ndarray, bbox: list) -> np.ndarray:
        """
        Crop a bounding box region from an image.

        Args:
            img: BGR numpy array (H, W, 3).
            bbox: [ymin, xmin, ymax, xmax] in absolute pixel coordinates.

        Returns:
            Cropped patch as numpy array.
        """
        ymin, xmin, ymax, xmax = bbox
        # Ensure integer indices and within bounds
        ymin = max(0, int(ymin))
        xmin = max(0, int(xmin))
        ymax = min(img.shape[0], int(ymax))
        xmax = min(img.shape[1], int(xmax))
        return img[ymin:ymax, xmin:xmax].copy()

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert a BGR image to grayscale."""
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def get_dimensions(img: np.ndarray) -> Tuple[int, int]:
        """Return (height, width) of an image."""
        return img.shape[0], img.shape[1]

    @staticmethod
    def normalize_dtype(img: np.ndarray) -> np.ndarray:
        """Ensure image is uint8. If float [0,1], scale to [0,255]."""
        if img.dtype == np.uint8:
            return img
        if img.dtype in (np.float32, np.float64):
            if img.max() <= 1.0:
                return (img * 255).astype(np.uint8)
            return img.astype(np.uint8)
        return img.astype(np.uint8)
