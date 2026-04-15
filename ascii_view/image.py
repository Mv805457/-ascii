"""Image loading and manipulation module."""

import os
import shutil
from PIL import Image
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Load image from path and return as numpy array."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    return np.array(img, dtype=np.float64)


def resize_image(
    image: np.ndarray,
    max_width: int = None,
    max_height: int = None,
    char_ratio: float = 2.0
) -> np.ndarray:
    """
    Resize image for terminal display.

    - Keeps image as large as possible
    - Only scales down if bigger than terminal
    - Corrects vertical stretching using char_ratio
    """

    height, width, _ = image.shape

    # Auto-detect terminal size if not provided
    if max_width is None or max_height is None:
        term_size = shutil.get_terminal_size()
        if max_width is None:
            max_width = term_size.columns
        if max_height is None:
            max_height = term_size.lines

    # Compute scale (ONLY shrink, never enlarge)
    scale_w = max_width / width
    scale_h = max_height / height

    scale = min(1.0, scale_w, scale_h)

    new_width = int(width * scale)
    new_height = int((height * scale) / char_ratio)

    # Safety clamp
    new_width = max(1, new_width)
    new_height = max(1, new_height)

    img = Image.fromarray(image.astype(np.uint8))
    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return np.array(resized, dtype=np.float64)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using luminance formula."""
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    return 0.299 * r + 0.587 * g + 0.114 * b


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize values to [0,1].

    THIS IS CRITICAL for correct ASCII mapping.
    """
    img = image.astype(np.float64)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return np.clip(img, 0.0, 1.0)
