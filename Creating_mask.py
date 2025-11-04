from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path
from typing import Tuple

if find_spec("cv2") is None:  # pragma: no cover - executed only when dependency missing
    raise ModuleNotFoundError(
        "The 'cv2' module is required to run this script. Install it with "
        "'pip install opencv-python' or 'pip install opencv-python-headless'."
    )

import cv2
import numpy as np


def preprocess_to_binary(
    image: np.ndarray,
    binary_thresh: float,
    background: np.ndarray,
    *,
    min_difference: float = 5.0,
) -> np.ndarray:
    """Convert an image to a binary mask using a scaled background threshold.

    Parameters
    ----------
    image:
        Grayscale input image as a floating-point NumPy array.
    binary_thresh:
        Value between 0 and 1 used to scale the background before thresholding.
    background:
        Background image with the same dimensions as ``image``.
    min_difference:
        Minimum threshold value to avoid overly aggressive masking.

    Returns
    -------
    np.ndarray
        Binary mask containing 0 or 255 values.
    """

    if image.shape != background.shape:
        raise ValueError("`image` and `background` must share the same dimensions")

    scaled_threshold = np.maximum(binary_thresh * background, min_difference)
    binary_image = np.where(image < scaled_threshold, 0, 255)
    return binary_image.astype(np.uint8)


def load_grayscale_image(path: Path) -> np.ndarray:
    """Load an image in grayscale and convert it to float32."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image.astype(np.float32)


def validate_kernel_size(value: str) -> int:
    """Ensure the Gaussian kernel size is a positive odd integer."""

    try:
        kernel_size = int(value)
    except ValueError as exc:  # pragma: no cover - argparse guarantees message formatting
        raise argparse.ArgumentTypeError("Kernel size must be an integer") from exc

    if kernel_size <= 0:
        raise argparse.ArgumentTypeError("Kernel size must be positive")

    if kernel_size % 2 == 0:
        kernel_size += 1  # GaussianBlur requires odd kernel sizes; adjust automatically.
    return kernel_size


def parse_args(args: Tuple[str, ...] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a binary mask using a background-aware threshold"
    )
    parser.add_argument("image", type=Path, help="Path to the grayscale image")
    parser.add_argument(
        "output",
        type=Path,
        help="Destination path where the binary mask will be written",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        dest="binary_thresh",
        help="Scaling factor applied to the background before thresholding (default: 0.7)",
    )
    parser.add_argument(
        "--kernel",
        type=validate_kernel_size,
        default=51,
        help=(
            "Kernel size used for Gaussian blur of the background. "
            "Even values will be rounded up to the next odd integer (default: 51)."
        ),
    )
    parser.add_argument(
        "--min-difference",
        type=float,
        default=5.0,
        help="Minimum threshold value to avoid losing faint foreground details (default: 5)",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()

    image = load_grayscale_image(args.image)
    background = cv2.GaussianBlur(image, (args.kernel, args.kernel), 0)

    binary_mask = preprocess_to_binary(
        image,
        args.binary_thresh,
        background,
        min_difference=args.min_difference,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), binary_mask):
        raise OSError(f"Failed to write mask to {args.output}")

    print(f"Mask saved to {args.output}")


if __name__ == "__main__":
    main()