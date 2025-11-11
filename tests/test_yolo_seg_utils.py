from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_seg_utils import masks_to_yolo_seg_lines


def _count_polygon_vertices(mask: np.ndarray, approx_eps: float) -> int:
    """Helper that matches the contour logic inside the converter."""
    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert contours, "expected at least one contour in the synthetic mask"
    cnt = contours[0]
    if approx_eps > 0:
        cnt = cv2.approxPolyDP(cnt, epsilon=approx_eps, closed=True)
    return cnt.reshape(-1, 2).shape[0]


def test_seg_lines_do_not_include_bounding_boxes():
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[5:19, 7:21] = 255  # single rectangular blob
    approx_eps = 1.5

    lines = masks_to_yolo_seg_lines(mask, class_id=3, approx_eps=approx_eps)
    assert len(lines) == 1

    tokens = lines[0].split()
    assert tokens[0] == "3"

    expected_pairs = _count_polygon_vertices(mask, approx_eps)
    assert len(tokens) == 1 + expected_pairs * 2, (
        "Segmentation labels must contain only the class id followed by polygon coordinate pairs."
    )
