from __future__ import annotations

from pathlib import Path
import argparse
import csv

import cv2
import numpy as np


HSV_RANGES = {
    "Door": (
        np.array([0, 120, 100], dtype=np.uint8),
        np.array([10, 255, 255], dtype=np.uint8),
    ),
    "Window": (
        np.array([35, 120, 100], dtype=np.uint8),
        np.array([77, 255, 255], dtype=np.uint8),
    ),
    "Column": (
        np.array([90, 120, 100], dtype=np.uint8),
        np.array([130, 255, 255], dtype=np.uint8),
    ),
}
DILATION_KERNEL = np.ones((7, 7), np.uint8)
DILATION_ITERATIONS = 3
MIN_CONTOUR_AREA_PX = 5.0


def extract_class_mask(annotation_bgr: np.ndarray, component_class: str) -> np.ndarray:
    hsv = cv2.cvtColor(annotation_bgr, cv2.COLOR_BGR2HSV)
    lower, upper = HSV_RANGES[component_class]
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.dilate(
        mask,
        DILATION_KERNEL,
        iterations=DILATION_ITERATIONS,
    )


def summarize_mask(mask: np.ndarray) -> dict[str, float]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= MIN_CONTOUR_AREA_PX
    ]

    areas = [float(cv2.contourArea(contour)) for contour in contours]
    aspect_ratios = []
    for contour in contours:
        _, _, width, height = cv2.boundingRect(contour)
        if height > 0:
            aspect_ratios.append(width / height)

    image_area = float(mask.shape[0] * mask.shape[1])
    return {
        "count": len(contours),
        "mean_area_px2": float(np.mean(areas)) if areas else 0.0,
        "mean_width_height_ratio": (
            float(np.mean(aspect_ratios)) if aspect_ratios else 0.0
        ),
        "total_area_ratio": float(sum(areas) / image_area) if image_area else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("opencv_masks"))
    args = parser.parse_args()

    image = cv2.imread(str(args.annotation_image))
    if image is None:
        raise FileNotFoundError(args.annotation_image)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for component_class in ["Door", "Window", "Column"]:
        mask = extract_class_mask(image, component_class)
        cv2.imwrite(
            str(args.output_dir / f"{component_class.lower()}_mask.png"),
            mask,
        )
        rows.append({
            "component_class": component_class,
            **summarize_mask(mask),
        })

    with (args.output_dir / "component_descriptors.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
