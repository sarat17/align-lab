"""Métricas calculadas realmente por el algoritmo."""

from __future__ import annotations

import time
from contextlib import contextmanager

import cv2
import numpy as np


@contextmanager
def timer():
    """Cronómetro simple: ``with timer() as t: ...; t['seconds']``."""
    box = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - start


def reprojection_errors(H: np.ndarray, pts_input: np.ndarray, pts_template: np.ndarray) -> np.ndarray:
    """Distancia euclídea (px) entre H·p_input y p_template para cada par."""
    projected = cv2.perspectiveTransform(pts_input.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    return np.linalg.norm(projected - pts_template.astype(np.float64), axis=1)


def reprojection_summary(H, pts_input, pts_template, inlier_mask) -> dict:
    """Error de reproyección sobre inliers (media, mediana, máximo), en píxeles de la plantilla."""
    if inlier_mask is None or not np.any(inlier_mask):
        return {"mean_px": None, "median_px": None, "max_px": None}
    err = reprojection_errors(H, pts_input[inlier_mask], pts_template[inlier_mask])
    return {
        "mean_px": float(np.mean(err)),
        "median_px": float(np.median(err)),
        "max_px": float(np.max(err)),
    }


def inlier_spatial_coverage(pts_template: np.ndarray, inlier_mask: np.ndarray, template_shape) -> float | None:
    """Fracción del área de la plantilla cubierta por el cierre convexo de los inliers.

    Indicador de dispersión espacial: muchos inliers concentrados en una zona
    pequeña dan una homografía poco fiable fuera de ella.
    """
    pts = pts_template[inlier_mask]
    if len(pts) < 3:
        return None
    hull = cv2.convexHull(pts.astype(np.float32))
    area = cv2.contourArea(hull)
    th, tw = template_shape[:2]
    return float(area / (th * tw))


def difference_stats(template_gray: np.ndarray, aligned_gray: np.ndarray, coverage: np.ndarray) -> dict:
    """Diferencia absoluta media restringida a píxeles válidos (cobertura)."""
    valid = coverage > 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"valid_pixels": 0, "coverage_fraction": 0.0, "mean_abs_diff": None}
    diff = cv2.absdiff(template_gray, aligned_gray)
    return {
        "valid_pixels": n_valid,
        "coverage_fraction": float(n_valid / valid.size),
        "mean_abs_diff": float(diff[valid].mean()),
    }


def build_metrics(
    *,
    detector_requested: str,
    detector_used: str,
    n_kp_input: int,
    n_kp_template: int,
    n_ratio_matches: int,
    n_used_matches: int,
    n_inliers: int,
    n_outliers: int,
    seconds: float,
    reproj: dict,
    spatial_coverage,
    diff_stats: dict,
    params: dict,
) -> dict:
    inlier_rate = (n_inliers / n_used_matches) if n_used_matches else None
    return {
        "detector_requested": detector_requested,
        "detector_used": detector_used,
        "keypoints_input": n_kp_input,
        "keypoints_template": n_kp_template,
        "matches_after_ratio_test": n_ratio_matches,
        "matches_used_for_homography": n_used_matches,
        "inliers": n_inliers,
        "outliers": n_outliers,
        "inlier_rate": inlier_rate,
        "reprojection_error_inliers_px": reproj,
        "inlier_spatial_coverage": spatial_coverage,
        "difference_stats": diff_stats,
        "processing_seconds": seconds,
        "parameters": params,
    }
