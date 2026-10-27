"""Visualizaciones y conversiones de color (todo en BGR uint8 internamente)."""

from __future__ import annotations

import cv2
import numpy as np


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Convierte gris/BGR/BGRA a BGR uint8 (el canal alfa se compone sobre negro)."""
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image.copy()
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return (image[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
    raise ValueError(f"Número de canales no soportado: {image.shape[2]}")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(to_bgr(image), cv2.COLOR_BGR2RGB)


def make_overlay(template: np.ndarray, aligned: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Mezcla plantilla y alineada. ``alpha`` es la opacidad de la imagen alineada."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    t, a = to_bgr(template), to_bgr(aligned)
    if t.shape != a.shape:
        a = cv2.resize(a, (t.shape[1], t.shape[0]))
    return cv2.addWeighted(t, 1.0 - alpha, a, alpha, 0)


def draw_matches(image, kps_input, template, kps_template, matches, inlier_mask=None, max_draw: int = 300):
    """Dibuja correspondencias. Con ``inlier_mask``: inliers en verde, outliers en rojo.

    ``matches`` y ``inlier_mask`` deben estar alineados (misma longitud/orden).
    """
    img, tpl = to_bgr(image), to_bgr(template)
    if inlier_mask is None:
        idx = list(range(len(matches)))[:max_draw]
        return cv2.drawMatches(
            img, kps_input, tpl, kps_template, [matches[i] for i in idx], None,
            matchColor=(0, 220, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    inlier_mask = np.asarray(inlier_mask, dtype=bool)
    in_idx = [i for i in range(len(matches)) if inlier_mask[i]][:max_draw]
    out_idx = [i for i in range(len(matches)) if not inlier_mask[i]][:max_draw]
    canvas = cv2.drawMatches(
        img, kps_input, tpl, kps_template, [matches[i] for i in out_idx], None,
        matchColor=(60, 60, 235), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    # Segunda pasada: inliers encima, reutilizando el lienzo ya dibujado.
    canvas = cv2.drawMatches(
        img, kps_input, tpl, kps_template, [matches[i] for i in in_idx], canvas,
        matchColor=(0, 220, 0), flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return canvas


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    l, r = to_bgr(left), to_bgr(right)
    h = max(l.shape[0], r.shape[0])
    def pad(x):
        out = np.full((h, x.shape[1], 3), 24, np.uint8)
        out[: x.shape[0]] = x
        return out
    spacer = np.full((h, gap, 3), 24, np.uint8)
    return np.hstack([pad(l), spacer, pad(r)])


def difference_map(template: np.ndarray, aligned: np.ndarray, coverage: np.ndarray | None = None, heatmap: bool = True):
    """Diferencia absoluta (gris) entre plantilla y alineada, solo en píxeles válidos.

    Los píxeles sin cobertura se muestran en gris oscuro uniforme para no
    confundirlos con diferencias reales.
    """
    t = cv2.cvtColor(to_bgr(template), cv2.COLOR_BGR2GRAY)
    a = cv2.cvtColor(to_bgr(aligned), cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(t, a)
    out = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO) if heatmap else cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    if coverage is not None:
        out[coverage == 0] = (40, 40, 40)
    return out
