"""Estimación de homografía (RANSAC), validaciones y warp."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class RegistrationError(RuntimeError):
    """Error de registro con mensaje apto para el usuario."""


@dataclass
class HomographyResult:
    H: np.ndarray  # 3x3, input -> template
    inlier_mask: np.ndarray  # bool (N,)
    pts_input: np.ndarray  # (N, 2) float32
    pts_template: np.ndarray  # (N, 2) float32

    @property
    def n_candidates(self) -> int:
        return int(len(self.inlier_mask))

    @property
    def n_inliers(self) -> int:
        return int(self.inlier_mask.sum())

    @property
    def n_outliers(self) -> int:
        return self.n_candidates - self.n_inliers


def matches_to_points(matches, kps_input, kps_template):
    pts_in = np.float32([kps_input[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts_tp = np.float32([kps_template[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return pts_in, pts_tp


def validate_homography(H, image_shape, template_shape) -> None:
    """Comprobaciones de degeneración. Lanza ``RegistrationError`` si falla."""
    if H is None:
        raise RegistrationError("OpenCV no pudo calcular una homografía con estas correspondencias.")
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise RegistrationError("La homografía contiene valores no válidos.")
    if abs(H[2, 2]) < 1e-12:
        # Normalizamos por H[2,2] solo si es utilizable.
        pass
    else:
        H = H / H[2, 2]

    det = np.linalg.det(H[:2, :2])
    if not np.isfinite(det) or abs(det) < 1e-6:
        raise RegistrationError("La homografía es degenerada (colapsa la imagen a una línea o punto).")
    if det < 0:
        raise RegistrationError(
            "La homografía invierte la orientación de la imagen (reflexión); es poco plausible."
        )
    if np.linalg.cond(H) > 1e10:
        raise RegistrationError("La homografía está mal condicionada (resultado numéricamente inestable).")

    # El cuadrilátero de la imagen proyectado debe ser convexo y de área razonable.
    h, w = image_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if not np.all(np.isfinite(proj)):
        raise RegistrationError("La homografía proyecta la imagen a coordenadas no finitas.")
    cross = []
    for i in range(4):
        a, b, c = proj[i], proj[(i + 1) % 4], proj[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        raise RegistrationError(
            "La homografía deforma la imagen en un cuadrilátero no convexo (resultado poco plausible)."
        )
    area = abs(cv2.contourArea(proj.astype(np.float32)))
    src_area = float(h * w)
    th, tw = template_shape[:2]
    tpl_area = float(th * tw)
    if area < 1e-4 * tpl_area:
        raise RegistrationError("La imagen alineada ocuparía un área casi nula en la plantilla.")
    if area > 100 * max(tpl_area, src_area):
        raise RegistrationError("La homografía amplía la imagen de forma extrema; resultado poco plausible.")


def estimate_homography(
    matches, kps_input, kps_template, ransac_threshold: float = 5.0, image_shape=None, template_shape=None
) -> HomographyResult:
    if not 0.5 <= ransac_threshold <= 50:
        raise RegistrationError("El umbral RANSAC debe estar entre 0.5 y 50 píxeles.")
    if len(matches) < 4:
        raise RegistrationError(
            f"Solo hay {len(matches)} correspondencias; se necesitan al menos 4 para estimar una homografía. "
            "Prueba a subir el Lowe ratio, aumentar las características o cambiar de detector."
        )
    pts_in, pts_tp = matches_to_points(matches, kps_input, kps_template)
    # MAGSAC++ (OpenCV ≥ 4.5) es más robusto que RANSAC clásico cuando hay muchos emparejamientos falsos.
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, mask = cv2.findHomography(pts_in.reshape(-1, 1, 2), pts_tp.reshape(-1, 1, 2), method, ransac_threshold,
                                 maxIters=10000, confidence=0.999)
    if H is None or mask is None:
        raise RegistrationError("OpenCV no pudo calcular una homografía con estas correspondencias.")
    inlier_mask = mask.ravel().astype(bool)
    if inlier_mask.sum() < 4:
        raise RegistrationError(
            f"RANSAC solo conservó {int(inlier_mask.sum())} inliers; no hay consenso geométrico suficiente."
        )
    if image_shape is not None and template_shape is not None:
        validate_homography(H, image_shape, template_shape)
    elif not np.all(np.isfinite(H)):
        raise RegistrationError("La homografía contiene valores no válidos.")
    return HomographyResult(H=np.asarray(H, dtype=np.float64), inlier_mask=inlier_mask, pts_input=pts_in, pts_template=pts_tp)


def warp_to_template(image: np.ndarray, H: np.ndarray, template_shape):
    """Aplica la homografía con el tamaño de la plantilla.

    Devuelve ``(aligned, coverage_mask)``; la máscara (uint8 0/255) marca los
    píxeles de salida que provienen de la imagen de entrada.
    """
    th, tw = template_shape[:2]
    aligned = cv2.warpPerspective(image, H, (tw, th), flags=cv2.INTER_LINEAR)
    ones = np.full(image.shape[:2], 255, dtype=np.uint8)
    coverage = cv2.warpPerspective(ones, H, (tw, th), flags=cv2.INTER_NEAREST, borderValue=0)
    coverage = cv2.erode(coverage, np.ones((3, 3), np.uint8))  # evita bordes interpolados
    return aligned, coverage
