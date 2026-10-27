"""Orquestación: decodificación segura de imágenes y flujo completo de alineación.

Decisión de diseño: este módulo se añade a la arquitectura pedida para que la
lógica de procesamiento quede fuera de ``app.py`` y sea testeable sin Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from . import detectors, matching, metrics, registration, visualization

MAX_FILE_MB = 25
MAX_PIXELS = 25_000_000  # ~5000x5000
ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg", "tif", "tiff")


class ImageLoadError(ValueError):
    """Error de carga con mensaje apto para el usuario."""


@dataclass
class AlignParams:
    detector: str = "SIFT"
    max_features: int = 5000
    ratio: float = 0.75
    ransac_threshold: float = 5.0
    keep_percent: float = 1.0  # 1.0 = usar todos los matches que pasan el ratio test


def validate_params(p: AlignParams) -> None:
    if p.detector.upper() not in detectors.SUPPORTED_DETECTORS:
        raise ValueError("Detector no soportado.")
    if not 100 <= p.max_features <= 50_000:
        raise ValueError("El máximo de características debe estar entre 100 y 50 000.")
    if not 0.3 <= p.ratio <= 0.95:
        raise ValueError("El Lowe ratio debe estar entre 0.30 y 0.95.")
    if not 0.5 <= p.ransac_threshold <= 50:
        raise ValueError("El umbral RANSAC debe estar entre 0.5 y 50 píxeles.")
    if not 0.05 <= p.keep_percent <= 1.0:
        raise ValueError("El porcentaje de matches debe estar entre 5 % y 100 %.")


def decode_image(data: bytes, filename: str = "", max_file_mb: float = MAX_FILE_MB, max_pixels: int = MAX_PIXELS):
    """Decodifica bytes a imagen BGR/BGRA/gris sin escribir a disco.

    Devuelve ``(image, info)``. El nombre de archivo solo se usa para validar
    la extensión; nunca como ruta.
    """
    if not data:
        raise ImageLoadError("El archivo está vacío.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_file_mb:
        raise ImageLoadError(f"El archivo pesa {size_mb:.1f} MB y el límite es {max_file_mb:g} MB.")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise ImageLoadError(f"Formato .{ext} no soportado. Usa PNG, JPG o TIFF.")

    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ImageLoadError("No se pudo leer la imagen: el archivo está dañado o no es una imagen válida.")
    if image.ndim not in (2, 3):
        raise ImageLoadError("Estructura de imagen no soportada.")
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        raise ImageLoadError(f"La imagen es demasiado pequeña ({w}×{h}); mínimo 32×32 píxeles.")
    if h * w > max_pixels:
        raise ImageLoadError(
            f"La imagen tiene {h * w / 1e6:.1f} MP y el límite es {max_pixels / 1e6:.0f} MP. Redúcela e inténtalo de nuevo."
        )
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    channels = 1 if image.ndim == 2 else image.shape[2]
    info = {"width": w, "height": h, "channels": channels, "size_kb": round(len(data) / 1024, 1)}
    return image, info


@dataclass
class AlignResult:
    H: np.ndarray
    aligned: np.ndarray
    coverage: np.ndarray
    matches_img: np.ndarray
    inliers_img: np.ndarray
    differences: np.ndarray
    differences_gray: np.ndarray
    overlay: np.ndarray
    metrics: dict
    template_bgr: np.ndarray
    input_bgr: np.ndarray


def align_images(image: np.ndarray, template: np.ndarray, params: AlignParams) -> AlignResult:
    """Flujo completo. Lanza errores con mensajes entendibles si algo falla."""
    validate_params(params)
    with metrics.timer() as t:
        detector_used = detectors.resolve_detector_name(params.detector)
        det_in = detectors.detect_and_describe(image, detector_used, params.max_features)
        det_tp = detectors.detect_and_describe(template, detector_used, params.max_features)

        ratio_matches = matching.match_descriptors(det_in.descriptors, det_tp.descriptors, detector_used, params.ratio)
        if len(ratio_matches) < 4:
            raise matching.MatchingError(
                f"Solo {len(ratio_matches)} correspondencias pasaron el ratio test (mínimo 4). "
                "Prueba a subir el Lowe ratio, aumentar las características o cambiar de detector."
            )
        used = matching.select_matches(ratio_matches, params.keep_percent)

        reg = registration.estimate_homography(
            used, det_in.keypoints, det_tp.keypoints, params.ransac_threshold,
            image_shape=image.shape, template_shape=template.shape,
        )
        aligned, coverage = registration.warp_to_template(image, reg.H, template.shape)

        reproj = metrics.reprojection_summary(reg.H, reg.pts_input, reg.pts_template, reg.inlier_mask)
        spatial = metrics.inlier_spatial_coverage(reg.pts_template, reg.inlier_mask, template.shape)

        tpl_bgr, in_bgr, al_bgr = visualization.to_bgr(template), visualization.to_bgr(image), visualization.to_bgr(aligned)
        tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
        al_gray = cv2.cvtColor(al_bgr, cv2.COLOR_BGR2GRAY)
        diff_stats = metrics.difference_stats(tpl_gray, al_gray, coverage)

        matches_img = visualization.draw_matches(in_bgr, det_in.keypoints, tpl_bgr, det_tp.keypoints, used)
        inliers_img = visualization.draw_matches(in_bgr, det_in.keypoints, tpl_bgr, det_tp.keypoints, used, reg.inlier_mask)
        overlay = visualization.make_overlay(tpl_bgr, al_bgr, 0.5)
        differences = visualization.difference_map(tpl_bgr, al_bgr, coverage, heatmap=True)
        differences_gray = visualization.difference_map(tpl_bgr, al_bgr, coverage, heatmap=False)

    m = metrics.build_metrics(
        detector_requested=params.detector.upper(),
        detector_used=detector_used,
        n_kp_input=det_in.count,
        n_kp_template=det_tp.count,
        n_ratio_matches=len(ratio_matches),
        n_used_matches=len(used),
        n_inliers=reg.n_inliers,
        n_outliers=reg.n_outliers,
        seconds=t["seconds"],
        reproj=reproj,
        spatial_coverage=spatial,
        diff_stats=diff_stats,
        params=asdict(params),
    )
    return AlignResult(
        H=reg.H, aligned=al_bgr, coverage=coverage, matches_img=matches_img, inliers_img=inliers_img,
        differences=differences, differences_gray=differences_gray, overlay=overlay, metrics=m,
        template_bgr=tpl_bgr, input_bgr=in_bgr,
    )


def compare_detectors(image, template, base: AlignParams, progress=None) -> list[dict]:
    """Ejecuta SIFT y ORB con parámetros comparables. Un fallo es un resultado fallido, no ceros."""
    rows = []
    names = detectors.SUPPORTED_DETECTORS
    for i, name in enumerate(names):
        if progress:
            progress(i / len(names), f"Ejecutando {name}…")
        row = {"detector": name, "status": "ok", "error": None}
        try:
            if name == detectors.SIFT and not detectors.sift_available():
                raise detectors.DetectionError("SIFT no está disponible en esta instalación.")
            p = AlignParams(**{**asdict(base), "detector": name})
            m = align_images(image, template, p).metrics
            row.update(
                seconds=m["processing_seconds"],
                keypoints_input=m["keypoints_input"],
                keypoints_template=m["keypoints_template"],
                matches=m["matches_after_ratio_test"],
                inliers=m["inliers"],
                inlier_rate=m["inlier_rate"],
                reproj_mean_px=m["reprojection_error_inliers_px"]["mean_px"],
            )
        except Exception as exc:  # fallo = resultado fallido explícito
            row.update(status="falló", error=str(exc))
        rows.append(row)
    if progress:
        progress(1.0, "Comparación terminada")
    return rows
