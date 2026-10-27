"""Detección y descripción de puntos clave (SIFT / ORB).

El nombre del detector ("SIFT" u "ORB") es la única fuente de verdad: el
matcher se deriva siempre del detector *realmente* usado (ver
``DetectionResult.detector``), nunca de un booleano aparte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

SIFT = "SIFT"
ORB = "ORB"
SUPPORTED_DETECTORS = (SIFT, ORB)


class DetectionError(RuntimeError):
    """Error de detección con mensaje apto para el usuario."""


def sift_available() -> bool:
    """True si esta instalación de OpenCV incluye SIFT utilizable."""
    try:
        cv2.SIFT_create()
        return True
    except Exception:
        return False


def available_detectors() -> list[str]:
    return [SIFT, ORB] if sift_available() else [ORB]


def resolve_detector_name(requested: str) -> str:
    """Devuelve el detector que se usará de verdad.

    Si se pide SIFT y no está disponible, se recurre a ORB. El resultado es el
    nombre que debe usarse también para elegir el matcher.
    """
    name = requested.upper()
    if name not in SUPPORTED_DETECTORS:
        raise DetectionError(
            f"Detector desconocido: {requested!r}. Opciones: {', '.join(SUPPORTED_DETECTORS)}."
        )
    if name == SIFT and not sift_available():
        return ORB
    return name


def create_detector(name: str, max_features: int = 5000):
    """Crea el detector indicado (sin fallback silencioso)."""
    if max_features < 10:
        raise DetectionError("El máximo de características debe ser al menos 10.")
    name = name.upper()
    if name == SIFT:
        if not sift_available():
            raise DetectionError("SIFT no está disponible en esta instalación de OpenCV.")
        return cv2.SIFT_create(nfeatures=int(max_features))
    if name == ORB:
        return cv2.ORB_create(nfeatures=int(max_features))
    raise DetectionError(f"Detector desconocido: {name!r}.")


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convierte gris, BGR o BGRA a gris uint8."""
    if image is None or image.size == 0:
        raise DetectionError("La imagen está vacía.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise DetectionError(f"Formato de imagen no soportado: forma {image.shape}.")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


@dataclass
class DetectionResult:
    detector: str  # detector realmente usado
    keypoints: list = field(default_factory=list)
    descriptors: Any = None

    @property
    def count(self) -> int:
        return len(self.keypoints)


def detect_and_describe(
    image: np.ndarray, detector_name: str, max_features: int = 5000, min_keypoints: int = 4,
    mask: np.ndarray | None = None,
) -> DetectionResult:
    """Detecta keypoints y descriptores. Lanza ``DetectionError`` si son insuficientes.

    ``mask`` (uint8, 0 = ignorar) limita la búsqueda a ciertas zonas de la imagen.
    """
    gray = to_gray(image)
    detector = create_detector(detector_name, max_features)
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    keypoints = list(keypoints or [])
    if descriptors is None or len(keypoints) < min_keypoints:
        raise DetectionError(
            f"Se detectaron muy pocos puntos clave ({len(keypoints)}) con {detector_name.upper()}. "
            "Prueba con una imagen con más textura, aumenta el máximo de características "
            "o cambia de detector."
        )
    return DetectionResult(detector=detector_name.upper(), keypoints=keypoints, descriptors=descriptors)
