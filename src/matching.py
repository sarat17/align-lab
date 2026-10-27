"""Emparejamiento de descriptores con Lowe ratio test.

El matcher se elige por el nombre del detector *efectivo* ("SIFT" u "ORB"),
de modo que no puede desincronizarse del tipo de descriptor.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detectors import ORB, SIFT


class MatchingError(RuntimeError):
    """Error de emparejamiento con mensaje apto para el usuario."""


def create_matcher(detector_name: str):
    name = detector_name.upper()
    if name == SIFT:
        # Descriptores float32 -> distancia L2 (BFMatcher exacto: simple y robusto).
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    if name == ORB:
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raise MatchingError(f"No hay matcher para el detector {detector_name!r}.")


def ratio_test(knn_matches, ratio: float = 0.75) -> list:
    """Aplica el ratio test de Lowe tolerando listas con menos de 2 vecinos."""
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def match_descriptors(desc_input, desc_template, detector_name: str, ratio: float = 0.75) -> list:
    """Empareja input -> template y devuelve los matches que pasan el ratio test."""
    if not 0.3 <= ratio <= 0.95:
        raise MatchingError("El Lowe ratio debe estar entre 0.30 y 0.95.")
    if desc_input is None or desc_template is None:
        raise MatchingError("Una de las imágenes no produjo descriptores.")
    if len(desc_input) < 2 or len(desc_template) < 2:
        raise MatchingError("No hay suficientes descriptores para emparejar (se necesitan al menos 2).")

    if detector_name.upper() == SIFT:
        desc_input = np.asarray(desc_input, dtype=np.float32)
        desc_template = np.asarray(desc_template, dtype=np.float32)

    matcher = create_matcher(detector_name)
    knn = matcher.knnMatch(desc_input, desc_template, k=2)
    return ratio_test(knn, ratio)


def sort_by_distance(matches: list) -> list:
    """Ordena por distancia de descriptor (no garantiza corrección geométrica)."""
    return sorted(matches, key=lambda m: m.distance)


def select_matches(matches: list, keep_percent: float = 1.0, minimum: int = 4) -> list:
    """Conserva el mejor ``keep_percent`` de matches (por distancia).

    ``keep_percent=1.0`` usa todos los que pasaron el ratio test (valor por
    defecto recomendado). Nunca devuelve menos de ``minimum`` si hay suficientes.
    """
    if not 0.05 <= keep_percent <= 1.0:
        raise MatchingError("El porcentaje de matches debe estar entre 5 % y 100 %.")
    ordered = sort_by_distance(matches)
    if keep_percent >= 1.0:
        return ordered
    n = max(minimum, int(round(len(ordered) * keep_percent)))
    return ordered[:n]
