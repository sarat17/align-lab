"""Extracción de texto (OCR) sobre la imagen ya alineada con la plantilla.

Adaptado de ``ocr_form.py``: mismas regiones de interés y mismo
preprocesamiento (umbral de Otsu + Tesseract). Como la imagen alineada tiene
el tamaño y el sistema de coordenadas de la plantilla, las ROI se expresan en
píxeles de la plantilla.
"""

from __future__ import annotations

import os
import shutil

import cv2
import numpy as np

# Cada campo: (x, y, w, h) en píxeles de la plantilla.
FIELDS: dict[str, tuple[int, int, int, int]] = {
    "nombre": (440, 240, 460, 130),
    "informacion": (99, 360, 1200, 240),
    "autor": (250, 630, 850, 140),
    "id_arbol": (10, 30, 250, 190),
}

NUMERIC_FIELDS = {"id_arbol"}

# Rutas habituales del instalador de Tesseract en Windows.
_WINDOWS_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


class OCRUnavailable(RuntimeError):
    """Tesseract o pytesseract no están instalados."""


def _get_pytesseract():
    try:
        import pytesseract
    except ImportError as exc:
        raise OCRUnavailable("Falta el paquete 'pytesseract'. Instálalo con: python -m pip install pytesseract") from exc
    if shutil.which("tesseract") is None:
        for path in _WINDOWS_PATHS:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
        else:
            raise OCRUnavailable(
                "No se encontró el programa Tesseract. En Windows instálalo desde "
                "https://github.com/UB-Mannheim/tesseract/wiki ; en Streamlit Cloud se instala con packages.txt."
            )
    return pytesseract


def ocr_available() -> bool:
    try:
        _get_pytesseract()
        return True
    except OCRUnavailable:
        return False


def _clip_roi(x: int, y: int, w: int, h: int, shape) -> tuple[int, int, int, int] | None:
    """Recorta la ROI a los límites de la imagen. Devuelve None si queda vacía."""
    H, W = shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def extract_fields(aligned_bgr: np.ndarray, fields: dict | None = None):
    """Aplica OCR a cada región. Devuelve ``(resultados, imagen_anotada)``.

    ``resultados`` es un dict ``{campo: texto}``. Una ROI fuera de la imagen
    devuelve cadena vacía en lugar de fallar.
    """
    pytesseract = _get_pytesseract()
    fields = fields or FIELDS
    annotated = aligned_bgr.copy()
    results: dict[str, str] = {}

    for field, (x, y, w, h) in fields.items():
        box = _clip_roi(x, y, w, h, aligned_bgr.shape)
        if box is None:
            results[field] = ""
            continue
        x0, y0, x1, y1 = box
        roi = aligned_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if field in NUMERIC_FIELDS:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(
                thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789"
            ).strip()
        else:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, lang="eng").strip()

        results[field] = text
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(annotated, field, (x0, max(20, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return results, annotated
