import cv2
import numpy as np
import pytest

from src import ocr
from src.pipeline import AlignParams, align_images

pytestmark = pytest.mark.skipif(not ocr.ocr_available(), reason="Tesseract no está instalado")


def _form():
    rng = np.random.default_rng(1)
    img = np.full((900, 1350, 3), 255, np.uint8)
    for _ in range(150):  # textura para que haya puntos clave
        c = tuple(int(v) for v in rng.integers(0, [1350, 900]))
        cv2.circle(img, c, int(rng.integers(3, 12)), tuple(int(v) for v in rng.integers(0, 200, 3)), 2)
    for (x, y, w, h) in ocr.FIELDS.values():
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.putText(img, "ROBLE", (480, 330), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5)
    cv2.putText(img, "4217", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5)
    return img


def test_ocr_after_alignment():
    template = _form()
    M = cv2.getRotationMatrix2D((675, 450), 4, 0.95)
    image = cv2.warpAffine(template, M, (1350, 900), borderValue=(255, 255, 255))
    aligned = align_images(image, template, AlignParams()).aligned
    results, annotated = ocr.extract_fields(aligned)
    assert set(results) == set(ocr.FIELDS)
    assert "ROBLE" in results["nombre"].upper()
    assert results["id_arbol"] == "4217"
    assert annotated.shape == aligned.shape


def test_roi_outside_image_is_empty():
    small = np.full((50, 50, 3), 255, np.uint8)
    results, _ = ocr.extract_fields(small, {"fuera": (500, 500, 10, 10)})
    assert results == {"fuera": ""}
