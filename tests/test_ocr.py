import cv2
import numpy as np
import pytest

from src import ocr
from src.pipeline import AlignParams, align_images

needs_tesseract = pytest.mark.skipif(not ocr.ocr_available(), reason="Tesseract no está instalado")


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


@needs_tesseract
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


@needs_tesseract
def test_roi_outside_image_is_empty():
    small = np.full((50, 50, 3), 255, np.uint8)
    results, _ = ocr.extract_fields(small, {"fuera": (500, 500, 10, 10)})
    assert results == {"fuera": ""}


# --- limpieza del texto (no necesita Tesseract) -------------------------------
from src.ocr import _clean_words  # noqa: E402


def _w(*words, conf=90.0):
    return [(w, conf) for w in words]


def test_screws_read_as_letters_are_removed():
    line = _w("e", "Lugar", "originario:", "Australia", "Y")
    assert _clean_words(line, strict=True) == "Lugar originario: Australia"
    assert _clean_words(_w("Lugar", "originario:", "América", "NO"), strict=True) == "Lugar originario: América"


def test_numbers_inside_lines_are_kept():
    assert _clean_words(_w("Altura", "del", "ejemplar:", "6", "metros", "|"), strict=True) == "Altura del ejemplar: 6 metros"


def test_labels_are_normalized():
    assert _clean_words(_w("Nombre.científico:", "Eucalyptus", "cinerea"), strict=True) == \
        "Nombre científico: Eucalyptus cinerea"
    assert _clean_words(_w("Lggar", "originario:", "México"), strict=True) == "Lugar originario: México"


def test_doubtful_place_is_corrected_only_in_origin_line():
    line = [("Lugar", 95), ("originario:", 95), ("Nueva", 96), ("Zelqgia", 0)]
    assert _clean_words(line, strict=True) == "Lugar originario: Nueva Zelanda"
    # "científico" con baja confianza no debe convertirse en un lugar
    line = [("Nombre", 93), ("científico:", 58), ("Pinus", 93), ("patula", 92)]
    assert _clean_words(line, strict=True) == "Nombre científico: Pinus patula"


def test_word_after_nueva_only_matches_nueva_places():
    for misread in ("Venezuela", "Zelqzuela", "Zelaya", "Zelanda"):
        line = [("Lugar", 95), ("originario:", 95), ("Nueva", 96), (misread, 90)]
        assert _clean_words(line, strict=True) == "Lugar originario: Nueva Zelanda"
    line = [("Lugar", 95), ("originario:", 95), ("Nueva", 96), ("Guinea", 95)]
    assert _clean_words(line, strict=True) == "Lugar originario: Nueva Guinea"
    assert _clean_words(_w("Lugar", "originario:", "Venezuela"), strict=True) == "Lugar originario: Venezuela"


def test_screw_read_as_number_at_line_start_is_removed():
    assert _clean_words(_w("7", "Altura", "del", "ejemplar:", "14", "metros"), strict=True) == \
        "Altura del ejemplar: 14 metros"


def test_origin_label_and_nueva_are_rebuilt():
    line = [("pgonr", 40), ("originario:", 90), ("África", 95), ("Oriental", 95), ("-", 90),
            ("Sudáfrica", 95), ("-", 90), ("Nueva", 96), ("Ze", 50), ("LON", 40)]
    assert _clean_words(line, strict=True) == \
        "Lugar originario: África Oriental - Sudáfrica - Nueva Zelanda"
    line = [("originario:", 90), ("Nueva", 96), ("4", 60)]
    assert _clean_words(line, strict=True) == "Lugar originario: Nueva Zelanda"
    assert _clean_words(_w("Lugar", "originario:", "México", "4"), strict=True) == "Lugar originario: México"


def test_split_name_is_joined():
    from src.ocr import _join_split_word
    assert _join_split_word("Cip rés") == "Ciprés"
    assert _join_split_word("Pino candelabro") == "Pino candelabro"
    assert _join_split_word("Pino de agua") == "Pino de agua"
    assert _join_split_word("Ají dulce") == "Ají dulce"


def test_badly_read_label_falls_back_to_closest():
    assert _clean_words(_w("Lugar", "orpinerlo:", "Australia"), strict=True) == "Lugar originario: Australia"
    assert _clean_words(_w("Altura", "del", "ejempiar:", "8", "metros"), strict=True) == "Altura del ejemplar: 8 metros"
