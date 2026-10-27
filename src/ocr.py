"""Extracción de texto (OCR) sobre la imagen ya alineada con la plantilla.

Adaptado de ``ocr_form.py``. Como la imagen alineada tiene el tamaño y el
sistema de coordenadas de la plantilla (1354×998), las regiones de interés se
expresan en píxeles de la plantilla.

Mejoras para fotos reales de las placas:
- Se corrigen sombras y reflejos antes de leer.
- El número del árbol se lee dentro del cuadro verde, que se localiza por color.
- Los tornillos de la placa, que el OCR confunde con letras sueltas, se descartan.
- Las etiquetas fijas ("Nombre científico:", "Lugar originario:", "Altura del
  ejemplar:") se normalizan, y las palabras dudosas del lugar de origen se
  comparan con una lista de países y regiones.
"""

from __future__ import annotations

import difflib
from collections import Counter
import os
import re
import shutil

import cv2
import numpy as np

# Cada campo: (x, y, w, h) en píxeles de la plantilla (1354×998).
FIELDS: dict[str, tuple[int, int, int, int]] = {
    # Nombre y autor van centrados y su largo cambia de un árbol a otro: casi todo el ancho.
    "nombre": (150, 235, 1050, 135),
    # Casi todo el ancho: hay líneas largas ("África Oriental - Sudáfrica - Nueva Zelanda").
    # Los tornillos quedan dentro y se filtran después.
    "informacion": (40, 360, 1280, 240),
    "autor": (100, 630, 1154, 140),
    "id_arbol": (10, 30, 250, 190),
}

NUMERIC_FIELDS = {"id_arbol"}
SINGLE_LINE_FIELDS = {"nombre", "autor"}  # una sola línea de texto: --psm 7

# Etiquetas fijas que aparecen al inicio de las líneas de información.
# Basta con reconocer la palabra clave ("cient…", "orig…", "ejem…"): lo que haya antes, aunque
# el OCR lo lea mal ("pgonr originario"), se reemplaza por la etiqueta correcta.
_LABELS = (
    (re.compile(r"^.{0,12}?cien\w*\W*", re.I), "Nombre científico: "),
    (re.compile(r"^.{0,12}?orig\w*\W*", re.I), "Lugar originario: "),
    (re.compile(r"^.{0,16}?ejem\w*\W*", re.I), "Altura del ejemplar: "),
)

# Países y regiones (en español) para corregir palabras dudosas del lugar de origen.
_PLACES = """
África Oriental Occidental Central Sudáfrica Norte Sur América Europa Asia Oceanía Australia
Nueva Zelanda Guinea Caledonia Madagascar Mediterráneo Himalaya Amazonía Andes Caribe Antillas
Colombia México Brasil Perú Ecuador Venezuela Bolivia Chile Argentina Paraguay Uruguay Panamá
Costa Rica Nicaragua Honduras Guatemala Salvador Belice Cuba Jamaica Haití Dominicana
Estados Unidos Canadá España Portugal Francia Italia Grecia Turquía Alemania Inglaterra
China Japón Corea India Indonesia Filipinas Malasia Tailandia Vietnam Birmania Sri Lanka
Egipto Marruecos Etiopía Kenia Tanzania Mozambique Angola Congo Nigeria Senegal Camerún
Arabia Irán Líbano Siria Israel Pakistán Afganistán Nepal Tropical Subtropical Templada
""".split()


class OCRUnavailable(RuntimeError):
    """Tesseract o pytesseract no están instalados."""


# Rutas habituales del instalador de Tesseract en Windows.
_WINDOWS_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


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


def _text_lang(pytesseract) -> str:
    """Usa español (lee tildes y ñ) si está instalado; si no, inglés."""
    try:
        langs = set(pytesseract.get_languages(config=""))
    except Exception:
        return "eng"
    return "spa" if "spa" in langs else "eng"


def ocr_available() -> bool:
    try:
        _get_pytesseract()
        return True
    except OCRUnavailable:
        return False


# ------------------------------------------------------------ preprocesado
def _prepare_text(gray: np.ndarray) -> np.ndarray:
    """Quita sombras y reflejos (divide por el fondo estimado), amplía ×2 y binariza con Otsu."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    flat = cv2.divide(gray, background, scale=255)
    flat = cv2.resize(flat, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh


def _isolate_digits(binary: np.ndarray, height: int) -> np.ndarray | None:
    """Deja solo los dígitos (manchas negras de altura razonable que no tocan el borde),
    los recorta y los escala a ``height`` píxeles: Tesseract lee mal los números muy grandes."""
    H, W = binary.shape
    n, labels, st, _ = cv2.connectedComponentsWithStats(255 - binary)
    keep = [k for k in range(1, n)
            if 0.25 * H <= st[k, 3] <= 0.9 * H and st[k, 2] <= 0.5 * W
            and st[k, 0] > 0 and st[k, 1] > 0 and st[k, 0] + st[k, 2] < W and st[k, 1] + st[k, 3] < H]
    if not keep:
        return None
    x0 = min(st[k, 0] for k in keep)
    y0 = min(st[k, 1] for k in keep)
    x1 = max(st[k, 0] + st[k, 2] for k in keep)
    y1 = max(st[k, 1] + st[k, 3] for k in keep)
    digits = np.where(np.isin(labels, keep), 0, 255).astype(np.uint8)[y0:y1, x0:x1]
    f = height / digits.shape[0]
    digits = cv2.resize(digits, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    _, digits = cv2.threshold(digits, 127, 255, cv2.THRESH_BINARY)
    return cv2.copyMakeBorder(digits, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def _read_number(pytesseract, roi_bgr: np.ndarray) -> str:
    """Lee el número blanco dentro del cuadro verde.

    El 5, el 6 y el 9 se confunden con facilidad, así que los dígitos se aíslan, se llevan a
    varios tamaños y gana la lectura más repetida. Si no hay cuadro verde, usa toda la región.
    """
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (30, 60, 60), (90, 255, 255))  # verde
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[i]
        if area > 0.05 * roi_bgr.shape[0] * roi_bgr.shape[1] and w > 8 and h > 8:
            gray = cv2.cvtColor(roi_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            votes: list[str] = []
            for scale in (2, 3):
                big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                _, binary = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                for height in (32, 40, 50, 64):
                    digits = _isolate_digits(binary, height)
                    if digits is not None:
                        text = pytesseract.image_to_string(digits, config=config).strip()
                        if text:
                            votes.append(text)
            if votes:
                # La más repetida; en empate, la más larga (un dígito perdido es el error típico).
                counts = Counter(votes)
                return max(counts, key=lambda v: (counts[v], len(v)))
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return pytesseract.image_to_string(t, config=config).strip()


# ------------------------------------------------------------- limpieza
def _alnum(word: str) -> str:
    return re.sub(r"[^\w]", "", word)


# Lugares que empiezan por "Nueva": la palabra siguiente solo se compara con estas.
_NUEVA = ("Zelanda", "Guinea", "Caledonia", "Granada", "Escocia", "Gales del Sur")


def _fix_place(word: str, previous: str = "") -> str:
    """Corrige una palabra dudosa con la lista de lugares (conserva la puntuación final).

    Después de "Nueva" solo se aceptan los lugares de ``_NUEVA`` (evita "Nueva Venezuela").
    """
    core = word.rstrip(",.;:")
    if core in _NUEVA or (core in _PLACES and _alnum(previous).lower() != "nueva"):
        return word
    if _alnum(previous).lower() == "nueva":
        # Parecido + misma letra inicial + leve preferencia por los más comunes (orden de la lista).
        def score(i_cand):
            i, cand = i_cand
            same_start = core[:1].lower() == cand[:1].lower()
            return difflib.SequenceMatcher(None, core.lower(), cand.lower()).ratio() + 0.3 * same_start \
                + 0.03 * (len(_NUEVA) - i)
        best = max(enumerate(_NUEVA), key=score)[1]
        match = [best]
    else:
        match = difflib.get_close_matches(core, _PLACES, n=1, cutoff=0.5)
    return match[0] + word[len(core):] if match else word


def _fix_origin_line(kept: list[list]) -> list[list]:
    """Corrige los lugares de la línea "Lugar originario".

    - Lo que sigue a "Nueva" (hasta el siguiente guion) se junta y se corrige como un solo lugar:
      "Nueva Ze LON", "Nueva 4" o "Nueva" sola → "Nueva Zelanda".
    - Las demás palabras dudosas se comparan con la lista de lugares.
    - Al final de un lugar no van números: se quitan.
    """
    start = next((i + 1 for i, (w, _) in enumerate(kept) if "orig" in w.lower()), 0)
    head, places = kept[:start], kept[start:]
    out: list[list] = []
    i = 0
    while i < len(places):
        word, conf = places[i]
        if _alnum(word).lower() == "nueva":
            j = i + 1
            while j < len(places) and places[j][0] not in ("-", "–"):
                j += 1
            rest = "".join(_alnum(w) for w, _ in places[i + 1:j])
            out.append(["Nueva", conf])
            out.append([_fix_place(rest or "Zelanda", "Nueva"), conf])
            i = j
            continue
        if len(_alnum(word)) >= 4 and conf < 60:
            word = _fix_place(word, out[-1][0] if out else "")
        out.append([word, conf])
        i += 1
    while out and (_alnum(out[-1][0]).isdigit() or out[-1][0] in ("-", "–")):
        out.pop()
    return head + out


def _clean_words(words: list[tuple[str, float]], strict: bool = False) -> str:
    """Arma una línea a partir de (palabra, confianza) descartando la basura del OCR.

    - Signos sin letras (salvo guiones entre palabras) y palabras de 1-2 caracteres con baja confianza.
    - Letras sueltas al inicio o al final: casi siempre son los tornillos de la placa.
      Con ``strict`` (líneas de información) también se quitan restos de 2 caracteres en los bordes.
    - En la línea "Lugar originario", las palabras dudosas se corrigen con la lista de lugares.
    """
    kept = []
    for word, conf in words:
        word = word.replace("|", "").strip()
        a = _alnum(word)
        if not word:
            continue
        if not a and word not in ("-", "–"):
            continue
        if len(a) <= 2 and conf < 40 and not a.isdigit():
            continue
        kept.append([word, conf])

    edge = 2 if strict else 1

    def is_junk(word: str) -> bool:
        a = _alnum(word)
        return len(a) <= edge and not a.isdigit()

    while kept and is_junk(kept[0][0]):
        kept.pop(0)
    if strict:  # las líneas de información empiezan con una etiqueta: nada de números o signos antes
        while kept and not re.match(r"[^\W\d_]", kept[0][0]):
            kept.pop(0)
    while kept and is_junk(kept[-1][0]):
        kept.pop()

    line = " ".join(w for w, _ in kept)
    if strict and _LABELS[1][0].match(line):  # "Lugar originario: ..."
        kept = _fix_origin_line(kept)
        line = " ".join(w for w, _ in kept)

    line = re.sub(r"^Nombre\W+cient", "Nombre cient", line)
    for pattern, label in _LABELS:
        if pattern.match(line):
            return pattern.sub(label, line, count=1).strip()
    if strict:  # respaldo: etiqueta muy deformada ("Lugar orpinerlo: ...") → la más parecida
        head, sep, tail = line.partition(":")
        if sep and len(head) <= 26:
            labels = [lab for _, lab in _LABELS]
            best = max(labels, key=lambda lab: difflib.SequenceMatcher(None, head.lower(), lab[:-2].lower()).ratio())
            if difflib.SequenceMatcher(None, head.lower(), best[:-2].lower()).ratio() >= 0.6:
                line = best + tail.strip()
                if best.startswith("Lugar"):
                    words = [[w, 50.0] for w in line.split()]
                    line = " ".join(w for w, _ in _fix_origin_line(words))
    return line


_SHORT_WORDS = {"de", "del", "la", "el", "los", "las", "y", "e", "o", "u", "en"}


def _join_split_word(line: str) -> str:
    """Une una palabra partida por el OCR en dos pedazos cortos: "Cip rés" → "Ciprés"."""
    words = line.split()
    out: list[str] = []
    for w in words:
        if (out and len(_alnum(out[-1])) <= 3 and len(_alnum(w)) <= 3 and w[:1].islower()
                and w.lower() not in _SHORT_WORDS and out[-1].lower() not in _SHORT_WORDS):
            out[-1] += w
        else:
            out.append(w)
    return " ".join(out)


def _read_lines(pytesseract, image: np.ndarray, lang: str, psm: int, strict: bool = False) -> str:
    data = pytesseract.image_to_data(image, lang=lang, config=f"--psm {psm}",
                                     output_type=pytesseract.Output.DICT)
    lines: dict[tuple, list] = {}
    for i, word in enumerate(data["text"]):
        if word and word.strip():
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append((word.strip(), float(data["conf"][i])))
    out = []
    for words in lines.values():
        line = _clean_words(words, strict=strict)
        if len(_alnum(line)) >= 3:
            out.append(line)
    return "\n".join(out)


# Se mantiene por compatibilidad con versiones anteriores del proyecto.
def _clean_text(text: str) -> str:
    return "\n".join(
        line for line in (_clean_words([(w, 100.0) for w in ln.split()]) for ln in text.splitlines())
        if len(_alnum(line)) >= 3
    )


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
    lang = _text_lang(pytesseract)
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

        if field in NUMERIC_FIELDS:
            text = _read_number(pytesseract, roi)
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            psm = 7 if field in SINGLE_LINE_FIELDS else 6
            text = _read_lines(pytesseract, _prepare_text(gray), lang, psm, strict=(field == "informacion"))
            if field == "nombre":
                text = _join_split_word(text)

        results[field] = text
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(annotated, field, (x0, max(20, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return results, annotated
