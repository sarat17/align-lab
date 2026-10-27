"""ALIGN LAB: interfaz Streamlit. Toda la lógica de procesamiento vive en ``src/``.

Flujo para estudiantes: suben la foto de la placa de un árbol, la app la
endereza contra la plantilla fija (``assets/plantilla.*``) y lee el texto.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import streamlit as st

from src import detectors, ocr, visualization
from src.pipeline import (
    AlignParams,
    ImageLoadError,
    MAX_FILE_MB,
    MAX_PIXELS,
    align_images_robust,
    decode_image,
)

st.set_page_config(page_title="Align Lab", page_icon="🌳", layout="wide")

# ------------------------------------------------------------------ estilo
# Paleta "guía de campo": verde bosque, hoja, liquen y el amarillo de la flor del chicalá.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,800&family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');

:root {
  --bosque: #1E3B2C;
  --hoja: #2F7A4D;
  --hoja-clara: #CFE3C8;
  --liquen: #EEF4EC;
  --chicala: #F2B705;
  --papel: #FFFFFF;
}

.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea {
  font-family: 'Atkinson Hyperlegible', system-ui, sans-serif;
}
h1, h2, h3, .al-display { font-family: 'Bricolage Grotesque', system-ui, sans-serif !important; color: var(--bosque); }

.block-container { padding-top: 2rem; max-width: 1180px; }

/* ---------- cabecera ---------- */
.al-hero { display: grid; grid-template-columns: 1fr auto; gap: 1.5rem; align-items: end;
  padding-bottom: 1.4rem; border-bottom: 3px solid var(--bosque); margin-bottom: 1.6rem; }
.al-hero h1 { font-size: clamp(2.6rem, 6vw, 4.4rem); font-weight: 800; line-height: .95;
  letter-spacing: -.02em; margin: 0; padding: 0; }
.al-hero p { font-size: 1.15rem; max-width: 34ch; margin: .7rem 0 0; color: var(--bosque); }
.al-leaf { width: 92px; height: 92px; border-radius: 0 60% 0 60%; background: var(--hoja);
  position: relative; transform: rotate(-8deg); }
.al-leaf::after { content: ""; position: absolute; left: 50%; top: 12%; bottom: 12%; width: 3px;
  background: var(--liquen); transform: translateX(-50%) rotate(45deg); border-radius: 2px; }
.al-leaf::before { content: ""; position: absolute; width: 30px; height: 30px; border-radius: 50%;
  background: var(--chicala); right: -10px; top: -10px; }

/* ---------- pasos (es una secuencia real) ---------- */
.al-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 0 0 1.8rem; }
.al-step { background: var(--papel); border-radius: 14px; padding: 1rem 1.1rem; border: 1.5px solid var(--hoja-clara); }
.al-step b { display: inline-grid; place-items: center; width: 2rem; height: 2rem; border-radius: 50%;
  background: var(--bosque); color: var(--liquen); font-family: 'Bricolage Grotesque', sans-serif; margin-bottom: .5rem; }
.al-step strong { display: block; font-size: 1.05rem; color: var(--bosque); }
.al-step span { color: #3F5A49; font-size: .95rem; }

/* ---------- etiquetas de imagen ---------- */
.al-label { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.3rem;
  color: var(--bosque); margin: 0 0 .4rem; }
.al-label small { font-family: 'Atkinson Hyperlegible', sans-serif; font-weight: 400; font-size: .9rem; color: #4A6656; margin-left: .4rem; }

/* ---------- ficha del árbol: el elemento protagonista ---------- */
.al-ficha { position: relative; background: var(--hoja); color: var(--liquen); border-radius: 22px;
  padding: 2.2rem 2.4rem 2rem; margin: 1.2rem 0 1rem; box-shadow: 0 0 0 6px var(--liquen), 0 0 0 9px var(--bosque);
  animation: al-aparece .6s cubic-bezier(.2,.8,.2,1) both; }
.al-ficha .al-id { position: absolute; top: -26px; right: 28px; background: var(--chicala); color: var(--bosque);
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.6rem; min-width: 84px; height: 84px;
  border-radius: 50%; display: flex; flex-direction: column; align-items: center; justify-content: center;
  border: 4px solid var(--liquen); padding: 0 .7rem; line-height: 1; }
.al-ficha .al-id small { font-family: 'Atkinson Hyperlegible', sans-serif; font-size: .7rem; font-weight: 700; margin-bottom: .2rem; }
.al-ficha .al-nombre { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; color: #FFFFFF;
  font-size: clamp(2rem, 5vw, 3.4rem); line-height: 1; margin: 0 6rem .9rem 0; letter-spacing: -.01em; }
.al-ficha .al-info { font-size: 1.12rem; line-height: 1.55; max-width: 62ch; margin: 0 0 1.2rem; white-space: pre-line; }
.al-ficha .al-autor { font-style: italic; opacity: .9; border-top: 1px solid rgba(238,244,236,.35); padding-top: .8rem; }
.al-ficha .al-vacio { opacity: .7; font-style: italic; }
@keyframes al-aparece { from { opacity: 0; transform: translateY(14px) scale(.98); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .al-ficha { animation: none; } }

/* ---------- barra lateral y botón ---------- */
section[data-testid="stSidebar"] { border-right: 3px solid var(--bosque); }
.stButton button[kind="primary"] { background: var(--bosque); border: none; font-weight: 700; font-size: 1.05rem; padding: .7rem 1rem; }
.stButton button[kind="primary"]:hover { background: var(--hoja); }
.stButton button:focus-visible { outline: 3px solid var(--chicala); outline-offset: 2px; }

@media (max-width: 760px) {
  .block-container { padding-top: 3.2rem; }
  .al-hero { grid-template-columns: 1fr; }
  .al-leaf { display: none; }
  .al-steps { grid-template-columns: 1fr; }
  .al-ficha { padding: 2.4rem 1.4rem 1.6rem; }
  .al-ficha .al-nombre { margin-right: 0; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def show_image(img, caption=None):
    """Muestra una imagen BGR ajustada al ancho, compatible con varias versiones de Streamlit."""
    rgb = visualization.bgr_to_rgb(img)
    try:
        st.image(rgb, caption=caption, width="stretch")
    except TypeError:
        st.image(rgb, caption=caption, use_container_width=True)


def label(title, note=""):
    note_html = f"<small>{html.escape(note)}</small>" if note else ""
    st.markdown(f'<p class="al-label">{html.escape(title)}{note_html}</p>', unsafe_allow_html=True)


def load_uploaded(uploaded):
    """Decodifica un archivo subido y devuelve la imagen o None mostrando el error."""
    if uploaded is None:
        return None
    try:
        return decode_image(uploaded.getvalue(), uploaded.name)[0]
    except ImageLoadError as exc:
        st.sidebar.error(f"Tu foto: {exc}")
        return None


def ficha_html(results: dict) -> str:
    """Construye la ficha del árbol con el texto leído por OCR (escapado)."""
    def val(key):
        text = (results.get(key) or "").strip()
        return html.escape(text) if text else ""

    vacio = '<span class="al-vacio">No se pudo leer</span>'
    id_arbol = val("id_arbol") or "?"
    return f"""
<div class="al-ficha" role="region" aria-label="Ficha del árbol">
  <div class="al-id"><small>árbol</small>{id_arbol}</div>
  <p class="al-nombre">{val("nombre") or vacio}</p>
  <p class="al-info">{val("informacion") or vacio}</p>
  <div class="al-autor">{val("autor") or vacio}</div>
</div>"""


# ------------------------------------------------------- plantilla fija
ASSETS_DIR = Path(__file__).parent / "assets"
TEMPLATE_NAMES = ("plantilla", "template")  # sin importar mayúsculas
TEMPLATE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def find_template(folder: Path) -> Path:
    """Busca plantilla.<ext> (o template.<ext>) en ``assets`` con cualquier extensión de imagen."""
    if folder.is_dir():
        for name in TEMPLATE_NAMES:
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.stem.lower() == name and f.suffix.lower() in TEMPLATE_EXTS:
                    return f
    return folder / "plantilla.jpg"  # no existe: el error lo explica abajo


TEMPLATE_PATH = find_template(ASSETS_DIR)


@st.cache_data
def load_fixed_template(path: str):
    with open(path, "rb") as f:
        return decode_image(f.read(), path)[0]


# Zonas de la plantilla cuyo texto cambia por completo de una placa a otra (nombre y autor).
# Se ignoran al buscar puntos en común. Las etiquetas "Nombre científico:", "Lugar originario:"
# y "Altura del ejemplar:" sí se usan: son iguales en todas las placas y ayudan a enderezar.
TEXT_ZONES = (ocr.FIELDS["nombre"], ocr.FIELDS["autor"])


def template_mask(shape) -> np.ndarray:
    mask = np.full(shape[:2], 255, np.uint8)
    for x, y, w, h in TEXT_ZONES:
        mask[max(0, y):y + h, max(0, x):x + w] = 0
    return mask


# ----------------------------------------------------------------- sidebar
st.sidebar.markdown('<p class="al-label">Tu foto</p>', unsafe_allow_html=True)
up_in = st.sidebar.file_uploader("Foto de la placa del árbol", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up_in",
                                 help="Tómala con el celular. No importa si sale torcida o de lado.")
image = load_uploaded(up_in)

try:
    template = load_fixed_template(str(TEMPLATE_PATH))
except (FileNotFoundError, ImageLoadError) as exc:
    template = None
    st.sidebar.error("Falta la plantilla: pon una imagen llamada \"plantilla\" (jpg, jpeg, png o tif) "
                     f"en la carpeta assets. Detalle: {exc}")

available = detectors.available_detectors()
params = AlignParams(detector=available[0])
with st.sidebar.expander("Ajustes avanzados"):
    st.caption("Para experimentar. Si no sabes qué tocar, déjalos así.")
    params.detector = st.selectbox(
        "Detector de puntos", available,
        help="SIFT: más preciso con giros y cambios de tamaño. ORB: mucho más rápido.")
    params.max_features = st.slider("Máximo de puntos", 500, 20000, 5000, 500,
                                    help="Más puntos = más posibilidades de acierto, pero más lento.")
    params.ratio = st.slider("Exigencia al emparejar (Lowe ratio)", 0.50, 0.95, 0.75, 0.01,
                             help="Más bajo = parejas más seguras pero menos. Más alto = más parejas y más errores.")
    params.ransac_threshold = st.slider("Tolerancia RANSAC (px)", 0.5, 20.0, 5.0, 0.5,
                                        help="Qué tan lejos puede quedar un punto para contar como bueno.")

_run_args = dict(type="primary", disabled=image is None or template is None, key="run_btn")
try:
    run = st.sidebar.button("Enderezar y leer", width="stretch", **_run_args)
except TypeError:  # versiones de Streamlit anteriores a ``width``
    run = st.sidebar.button("Enderezar y leer", use_container_width=True, **_run_args)
if st.sidebar.button("Empezar de nuevo"):
    for key in ("result", "run_error", "ocr", "ocr_error"):
        st.session_state.pop(key, None)

st.sidebar.caption(
    f"Tu foto se procesa en memoria y no se guarda. Máximo {MAX_FILE_MB} MB y "
    f"{MAX_PIXELS // 1_000_000} megapíxeles.")

# -------------------------------------------------------------------- main
st.markdown("""
<div class="al-hero">
  <div>
    <h1>Align Lab</h1>
    <p>Toma una foto de la placa de un árbol: la enderezamos y leemos lo que dice.</p>
  </div>
  <div class="al-leaf" aria-hidden="true"></div>
</div>
""", unsafe_allow_html=True)

if run and image is not None and template is not None:
    st.session_state.pop("run_error", None)
    st.session_state.pop("ocr", None)
    st.session_state.pop("ocr_error", None)
    with st.spinner("Buscando puntos en común con la plantilla y enderezando la foto…"):
        try:
            st.session_state["result"] = align_images_robust(image, template, params,
                                                             template_mask=template_mask(template.shape))
        except Exception as exc:  # los errores se explican sin traza
            st.session_state.pop("result", None)
            st.session_state["run_error"] = str(exc)
    if "result" in st.session_state:
        with st.spinner("Leyendo el texto de la placa…"):
            try:
                st.session_state["ocr"] = ocr.extract_fields(st.session_state["result"].aligned)
            except Exception as exc:
                st.session_state["ocr_error"] = str(exc)

result = st.session_state.get("result")

if result is None:
    st.markdown("""
<div class="al-steps">
  <div class="al-step"><b>1</b><strong>Sube tu foto</strong><span>En la barra de la izquierda. Puede estar torcida o de lado.</span></div>
  <div class="al-step"><b>2</b><strong>La enderezamos</strong><span>La comparamos con una foto de referencia y la giramos hasta que encajen.</span></div>
  <div class="al-step"><b>3</b><strong>Leemos la placa</strong><span>Con la foto derecha, el OCR lee el nombre y la información del árbol.</span></div>
</div>
""", unsafe_allow_html=True)
    if "run_error" in st.session_state:
        st.error(f"No pudimos enderezar la foto: {st.session_state['run_error']} "
                 "Prueba con una foto más cercana y bien iluminada de la placa.")
    if image is not None and template is not None:
        c1, c2 = st.columns(2)
        with c1:
            label("Tu foto")
            show_image(visualization.to_bgr(image))
        with c2:
            label("Referencia", "así debe verse la placa")
            show_image(visualization.to_bgr(template))
        st.caption("Pulsa **Enderezar y leer** en la barra de la izquierda.")
    elif image is None:
        st.info("👈 Sube la foto de una placa para empezar. En el celular, abre el menú » de arriba a la izquierda.")
else:
    a, b = st.columns(2)
    with a:
        label("Antes", "tu foto original")
        show_image(result.input_bgr)
    with b:
        label("Después", "enderezada con la plantilla")
        show_image(result.aligned)

    if "ocr_error" in st.session_state:
        st.error(f"La foto quedó derecha, pero no pudimos leer el texto: {st.session_state['ocr_error']}")
    elif "ocr" in st.session_state:
        ocr_results, ocr_img = st.session_state["ocr"]
        st.markdown(ficha_html(ocr_results), unsafe_allow_html=True)
        with st.expander("Ver dónde leímos cada dato"):
            show_image(ocr_img, "Cada recuadro verde es una zona donde el OCR busca texto.")

# ------------------------------------------------------------ explicaciones
st.write("")
with st.expander("¿Cómo funciona por dentro?"):
    st.markdown("""
1. **Puntos clave:** el detector (SIFT u ORB) busca esquinas y manchas que se reconocen aunque la foto esté girada o más lejos.
2. **Emparejar:** cada punto de tu foto se compara con los de la plantilla; solo se aceptan las parejas claras (*Lowe ratio test*).
3. **Homografía:** con las parejas se calcula una matriz 3×3 que dice cómo girar, escalar e inclinar tu foto.
4. **RANSAC:** prueba muchas combinaciones y se queda con la que más parejas explica, ignorando las equivocadas.
5. **Enderezar:** la foto se transforma con esa matriz al tamaño exacto de la plantilla.
6. **OCR:** como la foto ya coincide con la plantilla, sabemos en qué recuadro está cada dato y Tesseract lo lee.
""")
with st.expander("¿Por qué a veces falla?"):
    st.markdown("""
- Si la foto está muy borrosa, oscura o muy lejos, no hay suficientes puntos para emparejar.
- Solo funciona con objetos planos, como una placa. Un objeto con relieve no se puede enderezar bien.
- Si el texto no quedó dentro de los recuadros verdes, el OCR no lo leerá bien.
- El OCR puede confundir letras con tildes o fuentes decorativas.
""")
