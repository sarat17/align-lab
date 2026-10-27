"""ALIGN LAB: interfaz Streamlit. Toda la lógica de procesamiento vive en ``src/``."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import detectors, export, ocr, visualization
from src.pipeline import (
    AlignParams,
    ImageLoadError,
    MAX_FILE_MB,
    MAX_PIXELS,
    align_images,
    compare_detectors,
    decode_image,
)

st.set_page_config(page_title="ALIGN LAB", page_icon="🧪", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
h1 {letter-spacing: .04em; font-weight: 700;}
.subtitle {color: #9aa7b8; margin-top: -.6rem; margin-bottom: 1rem;}
[data-testid="stMetric"] {background: #151b26; border: 1px solid #263042; border-radius: 10px; padding: .6rem .9rem;}
[data-testid="stMetricLabel"] p {color: #9aa7b8; font-size: .78rem;}
.pill {display:inline-block; padding:.15rem .6rem; border-radius:999px; background:#1c2635; border:1px solid #2b3a52; color:#c9d4e3; font-size:.78rem; margin-right:.3rem;}
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


def fmt(value, digits=3, suffix=""):
    return "n/d" if value is None else f"{value:.{digits}f}{suffix}"


def load_uploaded(label, uploaded):
    """Decodifica un archivo subido y devuelve (imagen, info) o (None, None) mostrando el error."""
    if uploaded is None:
        return None, None
    try:
        return decode_image(uploaded.getvalue(), uploaded.name)
    except ImageLoadError as exc:
        st.sidebar.error(f"{label}: {exc}")
        return None, None


def describe(info):
    names = {1: "gris", 3: "BGR", 4: "BGRA"}
    return f"{info['width']}×{info['height']} px · {names.get(info['channels'], info['channels'])} · {info['size_kb']} KB"


# ----------------------------------------------------------------- sidebar
st.sidebar.title("🧪 ALIGN LAB")
st.sidebar.caption("Carga, modo y parámetros")

up_in = st.sidebar.file_uploader("Imagen de entrada", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up_in",
                                 help="La imagen que se transformará para encajar con la plantilla.")
image, info_in = load_uploaded("Entrada", up_in)


# Plantilla fija: vive dentro del proyecto, el usuario no la sube.
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
        return decode_image(f.read(), path)


try:
    template, info_tp = load_fixed_template(str(TEMPLATE_PATH))
    st.sidebar.caption(f"Plantilla fija: `assets/{TEMPLATE_PATH.name}`")
except (FileNotFoundError, ImageLoadError) as exc:
    template, info_tp = None, None
    st.sidebar.error(f"No se pudo cargar la plantilla fija: pon una imagen llamada \"plantilla\" (jpg, jpeg, png o tif) en la carpeta assets. Detalle: {exc}")

st.sidebar.divider()
mode = st.sidebar.radio("Modo", ["Automático", "Experto"], horizontal=True,
                        help="Automático usa valores iniciales razonables. Experto expone todos los parámetros.")

available = detectors.available_detectors()
params = AlignParams(detector=available[0])
if mode == "Experto":
    params.detector = st.sidebar.selectbox(
        "Detector", available,
        help="SIFT: más robusto a escala y rotación, más lento. ORB: muy rápido, menos preciso en cambios grandes.")
    if detectors.SIFT not in available:
        st.sidebar.warning("SIFT no está disponible en esta instalación de OpenCV; solo se ofrece ORB.")
    params.max_features = st.sidebar.slider("Máximo de características", 500, 20000, 5000, 500,
                                            help="Más puntos = más posibilidades de acierto, pero más lento.")
    params.ratio = st.sidebar.slider("Lowe ratio", 0.50, 0.95, 0.75, 0.01,
                                     help="Más bajo = emparejamientos más estrictos (menos, pero más fiables). Más alto = más matches y más ruido.")
    params.ransac_threshold = st.sidebar.slider("Umbral RANSAC (px)", 0.5, 20.0, 5.0, 0.5,
                                                help="Distancia máxima en píxeles para considerar un match como inlier.")
    use_subset = st.sidebar.checkbox("Usar solo un % de los mejores matches", value=False,
                                     help="Por defecto se usan todos los que pasan el ratio test. Filtrar por distancia puede descartar buenas correspondencias.")
    if use_subset:
        params.keep_percent = st.sidebar.slider("% de matches", 5, 100, 50, 5) / 100.0
else:
    st.sidebar.caption(f"Detector: **{params.detector}** · 5000 características · ratio 0.75 · RANSAC 5 px")

run = st.sidebar.button("▶ Alinear imágenes", type="primary", disabled=image is None or template is None,
                        help="Ejecuta la alineación con los parámetros actuales.")
if st.sidebar.button("Limpiar resultados", help="Borra de la sesión los resultados y las imágenes procesadas."):
    for key in ("result", "compare_rows", "run_error", "inputs", "params_used", "ocr", "ocr_error"):
        st.session_state.pop(key, None)

st.sidebar.divider()
st.sidebar.caption(
    f"Privacidad: las imágenes se procesan en memoria en el servidor donde corre esta app "
    f"(tu equipo en local; el servidor de la app si está publicada). No se guardan en disco. "
    f"Límites: {MAX_FILE_MB} MB y {MAX_PIXELS // 1_000_000} MP por imagen.")

# -------------------------------------------------------------------- main
st.title("ALIGN LAB")
st.markdown('<div class="subtitle">Laboratorio de registro geométrico: puntos clave → correspondencias → homografía RANSAC → alineación.</div>',
            unsafe_allow_html=True)

if image is None or template is None:
    st.info("Sube una imagen en la barra lateral para empezar.")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"Entrada · {describe(info_in)}")
        show_image(visualization.to_bgr(image))
    with c2:
        st.caption(f"Plantilla · {describe(info_tp)}")
        show_image(visualization.to_bgr(template))

if run and image is not None and template is not None:
    st.session_state.pop("run_error", None)
    st.session_state.pop("compare_rows", None)
    with st.spinner("Detectando, emparejando y estimando la homografía…"):
        try:
            st.session_state["result"] = align_images(image, template, params)
            st.session_state["params_used"] = asdict(params)
            st.session_state["inputs"] = (image, template)  # necesarias para la comparación SIFT/ORB
        except Exception as exc:  # los errores se explican sin traza
            st.session_state.pop("result", None)
            st.session_state["run_error"] = str(exc)
    st.session_state.pop("ocr", None)
    st.session_state.pop("ocr_error", None)
    if "result" in st.session_state:
        with st.spinner("Extrayendo texto (OCR) de los campos…"):
            try:
                st.session_state["ocr"] = ocr.extract_fields(st.session_state["result"].aligned)
            except Exception as exc:
                st.session_state["ocr_error"] = str(exc)

if "run_error" in st.session_state:
    st.error(f"No se pudo completar la alineación: {st.session_state['run_error']}")

result = st.session_state.get("result")
if result is not None:
    m = result.metrics
    st.subheader("Resultados")
    st.markdown(
        f'<span class="pill">Detector: {m["detector_used"]}</span>'
        f'<span class="pill">ratio {m["parameters"]["ratio"]}</span>'
        f'<span class="pill">RANSAC {m["parameters"]["ransac_threshold"]} px</span>'
        f'<span class="pill">{m["processing_seconds"]:.2f} s</span>',
        unsafe_allow_html=True)
    if m["detector_requested"] != m["detector_used"]:
        st.warning(f"Se pidió {m['detector_requested']} pero no está disponible; se usó {m['detector_used']}.")

    k = st.columns(6)
    k[0].metric("Keypoints entrada", m["keypoints_input"])
    k[1].metric("Keypoints plantilla", m["keypoints_template"])
    k[2].metric("Matches (ratio test)", m["matches_after_ratio_test"])
    k[3].metric("Usados en H", m["matches_used_for_homography"])
    k[4].metric("Inliers / outliers", f"{m['inliers']} / {m['outliers']}")
    k[5].metric("Tasa de inliers", f"{100 * m['inlier_rate']:.1f} %")
    k2 = st.columns(4)
    k2[0].metric("Error reproy. medio (inliers)", fmt(m["reprojection_error_inliers_px"]["mean_px"], 2, " px"))
    k2[1].metric("Error reproy. mediano", fmt(m["reprojection_error_inliers_px"]["median_px"], 2, " px"))
    sc = m["inlier_spatial_coverage"]
    k2[2].metric("Cobertura espacial de inliers", "n/d" if sc is None else f"{100 * sc:.1f} %")
    k2[3].metric("Cobertura del warp", f"{100 * m['difference_stats']['coverage_fraction']:.1f} %")
    st.caption(
        "Indicadores, no garantías: la tasa de inliers no es la probabilidad de que la alineación sea correcta. "
        "El error de reproyección es la distancia media (en píxeles de la plantilla) entre H·p_entrada y p_plantilla "
        "calculada solo sobre inliers. La cobertura espacial es el área del cierre convexo de los inliers dividida entre "
        "el área de la plantilla: un valor bajo indica puntos concentrados en una zona.")

    st.subheader("Texto extraído (OCR)")
    if "ocr_error" in st.session_state:
        st.error(f"No se pudo hacer el OCR: {st.session_state['ocr_error']}")
    elif "ocr" in st.session_state:
        ocr_results, ocr_img = st.session_state["ocr"]
        oc1, oc2 = st.columns([1, 2])
        with oc1:
            for field, text in ocr_results.items():
                st.markdown(f"**{field.upper()}**")
                st.code(text or "(sin texto detectado)", language=None)
            st.download_button("ocr.json", json.dumps(ocr_results, ensure_ascii=False, indent=2),
                               "ocr.json", "application/json")
        with oc2:
            show_image(ocr_img, "Regiones de interés sobre la imagen alineada")

    tabs = st.tabs(["Original", "Alineación", "Superposición", "Correspondencias", "Inliers RANSAC", "Diferencias"])
    with tabs[0]:
        a, b = st.columns(2)
        with a:
            show_image(result.input_bgr, "Entrada")
        with b:
            show_image(result.template_bgr, "Plantilla")
    with tabs[1]:
        show_image(result.aligned, f"Imagen alineada · {result.aligned.shape[1]}×{result.aligned.shape[0]} px")
        st.caption("Las zonas negras son áreas fuera de la cobertura de la imagen transformada.")
    with tabs[2]:
        alpha = st.slider("Opacidad de la imagen alineada", 0.0, 1.0, 0.5, 0.05, key="overlay_alpha")
        overlay_now = visualization.make_overlay(result.template_bgr, result.aligned, alpha)
        show_image(overlay_now)
    with tabs[3]:
        show_image(result.matches_img, f"{m['matches_used_for_homography']} correspondencias usadas (se dibujan hasta 300)")
    with tabs[4]:
        show_image(result.inliers_img, "Verde: inliers · Rojo: outliers descartados por RANSAC (hasta 300 de cada tipo)")
    with tabs[5]:
        heat = st.toggle("Mapa térmico", value=True, key="diff_heat")
        diff_now = result.differences if heat else result.differences_gray
        show_image(diff_now, "Diferencia absoluta (gris oscuro = sin cobertura)")
        ds = m["difference_stats"]
        st.caption(
            f"Diferencia media en píxeles válidos: {fmt(ds['mean_abs_diff'], 2)} (0–255). "
            "La iluminación, sombras, color e interpolación también generan diferencias; "
            "no es una medida universal de calidad.")

    st.subheader("Descargas")
    overlay_dl = visualization.make_overlay(result.template_bgr, result.aligned, st.session_state.get("overlay_alpha", 0.5))
    files = export.build_files(aligned=result.aligned, matches_img=result.matches_img, overlay=overlay_dl,
                               differences=result.differences, metrics=m, homography=result.H)
    d = st.columns(4)
    d[0].download_button("aligned.png", files["aligned.png"], "aligned.png", "image/png")
    d[1].download_button("matches.png", files["matches.png"], "matches.png", "image/png")
    d[2].download_button("overlay.png", files["overlay.png"], "overlay.png", "image/png")
    d[3].download_button("differences.png", files["differences.png"], "differences.png", "image/png")
    d2 = st.columns(3)
    d2[0].download_button("metrics.json", files["metrics.json"], "metrics.json", "application/json")
    d2[1].download_button("homography.json", files["homography.json"], "homography.json", "application/json")
    d2[2].download_button("Todo (ZIP)", export.zip_bytes(files), "align_lab_results.zip", "application/zip")
    with st.expander("Matriz de homografía (entrada → plantilla)"):
        st.dataframe(pd.DataFrame(result.H), hide_index=True)

    st.subheader("Comparación SIFT / ORB")
    st.caption("Ejecuta ambos detectores con los parámetros actuales. Los tiempos dependen del hardware, "
               "del tamaño de las imágenes y de la configuración.")
    if st.button("Comparar SIFT y ORB", help="Ejecuta los dos detectores sobre la misma pareja de imágenes."):
        bar = st.progress(0.0, text="Preparando…")
        base = AlignParams(**st.session_state["params_used"])
        img_in, img_tp = st.session_state["inputs"]
        st.session_state["compare_rows"] = compare_detectors(
            img_in, img_tp, base, progress=lambda f, msg: bar.progress(f, text=msg))
        bar.empty()
    rows = st.session_state.get("compare_rows")
    if rows:
        df = pd.DataFrame(rows).rename(columns={
            "detector": "Detector", "status": "Estado", "seconds": "Tiempo (s)", "keypoints_input": "KP entrada",
            "keypoints_template": "KP plantilla", "matches": "Matches", "inliers": "Inliers",
            "inlier_rate": "Tasa inliers", "reproj_mean_px": "Error reproy. (px)", "error": "Motivo del fallo"})
        st.dataframe(df, hide_index=True)

# ------------------------------------------------------------ explanations
with st.expander("¿Cómo funciona?"):
    st.markdown("""
1. **Detector:** busca puntos distintivos (esquinas, manchas) que se reconozcan aunque la imagen cambie algo de escala o ángulo. **SIFT** es más robusto; **ORB** es mucho más rápido.
2. **Descriptor:** resume el vecindario de cada punto en un vector numérico para poder compararlo con puntos de la otra imagen.
3. **Lowe ratio test:** para cada punto se buscan sus dos vecinos más parecidos en la otra imagen; solo se acepta si el mejor es claramente mejor que el segundo. Descarta emparejamientos ambiguos.
4. **Homografía:** matriz 3×3 que describe cómo una vista plana se transforma en otra por perspectiva (rotación, escala, inclinación).
5. **RANSAC:** prueba repetidamente con grupos pequeños de correspondencias y se queda con la homografía que concuerda con más puntos (*inliers*), ignorando los errores (*outliers*).
6. **Warp:** la imagen de entrada se deforma con la homografía al tamaño de la plantilla.
""")
with st.expander("Limitaciones"):
    st.markdown("""
- Una homografía modela bien **escenas planas** o vistas con perspectiva; **no** corrige deformaciones no rígidas ni escenas 3D con mucho relieve.
- Imágenes con **poca textura**, patrones repetitivos, desenfoque fuerte o grandes cambios de iluminación producen pocos o malos matches.
- Muchos inliers **no** garantizan una buena alineación: revisa también el error de reproyección, la cobertura espacial y la superposición visual.
- La diferencia de píxeles se ve afectada por iluminación, sombras, color e interpolación, y se calcula solo donde hay cobertura.
- Los tiempos varían según el hardware; la versión publicada comparte CPU y memoria limitadas.
""")
